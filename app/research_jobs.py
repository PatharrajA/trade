from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime

from sqlalchemy.orm import sessionmaker

from .models import ResearchSyncRun
from .providers import BrokerClient
from .research import sync_research_data


class ResearchJobService:
    def __init__(self, brokers: dict[str, BrokerClient], session_factory: sessionmaker) -> None:
        self.brokers = brokers
        self.session_factory = session_factory
        self.tasks: set[asyncio.Task] = set()

    def submit(self, run_id: int) -> None:
        task = asyncio.create_task(self._run(run_id), name=f"research-sync-{run_id}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run(self, run_id: int) -> None:
        with self.session_factory() as session:
            run = session.get(ResearchSyncRun, run_id)
            if run is None:
                return
            configuration = json.loads(run.configuration_json)
            run.status = "RUNNING"
            run.progress_percent = 1
            run.current_step = "Connecting to historical provider"
            session.commit()

        async def update(completed: int, total: int, message: str) -> None:
            with self.session_factory() as progress_session:
                current = progress_session.get(ResearchSyncRun, run_id)
                if current is None:
                    return
                current.progress_percent = min(99, int(completed / max(total, 1) * 100))
                current.current_step = message
                progress_session.commit()

        try:
            provider = str(configuration["provider"])
            result = await sync_research_data(
                self.brokers[provider],
                self.session_factory,
                list(configuration["indices"]),
                date.fromisoformat(configuration["start_date"]),
                date.fromisoformat(configuration["end_date"]),
                include_options=bool(configuration["include_options"]),
                include_one_minute=bool(configuration.get("include_one_minute", False)),
                progress=update,
            )
            with self.session_factory() as session:
                run = session.get(ResearchSyncRun, run_id)
                if run is None:
                    return
                run.status = "PARTIAL" if result.incomplete_reasons else "COMPLETE"
                run.progress_percent = 100
                run.current_step = (
                    "Download incomplete; submit the same range to resume"
                    if result.incomplete_reasons
                    else "Download complete"
                )
                run.result_json = json.dumps(
                    {
                        "spot": result.underlying_candles,
                        "futures": result.futures_candles,
                        "options": result.option_candles,
                        "future_contracts": result.futures_contracts,
                        "option_contracts": result.option_contracts,
                        "warnings": result.warnings,
                        "incomplete_reasons": result.incomplete_reasons,
                    }
                )
                run.error_message = "; ".join(result.incomplete_reasons) or None
                run.completed_at = datetime.now(UTC)
                session.commit()
        except asyncio.CancelledError:
            with self.session_factory() as session:
                run = session.get(ResearchSyncRun, run_id)
                if run:
                    run.status = "INTERRUPTED"
                    run.current_step = "Application stopped; submit the same range to resume"
                    session.commit()
            raise
        except Exception as exc:
            with self.session_factory() as session:
                run = session.get(ResearchSyncRun, run_id)
                if run:
                    run.status = "FAILED"
                    run.error_message = str(exc)
                    run.current_step = "Download failed; resubmission resumes from stored candles"
                    run.completed_at = datetime.now(UTC)
                    session.commit()

    async def shutdown(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
