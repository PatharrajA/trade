from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, ResearchSyncRun
from app.research import SyncResult
from app.research_jobs import ResearchJobService


@pytest.mark.asyncio
async def test_incomplete_required_data_is_marked_partial(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'research.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        run = ResearchSyncRun(
            provider="GROWW",
            configuration_json=json.dumps(
                {
                    "provider": "GROWW",
                    "indices": ["NIFTY"],
                    "start_date": "2023-01-01",
                    "end_date": "2025-12-31",
                    "include_options": False,
                    "include_one_minute": False,
                }
            ),
        )
        session.add(run)
        session.commit()
        run_id = run.id

    async def incomplete_sync(*args, **kwargs):
        return SyncResult(
            underlying_candles=58_000,
            futures_candles=0,
            option_candles=0,
            futures_contracts=0,
            option_contracts=0,
            warnings=["Groww request failed (429)"],
            incomplete_reasons=["NIFTY: no verified GROWW futures candles"],
        )

    monkeypatch.setattr("app.research_jobs.sync_research_data", incomplete_sync)
    service = ResearchJobService({"GROWW": object()}, factory)
    await service._run(run_id)

    with factory() as session:
        saved = session.get(ResearchSyncRun, run_id)
        assert saved is not None
        assert saved.status == "PARTIAL"
        assert saved.progress_percent == 100
        assert "submit the same range to resume" in saved.current_step
        assert "no verified GROWW futures" in saved.error_message
        assert json.loads(saved.result_json)["incomplete_reasons"]

    engine.dispose()
