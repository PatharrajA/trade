from __future__ import annotations

import asyncio
import csv
import io
import json
from contextlib import asynccontextmanager, suppress
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .backtest import BacktestService, setup_replay_quality_issues, trades_csv
from .config import config
from .dashboard import backtests_context, dashboard_context, orders_context, system_context, telegram_context
from .database import SessionLocal, database_health, get_session, init_db, settings_dict
from .groww import GrowwClient
from .model_registry import activate_candidate, register_candidate
from .mode_control import change_mode, reconcile
from .models import BacktestRun, BacktestTrade, Candle, Instrument, ModelArtifact, Order, ResearchSyncRun, RuntimeSetting, Setting, ShadowObservation, SystemEvent, Trade
from .providers import provider_name
from .operations import backup_status, file_permission_status
from .research_jobs import ResearchJobService
from .runtime import RuntimeService
from .setup_quality import SETUP_INDICES, SetupQualityModel, inspect_setup_training_data, train_setup_quality_models
from .shadow_quality import shadow_report
from .telegram import TelegramWorker, enqueue
from .training import load_model_metadata, train_regime_model
from .upstox import UpstoxClient

templates = Jinja2Templates(directory=str(config.root / "templates"))


def money(value) -> str:
    number = Decimal(value or 0)
    sign = "+" if number > 0 else ""
    return f"{sign}₹{number:,.2f}"


templates.env.filters["money"] = money
market_upstox = UpstoxClient(config.upstox_analytics_token_file, credential_name="Upstox Analytics")
trading_upstox = UpstoxClient(config.upstox_trading_token_file, credential_name="Upstox Trading")
groww = GrowwClient(config.groww_token_file)
market_brokers = {"UPSTOX": market_upstox, "GROWW": groww}
trading_brokers = {"UPSTOX": trading_upstox, "GROWW": groww}
runtime_service = RuntimeService(market_brokers, trading_brokers, SessionLocal)
telegram_worker = TelegramWorker(SessionLocal)
backtest_service = BacktestService(config.database_url, config.model_path)
research_service = ResearchJobService(market_brokers, SessionLocal)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    tasks: list[asyncio.Task] = []
    with SessionLocal() as session:
        for interrupted in session.query(ResearchSyncRun).filter(ResearchSyncRun.status.in_(["QUEUED", "RUNNING"])):
            interrupted.status = "INTERRUPTED"
            interrupted.current_step = "Application restarted; submit the same range to resume"
        runtime = session.get(RuntimeSetting, 1)
        startup_settings = settings_dict(session)
        if startup_settings.get("telegram_enabled") == "true":
            enqueue(
                session,
                "GROUP",
                "STARTUP",
                f"NiftyPilot started · mode {runtime.execution_mode if runtime else 'UNKNOWN'} · entries paused {runtime.entries_paused if runtime else True}",
                f"startup:{date.today().isoformat()}",
            )
        if runtime and runtime.execution_mode == "LIVE":
            runtime.entries_paused = True
            runtime.change_reason = "Live entries require confirmation after application restart"
            enqueue(session, "OWNER", "LIVE_RECONFIRM_REQUIRED", runtime.change_reason, f"live-reconfirm:{datetime.now().isoformat()}")
            try:
                provider = provider_name(settings_dict(session).get("active_broker"))
                market_broker, trading_broker = market_brokers[provider], trading_brokers[provider]
                market_ok, market_detail = await market_broker.market_data_health()
                trading_ok, trading_detail = await trading_broker.health()
                matched, reconcile_detail = await reconcile(session, trading_broker)
                detail = "; ".join(part for passed, part in ((market_ok, market_detail), (trading_ok, trading_detail), (matched, reconcile_detail)) if not passed)
                matched = market_ok and trading_ok and matched
            except Exception as exc:
                matched, detail = False, str(exc)
            if not matched:
                runtime.entries_paused = True
                session.add(SystemEvent(severity="CRITICAL", component="STARTUP", event_type="RECONCILIATION_FAILED", message=detail))
                enqueue(session, "OWNER", "RECONCILIATION_FAILED", f"Live entries paused: {detail}", f"startup-reconcile:{datetime.now().date()}")
        session.commit()
    if config.trading_loop_enabled:
        tasks.append(asyncio.create_task(runtime_service.run(), name="trading-loop"))
    tasks.append(asyncio.create_task(telegram_worker.run(), name="telegram-outbox"))
    try:
        yield
    finally:
        runtime_service.stop()
        telegram_worker.stop()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        await research_service.shutdown()
        backtest_service.shutdown()
        await market_upstox.client.aclose()
        await trading_upstox.client.aclose()
        await groww.client.aclose()
        await telegram_worker.client.aclose()


app = FastAPI(title="NiftyPilot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.root / "static")), name="static")


@app.websocket("/ws")
async def status_socket(websocket: WebSocket) -> None:
    """Keep local status clients connected without exposing trading controls."""
    await websocket.accept()
    await websocket.send_json({"type": "connected"})
    try:
        while True:
            if await websocket.receive_text() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


def page(request: Request, name: str, context: dict, *, status_code: int = 200):
    return templates.TemplateResponse(name=name, request=request, context={"active": name.split(".")[0], **context}, status_code=status_code)


def flash_redirect(url: str, message: str, ok: bool = True) -> RedirectResponse:
    joiner = "&" if "?" in url else "?"
    query = urlencode({"message": message, "ok": "1" if ok else "0"})
    return RedirectResponse(f"{url}{joiner}{query}", status_code=303)


def selected_brokers(session: Session):
    provider = provider_name(settings_dict(session).get("active_broker"))
    return provider, market_brokers[provider], trading_brokers[provider]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/overview")


@app.get("/overview", response_class=HTMLResponse)
def overview(request: Request, message: str = "", ok: bool = True, session: Session = Depends(get_session)):
    return page(request, "overview.html", {**dashboard_context(session, runtime_service), "message": message, "message_ok": ok})


@app.post("/mode")
async def set_mode(target: str = Form(...), confirmation: str = Form(""), session: Session = Depends(get_session)):
    _, market_broker, trading_broker = selected_brokers(session)
    app_settings = settings_dict(session)
    active_provider = provider_name(app_settings.get("active_broker"))
    setup_execution_model = SetupQualityModel(
        Path(app_settings.get("setup_execution_model_path") or "/nonexistent")
    )
    setup_execution_available = (
        app_settings.get("setup_execution_enabled") == "true" and setup_execution_model.available
    )
    paper_sessions = session.scalar(
        select(func.count(func.distinct(func.date(Trade.exit_time)))).where(
            Trade.mode == "PAPER", Trade.status == "CLOSED"
        )
    ) or 0
    latest_future = session.scalar(
        select(func.max(Candle.timestamp)).where(
            Candle.provider == active_provider,
            Candle.data_role == "FUTURE",
            Candle.timeframe == "5minute",
            Candle.source_verified.is_(True),
        )
    )
    if latest_future is not None:
        future_stamp = latest_future.replace(tzinfo=UTC) if latest_future.tzinfo is None else latest_future.astimezone(UTC)
        feed_fresh = (datetime.now(UTC) - future_stamp).total_seconds() <= 15 * 60
    else:
        feed_fresh = False
    success, message = await change_mode(
        session,
        market_broker,
        trading_broker,
        target,
        confirmation,
        model_available=setup_execution_available,
        paper_sessions=int(paper_sessions),
        required_paper_sessions=int(app_settings.get("required_paper_sessions", "20")),
        enforce_paper_sessions=app_settings.get("enforce_paper_session_requirement", "true") == "true",
        static_ip_verified=app_settings.get("static_ip_verified", "false") == "true",
        feed_fresh=feed_fresh,
    )
    return flash_redirect("/overview", message, success)


@app.post("/stop")
async def stop_trading(session: Session = Depends(get_session)):
    _, market_broker, trading_broker = selected_brokers(session)
    success, message = await change_mode(session, market_broker, trading_broker, "STOPPED")
    return flash_redirect("/overview", message, success)


@app.post("/entries/pause")
def pause_entries(session: Session = Depends(get_session)):
    runtime = session.get(RuntimeSetting, 1)
    runtime.entries_paused = True
    runtime.change_reason = "New entries paused manually"
    runtime.updated_at = datetime.now(UTC)
    session.add(SystemEvent(component="RISK", event_type="ENTRIES_PAUSED", message=runtime.change_reason))
    enqueue(session, "GROUP", "ENTRIES_PAUSED", runtime.change_reason, f"entries-paused:{runtime.updated_at.isoformat()}")
    session.commit()
    return flash_redirect("/overview", "New entries paused. Existing positions are still monitored.")


@app.post("/trades/{trade_id}/exit")
async def manual_exit_trade(
    trade_id: int,
    confirmation: str = Form(""),
    session: Session = Depends(get_session),
):
    trade = session.get(Trade, trade_id)
    if trade is None or trade.status != "OPEN":
        return flash_redirect("/overview", "Open trade was not found", False)
    if trade.mode == "LIVE" and confirmation != "EXIT NOW":
        return flash_redirect("/overview", "Type EXIT NOW to close a live broker position", False)
    try:
        broker = market_brokers[provider_name(trade.provider)]
        quote = (await broker.quotes([trade.instrument_key])).get(trade.instrument_key)
        if quote is None:
            raise RuntimeError("Executable option quote is unavailable")
        await runtime_service.execution.exit(session, trade, quote, "MANUAL_EXIT")
    except Exception as exc:
        return flash_redirect("/overview", f"Manual exit failed: {exc}", False)
    return flash_redirect("/overview", f"{trade.mode} position closed and reconciled")


@app.get("/orders", response_class=HTMLResponse)
def orders_page(request: Request, session: Session = Depends(get_session)):
    return page(request, "orders.html", {**orders_context(session), **dashboard_context(session, runtime_service)})


@app.get("/backtests", response_class=HTMLResponse)
def backtests_page(request: Request, run_id: int | None = None, message: str = "", ok: bool = True, session: Session = Depends(get_session)):
    return page(request, "backtests.html", {**backtests_context(session, run_id), "message": message, "message_ok": ok})


@app.post("/backtests")
def start_backtest(
    indices: list[str] = Form(...),
    start_date: date = Form(...),
    end_date: date = Form(...),
    capital: Decimal = Form(Decimal("500000")),
    risk_pct: Decimal = Form(Decimal("0.25")),
    slippage_pct: Decimal = Form(Decimal("0.05")),
    historical_provider: str = Form("UPSTOX"),
    cost_model: str = Form("UPSTOX"),
    strategy_mode: str = Form("SETUP_QUALITY"),
    strike_policy: str = Form("ATM_ITM1"),
    entry_delay_bars: int = Form(1),
    minimum_option_volume: int = Form(1),
    maximum_spread_pct: Decimal = Form(Decimal("1.5")),
    setup_model_id: int | None = Form(None),
    compare_rule_only: bool = Form(False),
    session: Session = Depends(get_session),
):
    allowed = {"NIFTY", "BANK_NIFTY", "SENSEX"}
    selected = [item for item in indices if item in allowed]
    historical_provider = provider_name(settings_dict(session).get("active_broker"))
    if historical_provider == "GROWW":
        cost_model = "GROWW"
    elif cost_model not in {"UPSTOX", "UPSTOX_PLUS"}:
        cost_model = "UPSTOX"
    if historical_provider not in {"UPSTOX", "GROWW"} or cost_model not in {"UPSTOX", "GROWW", "UPSTOX_PLUS"} or strategy_mode not in {"MODEL", "SETUP_QUALITY", "RULE_ONLY"} or strike_policy not in {"ATM_ONLY", "ATM_ITM1"}:
        return flash_redirect("/backtests", "Unknown data provider or cost model", False)
    if not selected or start_date > end_date or capital <= 0 or not (Decimal("0") < risk_pct <= Decimal("2")) or slippage_pct < 0 or entry_delay_bars < 1 or minimum_option_volume < 0 or maximum_spread_pct <= 0:
        return flash_redirect("/backtests", "Please check the backtest inputs", False)
    model_metadata = load_model_metadata(config.model_path)
    if strategy_mode == "MODEL" and not model_metadata:
        return flash_redirect("/backtests", "Train and activate a model before running a model backtest", False)
    selected_model_metadata = model_metadata
    setup_model_path = None
    if strategy_mode == "SETUP_QUALITY":
        artifact = session.get(ModelArtifact, setup_model_id) if setup_model_id else None
        if artifact is None:
            return flash_redirect("/backtests", "Select a setup-quality research model", False)
        try:
            selected_model_metadata = json.loads(artifact.metadata_json)
        except (TypeError, ValueError):
            selected_model_metadata = {}
        if selected_model_metadata.get("model_type") != "SETUP_QUALITY" or not Path(artifact.model_path).exists():
            return flash_redirect("/backtests", "The selected setup-quality model is unavailable", False)
        unsupported_indices = set(selected) - set(selected_model_metadata.get("indices") or [])
        if unsupported_indices:
            return flash_redirect(
                "/backtests",
                "The selected setup-quality model does not contain: " + ", ".join(sorted(unsupported_indices)),
                False,
            )
        setup_model_path = artifact.model_path
    training_end = (
        selected_model_metadata.get("validation_end")
        if strategy_mode == "SETUP_QUALITY"
        else selected_model_metadata.get("training_end")
    )
    if strategy_mode in {"MODEL", "SETUP_QUALITY"} and training_end and start_date <= date.fromisoformat(training_end):
        return flash_redirect(
            "/backtests",
            f"Model testing must start after its training period ({training_end})",
            False,
        )
    run_config = {"indices": selected, "capital": str(capital), "risk_pct": str(risk_pct), "slippage_pct": str(slippage_pct), "historical_provider": historical_provider, "cost_model": cost_model, "strategy_mode": strategy_mode, "strike_policy": strike_policy, "entry_delay_bars": entry_delay_bars, "minimum_option_volume": minimum_option_volume, "maximum_spread_pct": str(maximum_spread_pct), "model_version": selected_model_metadata.get("version"), "feature_schema": selected_model_metadata.get("feature_schema"), "setup_model_path": setup_model_path, "compare_rule_only": compare_rule_only}
    run = BacktestRun(
        configuration_json=json.dumps(run_config),
        start_date=start_date,
        end_date=end_date,
    )
    session.add(run)
    session.flush()
    comparison_run = None
    if compare_rule_only and strategy_mode in {"MODEL", "SETUP_QUALITY"}:
        comparison_config = {**run_config, "strategy_mode": "RULE_ONLY", "compare_rule_only": False, "comparison_parent_id": run.id}
        comparison_run = BacktestRun(
            configuration_json=json.dumps(comparison_config),
            start_date=start_date,
            end_date=end_date,
        )
        session.add(comparison_run)
        session.flush()
        run_config["comparison_run_id"] = comparison_run.id
        run.configuration_json = json.dumps(run_config)
    session.commit()
    backtest_service.submit(run.id)
    if comparison_run is not None:
        backtest_service.submit(comparison_run.id)
    return RedirectResponse(f"/backtests?run_id={run.id}", status_code=303)


@app.post("/backtests/sync-data")
async def sync_backtest_data(
    indices: list[str] = Form(...),
    start_date: date = Form(...),
    end_date: date = Form(...),
    historical_provider: str = Form("UPSTOX"),
    include_options: bool = Form(False),
    include_one_minute: bool = Form(False),
    repair_only: bool = Form(False),
    session: Session = Depends(get_session),
):
    allowed = {"NIFTY", "BANK_NIFTY", "SENSEX"}
    selected = [item for item in indices if item in allowed]
    historical_provider = provider_name(settings_dict(session).get("active_broker"))
    if historical_provider not in {"UPSTOX", "GROWW"} or not selected or start_date > end_date or end_date > date.today():
        return flash_redirect("/backtests", "Please check the historical-data inputs", False)
    if (end_date - start_date).days > (2192 if historical_provider == "GROWW" else 730):
        return flash_redirect("/backtests", "The selected date range is too large for this provider", False)
    active_download = session.scalar(
        select(ResearchSyncRun.id).where(ResearchSyncRun.status.in_(["QUEUED", "RUNNING"])).limit(1)
    )
    if active_download:
        return flash_redirect("/backtests", f"Research download #{active_download} is already running", False)
    run = ResearchSyncRun(
        provider=historical_provider,
        configuration_json=json.dumps(
            {
                "provider": historical_provider,
                "indices": selected,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "include_options": include_options,
                "include_one_minute": include_one_minute,
                "repair_only": repair_only,
            }
        ),
    )
    session.add(run)
    session.commit()
    research_service.submit(run.id)
    return flash_redirect(
        "/backtests",
        f"Research {'repair' if repair_only else 'download'} #{run.id} queued. Existing candles are preserved and duplicates are skipped.",
    )


@app.get("/backtests/research/{run_id}/status", response_class=HTMLResponse)
def research_status(request: Request, run_id: int, session: Session = Depends(get_session)):
    run = session.get(ResearchSyncRun, run_id)
    if run is None:
        raise HTTPException(404, "Research download not found")
    return templates.TemplateResponse(
        name="partials/research_status.html",
        request=request,
        context={"run": run},
    )


@app.get("/backtests/contract-metadata.csv")
def pending_contract_metadata_csv(session: Session = Depends(get_session)):
    rows = session.scalars(
        select(Instrument)
        .where(Instrument.instrument_kind == "OPTION", Instrument.metadata_verified.is_(False))
        .order_by(Instrument.provider, Instrument.underlying_code, Instrument.expiry, Instrument.strike)
    ).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["provider", "instrument_key", "expiry", "lot_size", "tick_size"])
    for row in rows:
        writer.writerow([row.provider, row.instrument_key, row.expiry, "", row.tick_size])
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pending-contract-metadata.csv"},
    )


@app.post("/backtests/contract-metadata")
async def import_contract_metadata(
    metadata_file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    content = await metadata_file.read(2_000_001)
    if len(content) > 2_000_000:
        return flash_redirect("/backtests", "Metadata CSV must be smaller than 2 MB", False)
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        required = {"provider", "instrument_key", "expiry", "lot_size", "tick_size"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV columns must include: {', '.join(sorted(required))}")
        updated = skipped = 0
        for item in reader:
            provider = provider_name(item.get("provider"))
            expiry = date.fromisoformat(str(item.get("expiry", "")))
            lot_size = int(str(item.get("lot_size", "0")))
            tick_size = Decimal(str(item.get("tick_size", "0")))
            if lot_size <= 0 or tick_size <= 0:
                raise ValueError("lot_size and tick_size must be positive")
            instruments = session.scalars(
                select(Instrument).where(
                    Instrument.provider == provider,
                    Instrument.instrument_key == str(item.get("instrument_key", "")).strip(),
                    Instrument.expiry == expiry,
                    Instrument.instrument_kind == "OPTION",
                )
            ).all()
            if not instruments:
                skipped += 1
                continue
            for instrument in instruments:
                instrument.lot_size = lot_size
                instrument.tick_size = tick_size
                instrument.metadata_verified = True
                instrument.metadata_source = "CSV_IMPORT"
                updated += 1
        session.commit()
    except (UnicodeDecodeError, ValueError, ArithmeticError) as exc:
        session.rollback()
        return flash_redirect("/backtests", f"Metadata import failed: {exc}", False)
    return flash_redirect("/backtests", f"Verified {updated} contract versions; {skipped} rows did not match stored contracts")


@app.post("/backtests/train")
async def train_model(
    indices: list[str] = Form(...),
    start_date: date = Form(...),
    end_date: date = Form(...),
    historical_provider: str = Form("UPSTOX"),
    session: Session = Depends(get_session),
):
    allowed = {"NIFTY", "BANK_NIFTY", "SENSEX"}
    selected = [item for item in indices if item in allowed]
    historical_provider = provider_name(settings_dict(session).get("active_broker"))
    if historical_provider not in {"UPSTOX", "GROWW"} or not selected or start_date >= end_date or end_date > date.today():
        return flash_redirect("/backtests", "Please check the training inputs", False)
    try:
        candidate_path = config.model_path.parent / "candidates" / f"candidate-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.json"
        metadata = await asyncio.to_thread(
            train_regime_model,
            config.database_url,
            candidate_path,
            selected,
            start_date,
            end_date,
            historical_provider,
        )
        artifact = register_candidate(session, candidate_path, metadata)
        quality = metadata.get("activation_quality") or {}
        quality_message = (
            "Activation quality gate passed."
            if quality.get("ready")
            else "Activation blocked: " + "; ".join(quality.get("issues") or ["validation quality is insufficient"])
        )
        return flash_redirect(
            "/backtests",
            f"Candidate #{artifact.id} trained on {metadata['samples']} samples. {quality_message}",
        )
    except ModuleNotFoundError:
        return flash_redirect("/backtests", "XGBoost is not installed. Run: pip install -r requirements.txt", False)
    except Exception as exc:
        return flash_redirect("/backtests", f"Model training failed: {exc}", False)


@app.post("/backtests/check-setup-data", response_class=HTMLResponse)
async def check_setup_training_data(
    request: Request,
    indices: list[str] = Form(...),
    fit_start: date = Form(...),
    fit_end: date = Form(...),
    tuning_end: date = Form(...),
    holdout_end: date = Form(...),
    session: Session = Depends(get_session),
):
    selected = list(dict.fromkeys(item for item in indices if item in SETUP_INDICES))
    if not selected or not fit_start <= fit_end < tuning_end < holdout_end or holdout_end > date.today():
        return flash_redirect("/backtests", "Please check the training indices and dates", False)
    provider = provider_name(settings_dict(session).get("active_broker"))
    try:
        readiness = await asyncio.to_thread(
            inspect_setup_training_data, config.database_url,
            selected, fit_start, fit_end, tuning_end, holdout_end, provider,
        )
    except Exception as exc:
        return flash_redirect("/backtests", f"Training data check failed: {exc}", False)
    return page(request, "backtests.html", {
        **backtests_context(session), "training_readiness": readiness,
        "selected_training_indices": selected,
        "training_dates": {
            "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(),
            "tuning_end": tuning_end.isoformat(), "holdout_end": holdout_end.isoformat(),
        },
    })


@app.post("/backtests/train-setup-quality")
async def train_setup_quality_model(
    indices: list[str] = Form(...),
    fit_start: date = Form(...),
    fit_end: date = Form(...),
    tuning_end: date = Form(...),
    holdout_end: date = Form(...),
    historical_provider: str = Form("UPSTOX"),
    session: Session = Depends(get_session),
):
    selected = [item for item in indices if item in SETUP_INDICES]
    historical_provider = provider_name(settings_dict(session).get("active_broker"))
    if (
        historical_provider not in {"UPSTOX", "GROWW"}
        or not selected
        or not fit_start <= fit_end < tuning_end < holdout_end
        or holdout_end > date.today()
    ):
        return flash_redirect("/backtests", "Please check the setup-quality training periods", False)
    try:
        candidate_path = (
            config.model_path.parent
            / "candidates"
            / f"setup-candidate-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.json"
        )
        metadata = await asyncio.to_thread(
            train_setup_quality_models,
            config.database_url,
            candidate_path,
            selected,
            fit_start,
            fit_end,
            tuning_end,
            holdout_end,
            historical_provider,
        )
        artifact = register_candidate(session, candidate_path, metadata)
        research_quality = metadata.get("research_quality") or {}
        activation_quality = metadata.get("activation_quality") or {}
        if activation_quality.get("ready"):
            message = "Historical quality gate passed. Continue with option backtesting and shadow validation."
        elif research_quality.get("ready_for_option_replay"):
            message = "Research gate passed, but activation remains blocked: " + "; ".join(
                activation_quality.get("issues") or ["verified option-price labels are required"]
            )
        else:
            message = "Research gate did not pass: " + "; ".join(
                research_quality.get("issues") or ["validation quality is insufficient"]
            )
        return flash_redirect(
            "/backtests",
            f"Setup-quality candidate #{artifact.id} trained on {metadata['samples']} rule setups. {message}",
        )
    except ModuleNotFoundError:
        return flash_redirect("/backtests", "XGBoost is not installed. Run: pip install -r requirements.txt", False)
    except Exception as exc:
        return flash_redirect("/backtests", f"Setup-quality training failed: {exc}", False)


@app.post("/backtests/models/{artifact_id}/activate")
def activate_model(artifact_id: int, session: Session = Depends(get_session)):
    runtime = session.get(RuntimeSetting, 1)
    if runtime and runtime.execution_mode != "STOPPED":
        return flash_redirect("/backtests", "Stop execution before activating a model", False)
    if session.scalar(select(BacktestRun.id).where(BacktestRun.status.in_(["QUEUED", "RUNNING"])).limit(1)):
        return flash_redirect("/backtests", "Wait for queued backtests to finish before changing the active model", False)
    try:
        artifact = activate_candidate(session, artifact_id, config.model_path)
        runtime_service.reload_model()
    except Exception as exc:
        return flash_redirect("/backtests", f"Model activation failed: {exc}", False)
    return flash_redirect("/backtests", f"Model {artifact.version} is now active")


@app.post("/backtests/models/{artifact_id}/shadow")
def enable_shadow_model(artifact_id: int, session: Session = Depends(get_session)):
    artifact = session.get(ModelArtifact, artifact_id)
    if artifact is None:
        return flash_redirect("/backtests", "Setup-quality model was not found", False)
    try:
        metadata = json.loads(artifact.metadata_json)
    except (TypeError, ValueError):
        metadata = {}
    path = Path(artifact.model_path)
    model = SetupQualityModel(path)
    if metadata.get("model_type") != "SETUP_QUALITY" or not model.available:
        return flash_redirect("/backtests", "Setup-quality model bundle cannot be loaded", False)
    for key, value in (
        ("setup_shadow_model_path", str(path)),
        ("setup_shadow_enabled", "true"),
    ):
        setting = session.get(Setting, key)
        if setting is None:
            session.add(Setting(key=key, value=value))
        else:
            setting.value = value
    for existing in session.scalars(select(ModelArtifact).where(ModelArtifact.status == "SHADOW")):
        existing.status = "CANDIDATE"
    artifact.status = "SHADOW"
    session.commit()
    runtime_service.reload_shadow_model(str(path))
    return flash_redirect(
        "/backtests", f"Shadow observation enabled for {metadata.get('version', artifact.version)}; no orders will be placed"
    )


@app.post("/backtests/shadow/disable")
def disable_shadow_model(session: Session = Depends(get_session)):
    path_setting = session.get(Setting, "setup_shadow_model_path")
    if path_setting and path_setting.value:
        artifact = session.scalar(
            select(ModelArtifact).where(ModelArtifact.model_path == path_setting.value).limit(1)
        )
        if artifact and artifact.status == "SHADOW":
            artifact.status = "CANDIDATE"
    setting = session.get(Setting, "setup_shadow_enabled")
    if setting is None:
        session.add(Setting(key="setup_shadow_enabled", value="false"))
    else:
        setting.value = "false"
    session.commit()
    runtime_service.reload_shadow_model("")
    return flash_redirect("/backtests", "Shadow observation disabled")


@app.post("/backtests/models/{artifact_id}/activate-setup")
def activate_setup_model(artifact_id: int, session: Session = Depends(get_session)):
    runtime = session.get(RuntimeSetting, 1)
    if runtime and runtime.execution_mode != "STOPPED":
        return flash_redirect("/backtests", "Stop execution before activating a setup-quality model", False)
    artifact = session.get(ModelArtifact, artifact_id)
    if artifact is None:
        return flash_redirect("/backtests", "Setup-quality model was not found", False)
    try:
        metadata = json.loads(artifact.metadata_json)
    except (TypeError, ValueError):
        metadata = {}
    quality = metadata.get("activation_quality") or {}
    path = Path(artifact.model_path)
    if metadata.get("model_type") != "SETUP_QUALITY" or not SetupQualityModel(path).available:
        return flash_redirect("/backtests", "Setup-quality model bundle cannot be loaded", False)
    if not quality.get("ready"):
        return flash_redirect(
            "/backtests",
            "Setup-quality activation blocked: " + "; ".join(quality.get("issues") or ["quality gate failed"]),
            False,
        )
    replay_issues = ["Complete a post-holdout option backtest with rule-only comparison"]
    for run in session.scalars(
        select(BacktestRun).where(BacktestRun.status == "COMPLETE").order_by(BacktestRun.id.desc())
    ):
        try:
            run_config = json.loads(run.configuration_json)
            run_summary = json.loads(run.summary_json) if run.summary_json else {}
        except (TypeError, ValueError):
            continue
        if (
            run_config.get("strategy_mode") != "SETUP_QUALITY"
            or run_config.get("model_version") != metadata.get("version")
            or run_config.get("historical_provider") != provider_name(settings_dict(session).get("active_broker"))
            or set(run_config.get("indices") or []) != set(metadata.get("indices") or [])
            or run.start_date <= date.fromisoformat(metadata["validation_end"])
        ):
            continue
        comparison = session.get(BacktestRun, run_config.get("comparison_run_id")) if run_config.get("comparison_run_id") else None
        if not comparison or comparison.status != "COMPLETE":
            continue
        try:
            comparison_config = json.loads(comparison.configuration_json)
            comparison_summary = json.loads(comparison.summary_json) if comparison.summary_json else {}
        except (TypeError, ValueError):
            continue
        if comparison_config.get("comparison_parent_id") != run.id or comparison_config.get("strategy_mode") != "RULE_ONLY":
            continue
        replay_issues = setup_replay_quality_issues(run_summary, comparison_summary, metadata["indices"])
        if not replay_issues:
            break
    if replay_issues:
        return flash_redirect("/backtests", "Setup-quality replay gate blocked activation: " + "; ".join(replay_issues), False)
    shadow_rows = session.scalars(
        select(ShadowObservation).where(
            ShadowObservation.model_version == metadata.get("version"),
            ShadowObservation.provider == provider_name(settings_dict(session).get("active_broker")),
        )
    ).all()
    shadow_issues = shadow_report(shadow_rows, metadata.get("indices") or []).get("issues", [])
    if shadow_issues:
        return flash_redirect(
            "/backtests", "Setup-quality shadow gate blocked activation: " + "; ".join(shadow_issues), False
        )
    for key, value in (
        ("setup_execution_model_path", str(path)),
        ("setup_execution_enabled", "true"),
    ):
        setting = session.get(Setting, key)
        if setting is None:
            session.add(Setting(key=key, value=value))
        else:
            setting.value = value
    for existing in session.scalars(select(ModelArtifact).where(ModelArtifact.status == "ACTIVE_SETUP")):
        existing.status = "ARCHIVED"
    artifact.status = "ACTIVE_SETUP"
    artifact.activated_at = datetime.now(UTC)
    session.commit()
    return flash_redirect("/backtests", f"Setup-quality model {artifact.version} is enabled for Paper/Live")


@app.post("/backtests/setup-execution/disable")
def disable_setup_execution(session: Session = Depends(get_session)):
    runtime = session.get(RuntimeSetting, 1)
    if runtime and runtime.execution_mode != "STOPPED":
        return flash_redirect("/backtests", "Stop execution before changing the active strategy model", False)
    path_setting = session.get(Setting, "setup_execution_model_path")
    if path_setting and path_setting.value:
        artifact = session.scalar(
            select(ModelArtifact).where(ModelArtifact.model_path == path_setting.value).limit(1)
        )
        if artifact and artifact.status == "ACTIVE_SETUP":
            artifact.status = "ARCHIVED"
    setting = session.get(Setting, "setup_execution_enabled")
    if setting is None:
        session.add(Setting(key="setup_execution_enabled", value="false"))
    else:
        setting.value = "false"
    session.commit()
    return flash_redirect("/backtests", "Setup-quality Paper/Live execution disabled")


@app.get("/backtests/{run_id}/status", response_class=HTMLResponse)
def backtest_status(request: Request, run_id: int, session: Session = Depends(get_session)):
    selected = session.get(BacktestRun, run_id)
    if not selected:
        raise HTTPException(404, "Backtest not found")
    context = backtests_context(session, run_id)
    return templates.TemplateResponse(name="partials/backtest_result.html", request=request, context=context)


@app.get("/backtests/{run_id}/trades.csv")
def backtest_csv(run_id: int, session: Session = Depends(get_session)):
    if not session.get(BacktestRun, run_id):
        raise HTTPException(404, "Backtest not found")
    rows = session.scalars(select(BacktestTrade).where(BacktestTrade.run_id == run_id).order_by(BacktestTrade.entry_time)).all()
    return Response(trades_csv(rows), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="backtest-{run_id}.csv"'})


@app.get("/telegram", response_class=HTMLResponse)
def telegram_page(request: Request, message: str = "", ok: bool = True, session: Session = Depends(get_session)):
    return page(request, "telegram.html", {**telegram_context(session), "message": message, "message_ok": ok, "bot_token_configured": bool(config.telegram_bot_token)})


@app.post("/telegram/settings")
def save_telegram(enabled: bool = Form(False), daily_report: bool = Form(False), monthly_report: bool = Form(False), group_chat_id: str = Form(""), owner_chat_id: str = Form(""), session: Session = Depends(get_session)):
    values = {"telegram_enabled": str(enabled).lower(), "telegram_daily_report": str(daily_report).lower(), "telegram_monthly_report": str(monthly_report).lower(), "telegram_group_chat_id": group_chat_id.strip(), "telegram_owner_chat_id": owner_chat_id.strip()}
    for key, value in values.items():
        row = session.get(Setting, key)
        if row: row.value = value
        else: session.add(Setting(key=key, value=value))
    session.commit()
    return flash_redirect("/telegram", "Telegram settings saved")


@app.post("/telegram/test")
def test_telegram(destination: str = Form("GROUP"), session: Session = Depends(get_session)):
    if destination not in {"GROUP", "OWNER"}:
        return flash_redirect("/telegram", "Invalid destination", False)
    enqueue(session, destination, "TEST", "NiftyPilot test message: notification delivery is working.", f"test:{destination}:{datetime.now().isoformat()}")
    session.commit()
    return flash_redirect("/telegram", "Test message queued")


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, message: str = "", ok: bool = True, session: Session = Depends(get_session)):
    app_settings = settings_dict(session)
    active_broker = provider_name(app_settings.get("active_broker"))
    analytics_configured = bool(market_upstox.access_token())
    trading_configured = bool(trading_upstox.access_token())
    groww_configured = bool(groww.access_token())
    analytics_ok, analytics_detail = False, "Inactive broker"
    trading_ok, trading_detail = False, "Inactive broker"
    groww_ok, groww_detail = False, "Inactive broker"
    if active_broker == "UPSTOX":
        analytics_ok, analytics_detail = await market_upstox.market_data_health() if analytics_configured else (False, "Token missing")
        trading_ok, trading_detail = await trading_upstox.health() if trading_configured else (False, "Token missing")
    else:
        groww_ok, groww_detail = await groww.health() if groww_configured else (False, "Token missing")
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    expiry_day = now_ist.date() if now_ist.time() < time(3, 30) else now_ist.date() + timedelta(days=1)
    trading_expiry = datetime.combine(expiry_day, time(3, 30), tzinfo=ZoneInfo("Asia/Kolkata")) if trading_configured else None
    permissions_ok, permissions_detail = file_permission_status(
        [config.upstox_analytics_token_file, config.upstox_trading_token_file, config.groww_token_file]
    )
    return page(request, "settings.html", {
        **system_context(session),
        **dashboard_context(session, runtime_service),
        "message": message,
        "message_ok": ok,
        "analytics_configured": analytics_configured,
        "analytics_ok": analytics_ok,
        "analytics_detail": analytics_detail,
        "trading_configured": trading_configured,
        "trading_ok": trading_ok,
        "trading_detail": trading_detail,
        "trading_expiry": trading_expiry,
        "groww_configured": groww_configured,
        "groww_ok": groww_ok,
        "groww_detail": groww_detail,
        "active_broker": active_broker,
        "default_historical_provider": active_broker,
        "active_market_ok": analytics_ok if active_broker == "UPSTOX" else groww_ok,
        "active_trading_ok": trading_ok if active_broker == "UPSTOX" else groww_ok,
        "model_metadata": load_model_metadata(config.model_path),
        "backup": backup_status(config.root),
        "permissions_ok": permissions_ok,
        "permissions_detail": permissions_detail,
    })


@app.post("/settings/analytics-token")
async def save_analytics_token(token: str = Form(...)):
    valid, detail = await market_upstox.validate_token(token, market_data=True)
    if not valid:
        return flash_redirect("/settings", f"Analytics token not saved: {detail}", False)
    market_upstox.save_token(token)
    return flash_redirect("/settings", "Analytics token verified and saved with 0600 permissions")


@app.post("/settings/trading-token")
async def save_trading_token(token: str = Form(...)):
    valid, detail = await trading_upstox.validate_token(token, market_data=False)
    if not valid:
        return flash_redirect("/settings", f"Trading token not saved: {detail}", False)
    trading_upstox.save_token(token)
    return flash_redirect("/settings", "Daily trading token verified and saved with 0600 permissions")


@app.post("/settings/groww-token")
async def save_groww_token(token: str = Form(...)):
    valid, detail = await groww.validate_token(token, market_data=True)
    if not valid:
        return flash_redirect("/settings", f"Groww token not saved: {detail}", False)
    groww.save_token(token)
    return flash_redirect("/settings", "Groww token verified and saved with 0600 permissions")


@app.post("/settings/groww-token/remove")
def remove_groww_token(session: Session = Depends(get_session)):
    runtime = session.get(RuntimeSetting, 1)
    active_broker = provider_name(settings_dict(session).get("active_broker"))
    if active_broker == "GROWW" or (runtime and runtime.execution_mode != "STOPPED"):
        return flash_redirect(
            "/settings",
            "Stop trading and switch the active broker away from Groww before removing its token",
            False,
        )
    try:
        groww.token_file.unlink(missing_ok=True)
    except OSError as exc:
        return flash_redirect("/settings", f"Groww token could not be removed: {exc}", False)
    return flash_redirect(
        "/settings",
        "Groww token removed. Downloaded candles, the common model and backtests were retained.",
    )


@app.post("/settings/broker")
def save_broker(
    active_broker: str = Form(...),
    session: Session = Depends(get_session),
):
    try:
        active_broker = provider_name(active_broker)
    except ValueError as exc:
        return flash_redirect("/settings", str(exc), False)
    runtime = session.get(RuntimeSetting, 1)
    if runtime.execution_mode != "STOPPED":
        return flash_redirect("/settings", "Stop trading before changing the active broker", False)
    if session.scalar(select(Trade.id).where(Trade.status == "OPEN").limit(1)):
        return flash_redirect("/settings", "Close the open position before changing broker", False)
    if session.scalar(select(Order.id).where(Order.status.in_(["CREATED", "SUBMITTED", "OPEN", "PARTIAL", "PENDING"])).limit(1)):
        return flash_redirect("/settings", "Resolve pending orders before changing broker", False)
    session.get(Setting, "active_broker").value = active_broker
    # Keep the UI and runtime unambiguous: the selected broker supplies live,
    # paper, research download, model training and backtest data.
    session.get(Setting, "default_historical_provider").value = active_broker
    runtime.change_reason = f"Active broker changed to {active_broker}"
    session.add(SystemEvent(component="BROKER", event_type="BROKER_CHANGED", message=runtime.change_reason))
    enqueue(
        session,
        "GROUP",
        "BROKER_CHANGED",
        f"NiftyPilot active broker changed to {active_broker}",
        f"broker:{active_broker}:{datetime.now().isoformat()}",
    )
    session.commit()
    return flash_redirect("/settings", f"Active broker set to {active_broker}")


@app.post("/settings/risk")
def save_risk(
    paper_capital: Decimal = Form(...), risk_per_trade_pct: Decimal = Form(...), daily_loss_limit_pct: Decimal = Form(...),
    max_trades_per_day: int = Form(...), max_consecutive_losses: int = Form(...), premium_emergency_stop_pct: Decimal = Form(...),
    reward_risk: Decimal = Form(...), paper_slippage_pct: Decimal = Form(...), session: Session = Depends(get_session),
):
    values = locals().copy()
    values.pop("session")
    if paper_capital <= 0 or not (Decimal("0") < risk_per_trade_pct <= 2) or not (Decimal("0") < daily_loss_limit_pct <= 5) or max_trades_per_day < 1 or max_consecutive_losses < 1 or not (Decimal("5") <= premium_emergency_stop_pct <= 50) or reward_risk <= 0 or paper_slippage_pct < 0:
        return flash_redirect("/settings", "Risk values are outside safe limits", False)
    for key, value in values.items():
        session.get(Setting, key).value = str(value)
    session.commit()
    return flash_redirect("/settings", "Risk settings saved")


@app.post("/settings/markets")
def save_markets(indices: list[str] = Form(...), session: Session = Depends(get_session)):
    selected = [item for item in indices if item in {"NIFTY", "BANK_NIFTY", "SENSEX"}]
    if not selected:
        return flash_redirect("/settings", "Enable at least one index", False)
    session.get(Setting, "enabled_indices").value = ",".join(selected)
    session.commit()
    return flash_redirect("/settings", "Enabled indices updated")


@app.post("/settings/live-safety")
def save_live_safety(
    enforce_paper_session_requirement: bool = Form(False),
    required_paper_sessions: int = Form(...),
    static_ip_verified: bool = Form(False),
    session: Session = Depends(get_session),
):
    if not 0 <= required_paper_sessions <= 60:
        return flash_redirect("/settings", "Required paper sessions must be between 0 and 60", False)
    previous = settings_dict(session)
    values = {
        "enforce_paper_session_requirement": str(enforce_paper_session_requirement).lower(),
        "required_paper_sessions": str(required_paper_sessions),
        "static_ip_verified": str(static_ip_verified).lower(),
    }
    changed = any(previous.get(key) != value for key, value in values.items())
    for key, value in values.items():
        row = session.get(Setting, key)
        if row:
            row.value = value
        else:
            session.add(Setting(key=key, value=value))
    if changed:
        requirement = f"enabled ({required_paper_sessions} sessions)" if enforce_paper_session_requirement else "disabled"
        message = f"Paper-session Live requirement {requirement}; static-IP acknowledgement {'enabled' if static_ip_verified else 'disabled'}"
        session.add(
            SystemEvent(
                severity="WARNING" if not enforce_paper_session_requirement else "INFO",
                component="RISK",
                event_type="LIVE_SAFETY_CHANGED",
                message=message,
            )
        )
        enqueue(
            session,
            "OWNER",
            "LIVE_SAFETY_CHANGED",
            message,
            f"live-safety:{datetime.now(UTC).isoformat()}",
        )
    session.commit()
    return flash_redirect("/settings", "Live safety gate updated immediately")


@app.post("/settings/refresh-instruments")
async def refresh_instruments(provider: str | None = Form(None)):
    try:
        count = await runtime_service.refresh_instruments(provider_name(provider) if provider else None)
        return flash_redirect("/settings", f"Instrument master refreshed: {count} new versions")
    except Exception as exc:
        return flash_redirect("/settings", f"Instrument refresh failed: {exc}", False)


@app.post("/settings/reconcile")
async def reconcile_now(session: Session = Depends(get_session)):
    try:
        _, _, trading_broker = selected_brokers(session)
        matched, detail = await reconcile(session, trading_broker)
    except Exception as exc:
        matched, detail = False, str(exc)
    runtime = session.get(RuntimeSetting, 1)
    if matched and runtime.execution_mode == "LIVE" and runtime.entries_paused:
        detail += "; entries remain paused until LIVE TRADING is confirmed again"
    return flash_redirect("/settings", detail, matched)


@app.get("/health")
def health(session: Session = Depends(get_session)):
    runtime = session.get(RuntimeSetting, 1)
    provider, market_broker, trading_broker = selected_brokers(session)
    db_ok, db_detail = database_health(session)
    model_ok = runtime_service.model.available
    last_tick_age = None
    if runtime_service.last_tick:
        tick = runtime_service.last_tick.replace(tzinfo=UTC) if runtime_service.last_tick.tzinfo is None else runtime_service.last_tick.astimezone(UTC)
        last_tick_age = round((datetime.now(UTC) - tick).total_seconds(), 1)
    backup = backup_status(config.root)
    permissions_ok, permissions_detail = file_permission_status(
        [config.upstox_analytics_token_file, config.upstox_trading_token_file, config.groww_token_file]
    )
    healthy = db_ok and model_ok and permissions_ok and runtime_service.last_error is None
    return {
        "status": "ok" if healthy else "degraded",
        "mode": runtime.execution_mode,
        "entries_paused": runtime.entries_paused,
        "active_broker": provider,
        "database": {"ok": db_ok, "detail": db_detail},
        "model": {"ok": model_ok, "version": runtime_service.model.metadata.get("version")},
        "market_data": {"last_tick": runtime_service.last_tick, "age_seconds": last_tick_age},
        "backup": backup,
        "secret_permissions": {"ok": permissions_ok, "detail": permissions_detail},
        "market_token_configured": bool(market_broker.access_token()),
        "trading_token_configured": bool(trading_broker.access_token()),
        "runtime_error": runtime_service.last_error,
    }
