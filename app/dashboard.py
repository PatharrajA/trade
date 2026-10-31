from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import database_health, settings_dict
from .config import config
from .data_quality import research_readiness
from .models import BacktestRun, Candle, Instrument, ModelArtifact, NotificationOutbox, Order, ResearchSyncRun, RuntimeSetting, ShadowObservation, Signal, SystemEvent, Trade
from .strategy import SUPPORTED_FEATURE_SCHEMAS
from .shadow_quality import shadow_report
from .training import load_model_metadata


def _today_start() -> datetime:
    local_today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    return datetime.combine(local_today, time.min, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def dashboard_context(session: Session, runtime_service=None) -> dict:
    runtime = session.get(RuntimeSetting, 1)
    settings = settings_dict(session)
    trades = session.scalars(select(Trade).order_by(Trade.entry_time.desc())).all()
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    closed_today = [row for row in trades if row.status == "CLOSED" and row.exit_time and _utc(row.exit_time) >= _today_start()]
    closed_month = [row for row in trades if row.status == "CLOSED" and row.exit_time and _utc(row.exit_time).astimezone(ZoneInfo("Asia/Kolkata")).year == now_ist.year and _utc(row.exit_time).astimezone(ZoneInfo("Asia/Kolkata")).month == now_ist.month]
    closed_month_sorted = sorted(closed_month, key=lambda item: _utc(item.exit_time))
    open_trade = next((row for row in trades if row.status == "OPEN"), None)
    today_pnl = sum((row.net_pnl for row in closed_today), Decimal("0"))
    month_pnl = sum((row.net_pnl for row in closed_month), Decimal("0"))
    month_gross_pnl = sum(((row.gross_pnl or Decimal("0")) for row in closed_month), Decimal("0"))
    month_charges = sum(((row.charges or Decimal("0")) for row in closed_month), Decimal("0"))
    open_pnl = ((open_trade.current_bid or open_trade.entry_price) - open_trade.entry_price) * open_trade.quantity if open_trade else Decimal("0")
    winning_trades = [row for row in closed_month if row.net_pnl > 0]
    losing_trades = [row for row in closed_month if row.net_pnl < 0]
    wins = len(winning_trades)
    gross_wins = sum((row.net_pnl for row in winning_trades), Decimal("0"))
    gross_losses = abs(sum((row.net_pnl for row in losing_trades), Decimal("0")))
    profit_factor = float(gross_wins / gross_losses) if gross_losses else None
    expectancy = month_pnl / len(closed_month) if closed_month else Decimal("0")
    average_win = gross_wins / len(winning_trades) if winning_trades else Decimal("0")
    average_loss = -gross_losses / len(losing_trades) if losing_trades else Decimal("0")
    largest_win = max((row.net_pnl for row in winning_trades), default=Decimal("0"))
    largest_loss = min((row.net_pnl for row in losing_trades), default=Decimal("0"))
    holding_minutes = [
        (_utc(row.exit_time) - _utc(row.entry_time)).total_seconds() / 60
        for row in closed_month
        if row.exit_time and row.entry_time
    ]
    average_holding_minutes = round(sum(holding_minutes) / len(holding_minutes)) if holding_minutes else 0
    db_ok, db_detail = database_health(session)
    per_index = {}
    for code in ("NIFTY", "BANK_NIFTY", "SENSEX"):
        rows = [row for row in closed_month if row.underlying_code == code]
        index_wins = sum(1 for row in rows if row.net_pnl > 0)
        per_index[code] = {
            "pnl": sum((row.net_pnl for row in rows), Decimal("0")),
            "trades": len(rows),
            "wins": index_wins,
            "win_rate": index_wins / len(rows) * 100 if rows else 0,
        }
    curve = [Decimal("0")]
    peak = Decimal("0")
    maximum_drawdown = Decimal("0")
    for row in closed_month_sorted:
        curve.append(curve[-1] + row.net_pnl)
        peak = max(peak, curve[-1])
        maximum_drawdown = max(maximum_drawdown, peak - curve[-1])
    low, high = min(curve), max(curve)
    spread = high - low
    points = []
    for index, value in enumerate(curve):
        x = 10 + (680 * index / max(len(curve) - 1, 1))
        y = 110 if spread == 0 else 200 - (float((value - low) / spread) * 170)
        points.append(f"{x:.1f},{y:.1f}")
    enabled_indices = {item for item in settings.get("enabled_indices", "").split(",") if item}
    paper_sessions = session.scalar(
        select(func.count(func.distinct(func.date(Trade.exit_time)))).where(
            Trade.mode == "PAPER", Trade.status == "CLOSED"
        )
    ) or 0
    required_paper_sessions = int(settings.get("required_paper_sessions", "20"))
    capital = Decimal(settings.get("paper_capital", "500000"))
    capital_return = float(month_pnl / capital * 100) if capital else 0.0
    daily_loss_budget = capital * Decimal(settings.get("daily_loss_limit_pct", "0.75")) / Decimal("100")
    open_risk = (open_trade.risk_budget or Decimal("0")) if open_trade else Decimal("0")
    realized_loss_today = max(Decimal("0"), -today_pnl)
    risk_used_amount = realized_loss_today + open_risk
    risk_usage_percent = min(100.0, float(risk_used_amount / daily_loss_budget * 100)) if daily_loss_budget else 0.0
    remaining_daily_risk = max(Decimal("0"), daily_loss_budget - risk_used_amount)
    trades_remaining = max(0, int(settings.get("max_trades_per_day", "5")) - len(closed_today) - (1 if open_trade else 0))
    consecutive_losses = 0
    for trade in sorted((row for row in trades if row.status == "CLOSED"), key=lambda item: _utc(item.exit_time), reverse=True):
        if trade.net_pnl >= 0:
            break
        consecutive_losses += 1

    def breakdown(field: str, values: tuple[str, ...]) -> dict[str, dict]:
        result = {}
        for value in values:
            rows = [row for row in closed_month if getattr(row, field) == value]
            result[value] = {
                "pnl": sum((row.net_pnl for row in rows), Decimal("0")),
                "trades": len(rows),
                "wins": sum(1 for row in rows if row.net_pnl > 0),
            }
        return result

    exit_values = tuple(dict.fromkeys(row.exit_reason or "UNKNOWN" for row in closed_month))
    daily_pnl: dict[int, Decimal] = {}
    for row in closed_month:
        day = _utc(row.exit_time).astimezone(ZoneInfo("Asia/Kolkata")).day
        daily_pnl[day] = daily_pnl.get(day, Decimal("0")) + row.net_pnl
    return {
        "runtime": runtime,
        "settings": settings,
        "open_trade": open_trade,
        "today_pnl": today_pnl,
        "month_pnl": month_pnl,
        "open_pnl": open_pnl,
        "month_gross_pnl": month_gross_pnl,
        "month_charges": month_charges,
        "capital_return": capital_return,
        "today_trades": len(closed_today),
        "win_rate": wins / len(closed_month) * 100 if closed_month else 0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_win": average_win,
        "average_loss": average_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "average_holding_minutes": average_holding_minutes,
        "maximum_drawdown": maximum_drawdown,
        "risk_usage_percent": risk_usage_percent,
        "daily_loss_budget": daily_loss_budget,
        "open_risk": open_risk,
        "remaining_daily_risk": remaining_daily_risk,
        "trades_remaining": trades_remaining,
        "consecutive_losses": consecutive_losses,
        "per_index": per_index,
        "option_breakdown": breakdown("option_type", ("CE", "PE")),
        "moneyness_breakdown": breakdown("moneyness", ("ATM", "ITM1")),
        "exit_breakdown": breakdown("exit_reason", exit_values),
        "daily_pnl": daily_pnl,
        "calendar_days": range(1, now_ist.day + 1),
        "recent_closed_trades": sorted(closed_month, key=lambda item: _utc(item.exit_time), reverse=True)[:5],
        "month_trade_count": len(closed_month),
        "equity_points": " ".join(points),
        "enabled_indices": enabled_indices,
        "db_ok": db_ok,
        "db_detail": db_detail,
        "last_tick": getattr(runtime_service, "last_tick", None),
        "last_candle": getattr(runtime_service, "last_candle", None),
        "runtime_error": getattr(runtime_service, "last_error", None),
        "active_broker": settings.get("active_broker", "UPSTOX"),
        "paper_sessions": int(paper_sessions),
        "required_paper_sessions": required_paper_sessions,
        "remaining_paper_sessions": max(0, required_paper_sessions - int(paper_sessions)),
        "enforce_paper_sessions": settings.get("enforce_paper_session_requirement", "true") == "true",
    }


def orders_context(session: Session) -> dict:
    signals = session.scalars(select(Signal).order_by(Signal.created_at.desc()).limit(100)).all()
    signal_details = {}
    for signal in signals:
        try:
            signal_details[signal.id] = json.loads(signal.decision_json) if signal.decision_json else {}
        except (TypeError, ValueError):
            signal_details[signal.id] = {}
    return {
        "trades": session.scalars(select(Trade).order_by(Trade.entry_time.desc()).limit(100)).all(),
        "orders": session.scalars(select(Order).order_by(Order.created_at.desc()).limit(100)).all(),
        "signals": signals,
        "signal_details": signal_details,
    }


def backtests_context(session: Session, selected_id: int | None = None) -> dict:
    runs = session.scalars(select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(30)).all()
    selected = session.get(BacktestRun, selected_id) if selected_id else (runs[0] if runs else None)
    summary = json.loads(selected.summary_json) if selected and selected.summary_json else None
    warnings = json.loads(selected.warnings_json) if selected and selected.warnings_json else []
    comparison = None
    if selected:
        try:
            selected_config = json.loads(selected.configuration_json)
        except (TypeError, ValueError):
            selected_config = {}
        comparison_id = selected_config.get("comparison_run_id") or selected_config.get("comparison_parent_id")
        comparison_run = session.get(BacktestRun, int(comparison_id)) if comparison_id else None
        if comparison_run:
            try:
                comparison_config = json.loads(comparison_run.configuration_json)
            except (TypeError, ValueError):
                comparison_config = {}
            comparison = {
                "run": comparison_run,
                "strategy_mode": comparison_config.get("strategy_mode", "RULE_ONLY"),
                "summary": json.loads(comparison_run.summary_json) if comparison_run.summary_json else None,
            }
    model_metadata = load_model_metadata(config.model_path)
    run_details = {}
    for run in runs:
        try:
            run_details[run.id] = json.loads(run.configuration_json)
        except (TypeError, ValueError):
            run_details[run.id] = {}
    verified_underlying = session.scalar(
        select(func.count()).select_from(Candle).where(Candle.data_role == "SPOT", Candle.source_verified.is_(True))
    ) or 0
    verified_futures = session.scalar(
        select(func.count()).select_from(Candle).where(Candle.data_role == "FUTURE", Candle.source_verified.is_(True))
    ) or 0
    verified_options = session.scalar(
        select(func.count()).select_from(Candle).where(Candle.data_role == "OPTION", Candle.source_verified.is_(True))
    ) or 0
    provider_counts = {}
    for provider in ("UPSTOX", "GROWW"):
        provider_counts[provider] = {
            "underlying": session.scalar(select(func.count()).select_from(Candle).where(Candle.provider == provider, Candle.data_role == "SPOT", Candle.source_verified.is_(True))) or 0,
            "futures": session.scalar(select(func.count()).select_from(Candle).where(Candle.provider == provider, Candle.data_role == "FUTURE", Candle.source_verified.is_(True))) or 0,
            "options": session.scalar(select(func.count()).select_from(Candle).where(Candle.provider == provider, Candle.data_role == "OPTION", Candle.source_verified.is_(True))) or 0,
        }
    research_runs = session.scalars(
        select(ResearchSyncRun).order_by(ResearchSyncRun.created_at.desc()).limit(10)
    ).all()
    model_artifacts = session.scalars(
        select(ModelArtifact).order_by(ModelArtifact.created_at.desc()).limit(10)
    ).all()
    model_details = {}
    for artifact in model_artifacts:
        try:
            model_details[artifact.id] = json.loads(artifact.metadata_json)
        except (TypeError, ValueError):
            model_details[artifact.id] = {}
    shadow_settings = settings_dict(session)
    shadow_path = shadow_settings.get("setup_shadow_model_path", "") or shadow_settings.get("setup_execution_model_path", "")
    shadow_artifact = session.scalar(
        select(ModelArtifact).where(ModelArtifact.model_path == shadow_path).limit(1)
    ) if shadow_path else None
    try:
        shadow_metadata = json.loads(shadow_artifact.metadata_json) if shadow_artifact else {}
    except (TypeError, ValueError):
        shadow_metadata = {}
    shadow_rows = session.scalars(
        select(ShadowObservation).where(
            ShadowObservation.provider == shadow_settings.get("active_broker", "UPSTOX"),
            ShadowObservation.model_version == shadow_metadata.get("version", ""),
        ).order_by(ShadowObservation.signal_time.desc())
    ).all() if shadow_metadata else []
    shadow_results = shadow_report(shadow_rows, shadow_metadata.get("indices") or [])
    return {
        "runs": runs,
        "selected": selected,
        "summary": summary,
        "comparison": comparison,
        "warnings": warnings,
        "model_ready": bool(model_metadata) and model_metadata.get("feature_schema") in SUPPORTED_FEATURE_SCHEMAS,
        "model_metadata": model_metadata,
        "run_details": run_details,
        "verified_underlying": verified_underlying,
        "verified_futures": verified_futures,
        "verified_options": verified_options,
        "provider_counts": provider_counts,
        "data_readiness": research_readiness(session),
        "research_runs": research_runs,
        "model_artifacts": model_artifacts,
        "setup_model_artifacts": [
            artifact
            for artifact in model_artifacts
            if model_details.get(artifact.id, {}).get("model_type") == "SETUP_QUALITY"
            and artifact.status != "PRUNED"
        ],
        "shadow_enabled": shadow_settings.get("setup_shadow_enabled") == "true",
        "shadow_model_path": shadow_settings.get("setup_shadow_model_path", ""),
        "setup_execution_enabled": shadow_settings.get("setup_execution_enabled") == "true",
        "setup_execution_model_path": shadow_settings.get("setup_execution_model_path", ""),
        "shadow_observations": shadow_rows[:20],
        "shadow_take": shadow_results["take"],
        "shadow_skip": shadow_results["skip"],
        "shadow_by_index": shadow_results["by_index"],
        "shadow_dimensions": shadow_results["dimensions"],
        "shadow_model_version": shadow_metadata.get("version"),
        "model_details": model_details,
        "active_broker": settings_dict(session).get("active_broker", "UPSTOX"),
        "default_historical_provider": settings_dict(session).get("default_historical_provider", "UPSTOX"),
    }


def telegram_context(session: Session) -> dict:
    return {
        "settings": settings_dict(session),
        "notifications": session.scalars(select(NotificationOutbox).order_by(NotificationOutbox.created_at.desc()).limit(50)).all(),
    }


def system_context(session: Session) -> dict:
    settings = settings_dict(session)
    paper_sessions = session.scalar(
        select(func.count(func.distinct(func.date(Trade.exit_time)))).where(
            Trade.mode == "PAPER", Trade.status == "CLOSED"
        )
    ) or 0
    return {
        "settings": settings,
        "events": session.scalars(select(SystemEvent).order_by(SystemEvent.created_at.desc()).limit(30)).all(),
        "instrument_count": session.scalar(select(func.count()).select_from(Instrument)) or 0,
        "paper_sessions": int(paper_sessions),
        "remaining_paper_sessions": max(0, int(settings.get("required_paper_sessions", "20")) - int(paper_sessions)),
    }
