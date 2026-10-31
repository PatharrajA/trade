from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from .database import database_health
from .models import NotificationOutbox, Order, RuntimeSetting, SystemEvent, Trade
from .providers import BrokerClient


def _broker_provider(broker: BrokerClient) -> str:
    """Return a provider name while keeping older test doubles compatible."""
    return str(getattr(broker, "provider", "UPSTOX")).upper()


async def reconcile(session, broker: BrokerClient) -> tuple[bool, str]:
    provider = _broker_provider(broker)
    if not broker.access_token():
        has_live = session.scalar(select(Trade.id).where(Trade.provider == provider, Trade.mode == "LIVE", Trade.status == "OPEN").limit(1))
        return (not bool(has_live), "Token missing" if has_live else "No live exposure")
    remote = await broker.positions()
    local = {trade.instrument_key: trade.quantity for trade in session.scalars(select(Trade).where(Trade.provider == provider, Trade.mode == "LIVE", Trade.status == "OPEN"))}
    return (remote == local, "Matched" if remote == local else f"{provider} position does not match the database")


async def change_mode(
    session,
    market_broker: BrokerClient,
    trading_broker: BrokerClient,
    target: str,
    confirmation: str = "",
    *,
    model_available: bool | None = None,
    paper_sessions: int | None = None,
    required_paper_sessions: int = 0,
    enforce_paper_sessions: bool = True,
    static_ip_verified: bool | None = None,
    feed_fresh: bool | None = None,
) -> tuple[bool, str]:
    target = target.upper()
    market_provider = _broker_provider(market_broker)
    trading_provider = _broker_provider(trading_broker)
    runtime = session.get(RuntimeSetting, 1)
    if target not in {"STOPPED", "PAPER", "LIVE"}:
        return False, "Unknown mode"
    unresolved = session.scalar(select(Order.id).where(Order.mode == "LIVE", Order.status.in_(["CREATED", "SUBMITTED", "OPEN", "PARTIAL", "PENDING"])).limit(1))
    live_trade = session.scalar(select(Trade.id).where(Trade.mode == "LIVE", Trade.status == "OPEN").limit(1))
    if runtime.execution_mode == "LIVE" and target != "LIVE" and (unresolved or live_trade):
        return False, "Exit or reconcile the live position/order before leaving LIVE"
    if target == "PAPER":
        if model_available is False:
            return False, "Enable an approved TAKE/SKIP model before starting Paper trading"
        market_connected, detail = await market_broker.market_data_health()
        if not market_connected:
            return False, f"Paper mode requires valid {market_provider} market-data access: {detail}"
    if target == "LIVE":
        if confirmation != "LIVE TRADING":
            return False, "Type LIVE TRADING exactly"
        if model_available is False:
            return False, "Activate a compatible trained model before enabling LIVE"
        if enforce_paper_sessions and paper_sessions is not None and paper_sessions < required_paper_sessions:
            return False, f"Complete {required_paper_sessions} paper sessions before LIVE ({paper_sessions} completed)"
        if static_ip_verified is False:
            return False, "Verify the broker static-IP setup in Settings before enabling LIVE"
        if feed_fresh is False:
            return False, "Live futures data is missing or stale; wait for a completed market-data sync"
        if session.scalar(select(Trade.id).where(Trade.mode == "PAPER", Trade.status == "OPEN").limit(1)):
            return False, "Close the paper position before enabling LIVE"
        ok, detail = database_health(session)
        if not ok:
            return False, f"Database preflight failed: {detail}"
        market_connected, detail = await market_broker.market_data_health()
        if not market_connected:
            return False, f"{market_provider} market-data preflight failed: {detail}"
        trading_connected, detail = await trading_broker.health()
        if not trading_connected:
            return False, f"Trading-token preflight failed ({trading_provider}): {detail}"
        matched, detail = await reconcile(session, trading_broker)
        if not matched:
            return False, detail
    previous = runtime.execution_mode
    runtime.execution_mode = target
    runtime.entries_paused = target == "STOPPED"
    runtime.updated_at = datetime.now(UTC)
    runtime.change_reason = f"Changed from {previous} to {target}"
    message = f"NiftyPilot execution changed from {previous} to {target}"
    session.add(SystemEvent(component="MODE", event_type="MODE_CHANGED", message=message))
    session.add(NotificationOutbox(destination="GROUP", event_type="MODE_CHANGED", message=message, dedupe_key=f"mode:{runtime.updated_at.isoformat()}"))
    session.commit()
    return True, message

