from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import func, select

from .database import settings_dict
from .domain import Contract, Quote
from .mode_control import reconcile
from .models import Order, RuntimeSetting, Signal, SystemEvent, Trade
from .providers import BrokerClient, provider_name
from .risk import can_enter, limits_from_settings, option_charges, paper_fill, position_size
from .telegram import enqueue


class ExecutionError(RuntimeError):
    pass


class ExecutionService:
    def __init__(self, trading_brokers: dict[str, BrokerClient] | BrokerClient) -> None:
        self.trading_brokers = trading_brokers if isinstance(trading_brokers, dict) else {"UPSTOX": trading_brokers}

    async def _fill(self, session, broker: BrokerClient, order: Order, timeout_seconds: int = 30) -> tuple[Decimal, int]:
        for _ in range(timeout_seconds * 2):
            history = await broker.order_history(order.broker_order_id or "")
            latest = history[-1] if history else None
            if latest:
                previous_status = order.status
                order.status = latest.status
                order.average_price = latest.average_price
                order.filled_quantity = latest.filled_quantity
                session.commit()
                if latest.status != previous_status:
                    enqueue(session, "GROUP", "ORDER_STATUS", f"{order.mode} {order.purpose} order · {broker.provider}\n{order.trading_symbol}\n{previous_status} → {latest.status} · filled {latest.filled_quantity}/{latest.quantity}", f"order-status:{broker.provider}:{order.broker_order_id}:{latest.status}:{latest.filled_quantity}")
                    session.commit()
                if latest.status in {"COMPLETE", "COMPLETED", "EXECUTED", "DELIVERY_AWAITED"}:
                    return latest.average_price, latest.filled_quantity
                if latest.status in {"REJECTED", "FAILED", "CANCELLED", "CANCELED"}:
                    if latest.filled_quantity:
                        return latest.average_price, latest.filled_quantity
                    raise ExecutionError(f"{broker.provider} order {latest.status}")
            await asyncio.sleep(0.5)
        if order.filled_quantity:
            order.status = "PARTIAL_RECONCILIATION_REQUIRED"
            session.commit()
            return order.average_price or Decimal("0"), order.filled_quantity
        order.status = "RECONCILIATION_REQUIRED"
        session.commit()
        raise ExecutionError("Fill timed out; new entries are blocked pending reconciliation")

    async def enter(
        self,
        session,
        *,
        underlying_code: str,
        candle_time: datetime,
        regime: str,
        confidence: float,
        direction: str,
        underlying_price: Decimal,
        underlying_stop: Decimal,
        contract: Contract,
        quote: Quote,
        moneyness: str,
        decision_context: dict | None = None,
    ) -> Trade:
        runtime = session.get(RuntimeSetting, 1)
        if not runtime or runtime.execution_mode not in {"PAPER", "LIVE"} or runtime.entries_paused:
            raise ExecutionError("Execution is stopped")
        settings = settings_dict(session)
        provider = provider_name(settings.get("active_broker"))
        broker = self.trading_brokers[provider]
        signal_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{provider}:{runtime.execution_mode}:{underlying_code}:{candle_time.isoformat()}:{direction}").hex
        if session.get(Signal, signal_id):
            raise ExecutionError("Duplicate signal")
        signal = Signal(provider=provider, id=signal_id, mode=runtime.execution_mode, underlying_code=underlying_code, candle_timestamp=candle_time, direction=direction, regime=regime, confidence=confidence, selected_instrument=contract.instrument_key, status="VALIDATING", decision_json=json.dumps(decision_context or {}, default=str, sort_keys=True))
        session.add(signal)
        session.flush()
        limits = limits_from_settings(settings)
        open_positions = session.scalar(select(func.count()).select_from(Trade).where(Trade.status == "OPEN")) or 0
        today_start = datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC)
        closed_today = session.scalars(select(Trade).where(Trade.mode == runtime.execution_mode, Trade.status == "CLOSED", Trade.exit_time >= today_start).order_by(Trade.exit_time.desc())).all()
        consecutive = 0
        for trade in closed_today:
            if trade.net_pnl < 0: consecutive += 1
            else: break
        reconciled = True
        connected = True
        if runtime.execution_mode == "LIVE":
            connected, _ = await broker.health()
            reconciled, _ = await reconcile(session, broker)
        quote_age = (datetime.now(UTC) - quote.timestamp.astimezone(UTC)).total_seconds()
        allowed, reason = can_enter(limits=limits, open_positions=open_positions, trades_today=len(closed_today), consecutive_losses=consecutive, realized_pnl_today=sum((row.net_pnl for row in closed_today), Decimal("0")), quote_fresh=quote_age <= int(settings["max_quote_age_seconds"]), broker_connected=connected, reconciled=reconciled)
        if not allowed:
            signal.status, signal.rejection_reason = "REJECTED", reason
            if reason in {"Daily-loss limit reached", "Consecutive-loss limit reached"}:
                runtime.entries_paused = True
                runtime.change_reason = reason
                session.add(SystemEvent(severity="CRITICAL", component="RISK", event_type="RISK_SHUTDOWN", message=reason))
                enqueue(
                    session,
                    "OWNER",
                    "RISK_SHUTDOWN",
                    f"{runtime.execution_mode} entries paused: {reason}",
                    f"risk-shutdown:{runtime.execution_mode}:{date.today().isoformat()}:{reason}",
                )
            session.commit()
            raise ExecutionError(reason)
        mode = runtime.execution_mode
        expected = quote.ask if mode == "LIVE" else paper_fill("BUY", quote.bid, quote.ask, Decimal(settings["paper_slippage_pct"]), contract.tick_size)
        plan = position_size(expected, contract.lot_size, limits, mode)
        if not plan.allowed:
            signal.status, signal.rejection_reason = "REJECTED", plan.reason
            session.commit()
            raise ExecutionError(plan.reason)
        order = Order(provider=provider, signal_id=signal_id, mode=mode, purpose="ENTRY", tag=f"np{signal_id[:18]}", instrument_key=contract.instrument_key, trading_symbol=contract.trading_symbol, side="BUY", quantity=plan.quantity, requested_price=expected, status="COMPLETE" if mode == "PAPER" else "CREATED", filled_quantity=plan.quantity if mode == "PAPER" else 0, average_price=expected if mode == "PAPER" else None)
        session.add(order)
        signal.status = "APPROVED"
        session.commit()
        actual, filled = expected, plan.quantity
        if mode == "LIVE":
            try:
                order.broker_order_id = await broker.place_market_order(instrument_key=contract.instrument_key, side="BUY", quantity=plan.quantity, tag=order.tag)
                order.status = "SUBMITTED"
                session.commit()
                actual, filled = await self._fill(session, broker, order)
            except Exception as exc:
                if not order.filled_quantity:
                    signal.status = "ORDER_ERROR"
                session.add(SystemEvent(severity="CRITICAL", component="EXECUTION", event_type="ENTRY_ERROR", message=str(exc)))
                enqueue(session, "OWNER", "ORDER_ERROR", f"{provider} entry requires attention: {exc}", f"entry-error:{signal_id}")
                session.commit()
                raise
        if filled <= 0:
            raise ExecutionError("No filled quantity")
        risk_points = abs(underlying_price - underlying_stop)
        target = underlying_price + risk_points * limits.reward_risk if direction == "CE" else underlying_price - risk_points * limits.reward_risk
        trade = Trade(provider=provider, signal_id=signal_id, mode=mode, underlying_code=underlying_code, instrument_key=contract.instrument_key, trading_symbol=contract.trading_symbol, option_type=direction, moneyness=moneyness, expiry=contract.expiry, quantity=filled, lot_size=contract.lot_size, tick_size=contract.tick_size, entry_price=actual, entry_bid=quote.bid, entry_ask=quote.ask, entry_slippage=max(Decimal("0"), actual - quote.ask), entry_delta=quote.delta, entry_iv=quote.iv, risk_budget=plan.risk_budget, current_bid=quote.bid, underlying_entry=underlying_price, underlying_stop=underlying_stop, underlying_target=target, premium_emergency_stop=actual * (Decimal("1") - limits.premium_emergency_stop_pct / 100))
        session.add(trade)
        signal.status = "ENTERED" if filled == plan.quantity else "PARTIAL_POSITION"
        enqueue(session, "GROUP", "ENTRY", f"{mode} ENTRY · {provider}\n{underlying_code} {contract.strike} {direction} · {moneyness}\nQty {filled} @ ₹{actual}\nIndex stop {underlying_stop} · target {target}", f"entry:{signal_id}")
        session.commit()
        return trade

    async def exit(self, session, trade: Trade, quote: Quote, reason: str) -> Trade:
        if trade.status != "OPEN":
            raise ExecutionError("Trade is not open")
        settings = settings_dict(session)
        provider = provider_name(trade.provider)
        broker = self.trading_brokers[provider]
        price = paper_fill("SELL", quote.bid, quote.ask, Decimal(settings["paper_slippage_pct"]), trade.tick_size) if trade.mode == "PAPER" else quote.bid
        if trade.mode == "LIVE":
            order = Order(provider=provider, signal_id=trade.signal_id, mode="LIVE", purpose="EXIT", tag=f"npx{trade.signal_id[:17]}", instrument_key=trade.instrument_key, trading_symbol=trade.trading_symbol, side="SELL", quantity=trade.quantity, requested_price=quote.bid)
            session.add(order)
            session.commit()
            order.broker_order_id = await broker.place_market_order(instrument_key=trade.instrument_key, side="SELL", quantity=trade.quantity, tag=order.tag)
            order.status = "SUBMITTED"
            session.commit()
            price, filled = await self._fill(session, broker, order)
            if filled != trade.quantity:
                raise ExecutionError("Exit partially filled; position remains under reconciliation")
        trade.exit_price = price
        trade.gross_pnl = (price - trade.entry_price) * trade.quantity
        exchange_name = "BSE" if trade.underlying_code == "SENSEX" else "NSE"
        trade.charges = option_charges(
            trade.entry_price,
            price,
            trade.quantity,
            provider,
            exchange_name=exchange_name,
        )
        trade.net_pnl = trade.gross_pnl - trade.charges
        trade.exit_reason, trade.exit_time, trade.status = reason, datetime.now(UTC), "CLOSED"
        enqueue(session, "GROUP", "EXIT", f"{trade.mode} EXIT · {provider}\n{trade.underlying_code} {trade.trading_symbol}\nNet P&L ₹{trade.net_pnl} · {reason}", f"exit:{trade.signal_id}")
        limits = limits_from_settings(settings)
        today_start = datetime.combine(date.today(), datetime.min.time(), tzinfo=UTC)
        closed_today = session.scalars(
            select(Trade).where(
                Trade.mode == trade.mode,
                Trade.status == "CLOSED",
                Trade.exit_time >= today_start,
            ).order_by(Trade.exit_time.desc())
        ).all()
        realized = sum((item.net_pnl for item in closed_today), Decimal("0"))
        consecutive = 0
        for item in closed_today:
            if item.net_pnl < 0:
                consecutive += 1
            else:
                break
        shutdown_reason = None
        if realized <= -(limits.capital * limits.daily_loss_limit_pct / 100):
            shutdown_reason = "Daily-loss limit reached"
        elif consecutive >= limits.max_consecutive_losses:
            shutdown_reason = "Consecutive-loss limit reached"
        if shutdown_reason:
            runtime = session.get(RuntimeSetting, 1)
            runtime.entries_paused = True
            runtime.change_reason = shutdown_reason
            session.add(SystemEvent(severity="CRITICAL", component="RISK", event_type="RISK_SHUTDOWN", message=shutdown_reason))
            enqueue(session, "OWNER", "RISK_SHUTDOWN", f"{trade.mode} entries paused: {shutdown_reason}", f"risk-shutdown:{trade.mode}:{date.today().isoformat()}:{shutdown_reason}")
        session.commit()
        return trade

