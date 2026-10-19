from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select

from .config import config
from .database import settings_dict
from .domain import Contract
from .execution import ExecutionService
from .mode_control import reconcile
from .models import Candle, Instrument, RuntimeSetting, Signal, SystemEvent, Trade
from .option_selector import select_option
from .providers import BrokerClient, provider_name
from .strategy import RegimeModel, decide, with_futures_features
from .telegram import enqueue
from .training import front_future_frame


class RuntimeService:
    def __init__(self, market_brokers: dict[str, BrokerClient], trading_brokers: dict[str, BrokerClient], session_factory) -> None:
        self.market_brokers = market_brokers
        self.trading_brokers = trading_brokers
        self.session_factory = session_factory
        self.execution = ExecutionService(trading_brokers)
        self.model = RegimeModel(config.model_path)
        self.stop_event = asyncio.Event()
        self.last_tick: datetime | None = None
        self.last_candle: datetime | None = None
        self.last_instrument_refresh: tuple[str, date] | None = None
        self.last_error: str | None = None
        self.last_report_day: date | None = None
        self.last_reconcile_at: datetime | None = None

    def _active_provider(self, session) -> str:
        return provider_name(settings_dict(session).get("active_broker"))

    def reload_model(self) -> None:
        """Load a newly-trained model without restarting the application."""
        self.model = RegimeModel(config.model_path)

    @staticmethod
    def _parse(raw: list) -> tuple[datetime, Decimal, Decimal, Decimal, Decimal, int, int]:
        stamp = datetime.fromisoformat(str(raw[0]).replace("Z", "+00:00"))
        if stamp.tzinfo is None: stamp = stamp.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        return stamp.astimezone(UTC), *(Decimal(str(raw[index])) for index in range(1, 5)), int(raw[5] or 0), int(raw[6] or 0) if len(raw) > 6 else 0

    @staticmethod
    def _db_stamp(value: datetime) -> datetime:
        return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).replace(tzinfo=None)

    def _store_runtime_rows(
        self,
        session,
        *,
        provider: str,
        code: str,
        instrument_key: str,
        rows: list[tuple[datetime, Decimal, Decimal, Decimal, Decimal, int, int]],
        data_role: str,
        context: dict[str, float | None] | None = None,
    ) -> None:
        existing_rows = session.scalars(
            select(Candle).where(
                Candle.provider == provider,
                Candle.instrument_key == instrument_key,
                Candle.timeframe == "5minute",
            )
        ).all()
        existing = {self._db_stamp(item.timestamp): item for item in existing_rows}
        latest_stamp = max((row[0] for row in rows), default=None)
        for stamp, open_, high, low, close, volume, oi in rows:
            normalized = self._db_stamp(stamp)
            stored = existing.get(normalized)
            is_latest = stamp == latest_stamp
            if stored is None:
                stored = Candle(
                    provider=provider,
                    instrument_key=instrument_key,
                    underlying_code=code,
                    timeframe="5minute",
                    timestamp=normalized,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    open_interest=oi,
                    source_verified=True,
                    data_role=data_role,
                )
                session.add(stored)
                existing[normalized] = stored
            else:
                stored.open, stored.high, stored.low, stored.close = open_, high, low, close
                stored.volume, stored.open_interest = volume, oi
                stored.source_verified, stored.data_role = True, data_role
            if context and is_latest:
                stored.india_vix = context.get("india_vix")
                stored.futures_basis_pct = context.get("futures_basis_pct")
                stored.pcr = context.get("pcr")

    async def refresh_instruments(self, provider: str | None = None) -> int:
        total = 0
        today = date.today()
        if provider is None:
            with self.session_factory() as session:
                provider = self._active_provider(session)
        provider = provider_name(provider)
        broker = self.market_brokers[provider]
        for code in ("NIFTY", "BANK_NIFTY", "SENSEX"):
            contracts = await broker.contracts(broker.index_key(code), code)
            current_keys = {item.instrument_key for item in contracts}
            with self.session_factory() as session:
                active = session.scalars(
                    select(Instrument).where(
                        Instrument.provider == provider,
                        Instrument.underlying_code == code,
                        Instrument.instrument_kind == "OPTION",
                        Instrument.active.is_(True),
                    )
                ).all()
                for old in active:
                    if old.instrument_key not in current_keys:
                        old.active, old.valid_to = False, today - timedelta(days=1)
                existing = {(item.instrument_key, item.valid_from) for item in session.scalars(select(Instrument).where(Instrument.provider == provider, Instrument.underlying_code == code))}
                for item in contracts:
                    if (item.instrument_key, today) in existing: continue
                    session.add(
                        Instrument(
                            provider=provider,
                            instrument_key=item.instrument_key,
                            underlying_code=code,
                            exchange=item.exchange,
                            trading_symbol=item.trading_symbol,
                            option_type=item.option_type,
                            strike=item.strike,
                            expiry=item.expiry,
                            lot_size=item.lot_size,
                            tick_size=item.tick_size,
                            valid_from=today,
                            instrument_kind="OPTION",
                            metadata_verified=item.metadata_verified,
                            metadata_source=item.metadata_source,
                        )
                    )
                    total += 1
                session.commit()
        self.last_instrument_refresh = (provider, today)
        return total

    async def sync_candles(self) -> None:
        with self.session_factory() as session:
            provider = self._active_provider(session)
        broker = self.market_brokers[provider]
        for code in ("NIFTY", "BANK_NIFTY", "SENSEX"):
            key = broker.index_key(code)
            rows = await broker.historical_candles(key, date.today() - timedelta(days=3), date.today())
            if not rows: continue
            parsed = [self._parse(row) for row in rows]
            parsed.sort(key=lambda row: row[0])
            latest = parsed[-1]
            context = await broker.market_context(key, latest[4])
            with self.session_factory() as session:
                self._store_runtime_rows(
                    session,
                    provider=provider,
                    code=code,
                    instrument_key=key,
                    rows=parsed,
                    data_role="SPOT",
                    context=context,
                )
                session.commit()
            futures = await broker.future_contracts(code)
            eligible_futures = [item for item in futures if item.expiry > date.today()]
            if eligible_futures:
                future = min(eligible_futures, key=lambda item: item.expiry)
                future_rows = await broker.historical_candles(
                    future.instrument_key, date.today() - timedelta(days=3), date.today()
                )
                future_parsed = sorted((self._parse(row) for row in future_rows), key=lambda row: row[0])
                if future_parsed:
                    with self.session_factory() as session:
                        existing_future = session.scalar(
                            select(Instrument).where(
                                Instrument.provider == provider,
                                Instrument.instrument_key == future.instrument_key,
                                Instrument.valid_from == date.today(),
                            )
                        )
                        if existing_future is None:
                            session.add(
                                Instrument(
                                    provider=provider,
                                    instrument_key=future.instrument_key,
                                    underlying_code=code,
                                    exchange=future.exchange,
                                    trading_symbol=future.trading_symbol,
                                    option_type="FU",
                                    strike=Decimal("0"),
                                    expiry=future.expiry,
                                    lot_size=future.lot_size,
                                    tick_size=future.tick_size,
                                    valid_from=date.today(),
                                    valid_to=future.expiry,
                                    active=True,
                                    instrument_kind="FUTURE",
                                    metadata_verified=future.metadata_verified,
                                    metadata_source=future.metadata_source,
                                )
                            )
                        self._store_runtime_rows(
                            session,
                            provider=provider,
                            code=code,
                            instrument_key=future.instrument_key,
                            rows=future_parsed,
                            data_role="FUTURE",
                        )
                        session.commit()
            self.last_candle = max(self.last_candle or latest[0], latest[0])

    async def process_signals(self, boundary: datetime) -> None:
        with self.session_factory() as session:
            runtime = session.get(RuntimeSetting, 1)
            if not runtime or runtime.execution_mode not in {"PAPER", "LIVE"} or runtime.entries_paused: return
            if session.scalar(select(Trade.id).where(Trade.status == "OPEN").limit(1)): return
            settings = settings_dict(session)
            provider = provider_name(settings.get("active_broker"))
            broker = self.market_brokers[provider]
            candidates = []
            for code in settings["enabled_indices"].split(","):
                rows = session.scalars(select(Candle).where(Candle.provider == provider, Candle.underlying_code == code, Candle.data_role == "SPOT", Candle.timeframe == "5minute", Candle.timestamp < boundary).order_by(Candle.timestamp.desc()).limit(80)).all()[::-1]
                if len(rows) < 50: continue
                future_rows = session.scalars(select(Candle).where(Candle.provider == provider, Candle.underlying_code == code, Candle.data_role == "FUTURE", Candle.timeframe == "5minute", Candle.timestamp < boundary).order_by(Candle.timestamp.desc()).limit(240)).all()
                future_instruments = session.scalars(select(Instrument).where(Instrument.provider == provider, Instrument.underlying_code == code, Instrument.instrument_kind == "FUTURE")).all()
                futures_frame = front_future_frame(future_rows, future_instruments)
                if len(futures_frame) < 50: continue
                frame = with_futures_features(
                    pd.DataFrame([{"timestamp": item.timestamp, "open": float(item.open), "high": float(item.high), "low": float(item.low), "close": float(item.close), "volume": item.volume} for item in rows]),
                    futures_frame,
                )
                candle = rows[-1]
                decision = decide(frame.iloc[-1], {"india_vix": candle.india_vix, "futures_basis_pct": candle.futures_basis_pct, "pcr": candle.pcr}, self.model)
                if decision.direction == "NO_TRADE":
                    signal_id = f"reject-{provider.lower()}-{code}-{int(candle.timestamp.timestamp())}"
                    if not session.get(Signal, signal_id): session.add(Signal(provider=provider, id=signal_id, mode=runtime.execution_mode, underlying_code=code, candle_timestamp=candle.timestamp, direction="NO_TRADE", regime=decision.regime, confidence=decision.confidence, status="REJECTED", rejection_reason=decision.reason, decision_json=json.dumps({"features": decision.features, "checks": {"confidence": decision.confidence >= .70, "above_futures_vwap": float(frame.iloc[-1].futures_close) > float(frame.iloc[-1].futures_vwap), "ema9_above_ema20": float(frame.iloc[-1].ema9) > float(frame.iloc[-1].ema20), "breakout": float(frame.iloc[-1].close) > float(frame.iloc[-1].opening_range_high)}}, default=str)))
                    continue
                db_contracts = session.scalars(select(Instrument).where(Instrument.provider == provider, Instrument.underlying_code == code, Instrument.option_type == decision.direction, Instrument.active.is_(True))).all()
                contracts = [Contract(item.instrument_key, item.trading_symbol, item.exchange, code, item.expiry, item.strike, item.option_type, item.lot_size, item.tick_size, provider=provider) for item in db_contracts]
                expiries = sorted({item.expiry for item in contracts if item.expiry > date.today()})
                chain = [item for item in contracts if expiries and item.expiry == expiries[0]]
                strikes = sorted({item.strike for item in chain})
                if not strikes: continue
                spot = Decimal(candle.close)
                atm = min(strikes, key=lambda value: abs(value - spot)); idx = strikes.index(atm)
                itm = strikes[max(0, idx - 1)] if decision.direction == "CE" else strikes[min(len(strikes) - 1, idx + 1)]
                short_list = [item for item in chain if item.strike in {atm, itm}]
                quotes = await broker.quotes([item.instrument_key for item in short_list])
                selection = select_option(short_list, quotes, spot=spot, direction=decision.direction, trading_day=date.today(), max_spread_pct=Decimal(settings["max_spread_pct"]), max_quote_age_seconds=int(settings["max_quote_age_seconds"]), delta_min=float(settings["delta_min"]), delta_max=float(settings["delta_max"]))
                if selection.contract:
                    spread = float((selection.quote.ask - selection.quote.bid) / ((selection.quote.ask + selection.quote.bid) / 2) * 100)
                    score = decision.confidence - spread * 0.02 + min(selection.quote.volume / 100000, 1) * 0.08
                    candidates.append((score, code, candle, decision, selection, frame.iloc[-1]))
            session.commit()
            if not candidates: return
            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, code, candle, decision, selection, _row in candidates[1:]:
                signal_id = f"rank-{provider.lower()}-{code}-{int(candle.timestamp.timestamp())}"
                if not session.get(Signal, signal_id): session.add(Signal(provider=provider, id=signal_id, mode=runtime.execution_mode, underlying_code=code, candle_timestamp=candle.timestamp, direction=decision.direction, regime=decision.regime, confidence=decision.confidence, status="REJECTED", rejection_reason="Lower-ranked simultaneous signal", decision_json=json.dumps({"features": decision.features, "ranking": "lower-ranked simultaneous signal"}, default=str)))
            session.commit()
            _, code, candle, decision, selection, row = candidates[0]
            atr = Decimal(str(row.atr))
            underlying_stop = Decimal(candle.close) - atr if decision.direction == "CE" else Decimal(candle.close) + atr
            await self.execution.enter(session, underlying_code=code, candle_time=candle.timestamp, regime=decision.regime, confidence=decision.confidence, direction=decision.direction, underlying_price=Decimal(candle.close), underlying_stop=underlying_stop, contract=selection.contract, quote=selection.quote, moneyness=selection.moneyness, decision_context={"features": decision.features, "reason": decision.reason, "checks": {"confidence": decision.confidence >= .70, "futures_vwap": f"{row.futures_close} vs {row.futures_vwap}", "ema9_20": f"{row.ema9} vs {row.ema20}", "opening_range": f"{row.close} / {row.opening_range_high} / {row.opening_range_low}"}, "selection": {"instrument": selection.contract.instrument_key, "symbol": selection.contract.trading_symbol, "strike": str(selection.contract.strike), "moneyness": selection.moneyness, "bid": str(selection.quote.bid), "ask": str(selection.quote.ask), "delta": selection.quote.delta, "iv": selection.quote.iv}})

    async def monitor(self) -> None:
        with self.session_factory() as session:
            trade = session.scalar(select(Trade).where(Trade.status == "OPEN").limit(1))
            if not trade: return
            provider = provider_name(trade.provider)
            broker = self.market_brokers[provider]
            underlying_key = broker.index_key(trade.underlying_code)
            option_quote = (await broker.quotes([trade.instrument_key])).get(trade.instrument_key)
            underlying_quote = (await broker.quotes([underlying_key])).get(underlying_key)
            if not option_quote or not underlying_quote: return
            self.last_tick = min(option_quote.timestamp, underlying_quote.timestamp)
            trade.current_bid = option_quote.bid
            session.commit()
            reason = None
            if option_quote.bid <= trade.premium_emergency_stop: reason = "PREMIUM_EMERGENCY_STOP"
            elif trade.option_type == "CE" and underlying_quote.last <= trade.underlying_stop: reason = "UNDERLYING_STOP"
            elif trade.option_type == "PE" and underlying_quote.last >= trade.underlying_stop: reason = "UNDERLYING_STOP"
            elif trade.option_type == "CE" and underlying_quote.last >= trade.underlying_target: reason = "UNDERLYING_TARGET"
            elif trade.option_type == "PE" and underlying_quote.last <= trade.underlying_target: reason = "UNDERLYING_TARGET"
            elif datetime.now(UTC).astimezone(ZoneInfo("Asia/Kolkata")).time() >= time(15, 20): reason = "TIME_EXIT"
            if reason: await self.execution.exit(session, trade, option_quote, reason)

    async def validate_live_token(self) -> bool:
        with self.session_factory() as session:
            runtime = session.get(RuntimeSetting, 1)
            if not runtime or runtime.execution_mode != "LIVE" or runtime.entries_paused:
                return True
            provider = self._active_provider(session)
            broker = self.trading_brokers[provider]
            connected, detail = await broker.health()
            if connected:
                return True
            runtime.entries_paused = True
            runtime.change_reason = f"Live entries paused: {detail}"
            session.add(SystemEvent(severity="CRITICAL", component=provider, event_type="TRADING_TOKEN_INVALID", message=detail))
            enqueue(session, "OWNER", "TRADING_TOKEN_INVALID", f"Live entries paused. {provider} trading token check failed: {detail}", f"trading-token-invalid:{provider}:{date.today().isoformat()}")
            session.commit()
            return False

    async def validate_live_reconciliation(self) -> bool:
        now = datetime.now(UTC)
        if self.last_reconcile_at and (now - self.last_reconcile_at).total_seconds() < 60:
            return True
        with self.session_factory() as session:
            runtime = session.get(RuntimeSetting, 1)
            if not runtime or runtime.execution_mode != "LIVE":
                return True
            provider = self._active_provider(session)
            try:
                matched, detail = await reconcile(session, self.trading_brokers[provider])
            except Exception as exc:
                matched, detail = False, str(exc)
            self.last_reconcile_at = now
            if matched:
                return True
            runtime.entries_paused = True
            runtime.change_reason = f"Live reconciliation failed: {detail}"
            session.add(SystemEvent(severity="CRITICAL", component=provider, event_type="POSITION_MISMATCH", message=detail))
            enqueue(session, "OWNER", "POSITION_MISMATCH", f"Live entries paused: {detail}", f"position-mismatch:{provider}:{now.isoformat(timespec='minutes')}")
            session.commit()
            return False

    def send_scheduled_reports(self) -> None:
        now = datetime.now(UTC).astimezone(ZoneInfo("Asia/Kolkata"))
        if now.time() < time(15, 31) or self.last_report_day == now.date():
            return
        with self.session_factory() as session:
            settings = settings_dict(session)
            if settings.get("telegram_enabled") != "true":
                self.last_report_day = now.date()
                return
            trades = session.scalars(
                select(Trade).where(Trade.status == "CLOSED").order_by(Trade.exit_time)
            ).all()
            today_rows = [
                item for item in trades
                if item.exit_time and (item.exit_time.replace(tzinfo=UTC) if item.exit_time.tzinfo is None else item.exit_time.astimezone(UTC)).astimezone(ZoneInfo("Asia/Kolkata")).date() == now.date()
            ]
            daily_net = sum((item.net_pnl for item in today_rows), Decimal("0"))
            wins = sum(1 for item in today_rows if item.net_pnl > 0)
            if settings.get("telegram_daily_report", "true") == "true":
                enqueue(session, "GROUP", "DAILY_REPORT", f"NiftyPilot daily report · {now.date()}\nTrades {len(today_rows)} · wins {wins}\nNet P&L ₹{daily_net}", f"daily-report:{now.date().isoformat()}")
            if settings.get("telegram_monthly_report", "true") == "true" and (now.date() + timedelta(days=1)).month != now.month:
                month_rows = [
                    item for item in trades
                    if item.exit_time and (item.exit_time.replace(tzinfo=UTC) if item.exit_time.tzinfo is None else item.exit_time.astimezone(UTC)).astimezone(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m") == now.strftime("%Y-%m")
                ]
                month_net = sum((item.net_pnl for item in month_rows), Decimal("0"))
                enqueue(session, "GROUP", "MONTHLY_REPORT", f"NiftyPilot monthly report · {now:%B %Y}\nTrades {len(month_rows)}\nNet P&L ₹{month_net}", f"monthly-report:{now:%Y-%m}")
            session.commit()
        self.last_report_day = now.date()

    async def run(self) -> None:
        last_boundary = None
        while not self.stop_event.is_set():
            try:
                self.send_scheduled_reports()
                with self.session_factory() as session:
                    provider = self._active_provider(session)
                broker = self.market_brokers[provider]
                if not broker.access_token():
                    await asyncio.sleep(5); continue
                if self.last_instrument_refresh != (provider, date.today()): await self.refresh_instruments(provider)
                await self.monitor()
                await self.validate_live_reconciliation()
                now = datetime.now(UTC)
                boundary = now.replace(second=0, microsecond=0, minute=now.minute // 5 * 5)
                if now.second >= 10 and boundary != last_boundary:
                    await self.validate_live_token()
                    await self.sync_candles()
                    await self.process_signals(boundary)
                    last_boundary = boundary
                self.last_error = None
            except asyncio.CancelledError: raise
            except Exception as exc:
                self.last_error = str(exc)
                with self.session_factory() as session:
                    runtime = session.get(RuntimeSetting, 1)
                    if runtime and runtime.execution_mode == "LIVE":
                        runtime.entries_paused = True
                    provider = self._active_provider(session)
                    session.add(SystemEvent(severity="ERROR", component=provider, event_type="LOOP_ERROR", message=str(exc)))
                    enqueue(session, "OWNER", "BROKER_CONNECTION_ERROR", f"{provider} connection failed: {exc}", f"broker-error:{provider}:{date.today().isoformat()}")
                    session.commit()
            await asyncio.sleep(1)

    def stop(self) -> None:
        self.stop_event.set()

