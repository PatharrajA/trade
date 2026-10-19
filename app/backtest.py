from __future__ import annotations

import csv
import io
import json
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import BacktestRun, BacktestTrade, Candle, Instrument
from .risk import RiskLimits, can_enter, option_charges, position_size
from .strategy import RegimeModel, decide, rule_only_decide, with_futures_features
from .training import front_future_frame


def metrics(rows: list[dict], capital: Decimal) -> dict:
    ordered = sorted(rows, key=lambda row: row["exit_time"])
    equity, peak, max_drawdown = capital, capital, Decimal("0")
    curve: list[float] = []
    for row in ordered:
        equity += row["net_pnl"]
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak * 100 if peak else Decimal("0"))
        curve.append(float(equity))
    pnls = [row["net_pnl"] for row in ordered]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
    by_index: dict[str, Decimal] = {}
    by_moneyness: dict[str, Decimal] = {}
    by_option: dict[str, Decimal] = {}
    monthly: dict[str, Decimal] = {}
    by_exit: dict[str, Decimal] = {}
    loss_streak = max_loss_streak = 0
    for row in ordered:
        for target, key in ((by_index, row["underlying_code"]), (by_moneyness, row["moneyness"]), (by_option, row["option_type"])):
            target[key] = target.get(key, Decimal("0")) + row["net_pnl"]
        month = row["exit_time"].strftime("%Y-%m")
        monthly[month] = monthly.get(month, Decimal("0")) + row["net_pnl"]
        by_exit[row.get("exit_reason", "UNKNOWN")] = by_exit.get(row.get("exit_reason", "UNKNOWN"), Decimal("0")) + row["net_pnl"]
        loss_streak = loss_streak + 1 if row["net_pnl"] < 0 else 0
        max_loss_streak = max(max_loss_streak, loss_streak)
    total = sum(pnls, Decimal("0"))
    return {"net_pnl": float(total), "return_pct": float(total / capital * 100) if capital else 0, "max_drawdown_pct": float(max_drawdown), "profit_factor": float(sum(wins, Decimal("0")) / abs(sum(losses, Decimal("0")))) if losses else None, "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else 0, "expectancy": float(total / len(pnls)) if pnls else 0, "max_consecutive_losses": max_loss_streak, "trades": len(pnls), "equity_curve": curve, "per_index": {k: float(v) for k, v in by_index.items()}, "atm_itm1": {k: float(v) for k, v in by_moneyness.items()}, "ce_pe": {k: float(v) for k, v in by_option.items()}, "exit_reasons": {k: float(v) for k, v in by_exit.items()}, "monthly": {k: float(v) for k, v in monthly.items()}}


def _historical_contract(session: Session, provider: str, code: str, direction: str, day: date, spot: Decimal, policy: str = "ATM_ITM1") -> list[tuple[Instrument, str]]:
    available = session.scalars(select(Instrument).where(Instrument.provider == provider, Instrument.underlying_code == code, Instrument.instrument_kind == "OPTION", Instrument.option_type == direction, Instrument.metadata_verified.is_(True), Instrument.lot_size > 0, Instrument.expiry > day, Instrument.valid_from <= day, (Instrument.valid_to.is_(None) | (Instrument.valid_to >= day))).order_by(Instrument.expiry, Instrument.strike)).all()
    if not available:
        return []
    expiry = min(item.expiry for item in available)
    chain = [item for item in available if item.expiry == expiry]
    strikes = sorted({item.strike for item in chain})
    atm = min(strikes, key=lambda strike: abs(strike - spot))
    position = strikes.index(atm)
    itm = strikes[max(0, position - 1)] if direction == "CE" else strikes[min(len(strikes) - 1, position + 1)]
    result = [(next(item for item in chain if item.strike == atm), "ATM")]
    if policy == "ATM_ITM1" and itm != atm:
        result.append((next(item for item in chain if item.strike == itm), "ITM1"))
    return result


def _intrabar_underlying_outcome(
    session: Session,
    provider: str,
    code: str,
    bar: Candle,
    direction: str,
    stop: Decimal,
    target: Decimal,
) -> str | None:
    """Resolve an ambiguous five-minute stop/target with stored one-minute data."""
    rows = session.scalars(
        select(Candle)
        .where(
            Candle.provider == provider,
            Candle.underlying_code == code,
            Candle.data_role == "SPOT",
            Candle.timeframe == "1minute",
            Candle.timestamp >= bar.timestamp,
            Candle.timestamp < bar.timestamp + timedelta(minutes=5),
            Candle.source_verified.is_(True),
        )
        .order_by(Candle.timestamp)
    ).all()
    for row in rows:
        stop_hit = (direction == "CE" and row.low <= stop) or (direction == "PE" and row.high >= stop)
        target_hit = (direction == "CE" and row.high >= target) or (direction == "PE" and row.low <= target)
        if stop_hit:
            return "UNDERLYING_STOP"
        if target_hit:
            return "UNDERLYING_TARGET"
    return None


def run_backtest_job(db_url: str, run_id: int, model_path: str) -> None:
    engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {})
    with Session(engine) as session:
        run = session.get(BacktestRun, run_id)
        if not run: return
        run.status, run.progress_percent = "RUNNING", 1
        session.commit()
        warnings: set[str] = set()
        try:
            config = json.loads(run.configuration_json)
            provider = str(config.get("historical_provider", "UPSTOX")).upper()
            cost_model = str(config.get("cost_model", provider)).upper()
            strategy_mode = str(config.get("strategy_mode", "MODEL")).upper()
            strike_policy = str(config.get("strike_policy", "ATM_ITM1")).upper()
            entry_delay_bars = max(1, int(config.get("entry_delay_bars", 1)))
            minimum_option_volume = max(0, int(config.get("minimum_option_volume", 1)))
            maximum_spread_pct = Decimal(str(config.get("maximum_spread_pct", "1.5")))
            model = RegimeModel(Path(model_path)) if strategy_mode == "MODEL" else None
            if strategy_mode == "MODEL" and (model is None or not model.available):
                raise RuntimeError("Model unavailable. Download verified index candles and train the XGBoost model first.")
            training_end = model.metadata.get("training_end") if model else None
            if training_end and run.start_date <= date.fromisoformat(training_end):
                raise RuntimeError(
                    f"Backtest must start after the model training period ({training_end}) to prevent in-sample results"
                )
            events: list[dict] = []
            for code_index, code in enumerate(config["indices"]):
                candles = session.scalars(select(Candle).where(Candle.provider == provider, Candle.underlying_code == code, Candle.data_role == "SPOT", Candle.timeframe == "5minute", Candle.timestamp >= datetime.combine(run.start_date, time.min, tzinfo=UTC), Candle.timestamp <= datetime.combine(run.end_date, time.max, tzinfo=UTC), Candle.source_verified.is_(True)).order_by(Candle.timestamp)).all()
                if len(candles) < 60:
                    warnings.add(f"{code}: fewer than 60 verified candles")
                    continue
                future_candles = session.scalars(select(Candle).where(Candle.provider == provider, Candle.underlying_code == code, Candle.data_role == "FUTURE", Candle.timeframe == "5minute", Candle.timestamp >= datetime.combine(run.start_date, time.min, tzinfo=UTC), Candle.timestamp <= datetime.combine(run.end_date, time.max, tzinfo=UTC), Candle.source_verified.is_(True)).order_by(Candle.timestamp)).all()
                future_instruments = session.scalars(select(Instrument).where(Instrument.provider == provider, Instrument.underlying_code == code, Instrument.instrument_kind == "FUTURE")).all()
                futures_frame = front_future_frame(future_candles, future_instruments)
                if len(futures_frame) < 60:
                    warnings.add(f"{code}: fewer than 60 aligned futures candles")
                    continue
                frame = with_futures_features(
                    pd.DataFrame([{"timestamp": item.timestamp, "open": float(item.open), "high": float(item.high), "low": float(item.low), "close": float(item.close), "volume": item.volume} for item in candles]),
                    futures_frame,
                )
                for index in range(50, len(frame)):
                    row = frame.iloc[index]
                    stamp = pd.Timestamp(row.timestamp).to_pydatetime().astimezone(UTC)
                    source = candles[index]
                    decision = (
                        decide(row, {"india_vix": source.india_vix, "futures_basis_pct": source.futures_basis_pct, "pcr": source.pcr}, model)
                        if strategy_mode == "MODEL"
                        else rule_only_decide(row)
                    )
                    if decision.direction != "NO_TRADE":
                        events.append({"time": stamp, "code": code, "decision": decision, "row": row})
                run.progress_percent = 5 + int((code_index + 1) / len(config["indices"]) * 30)
                session.commit()
            events.sort(key=lambda item: (-item["decision"].confidence, item["code"]))
            events.sort(key=lambda item: item["time"])
            capital = Decimal(str(config["capital"]))
            limits = RiskLimits(capital, Decimal(str(config["risk_pct"])), Decimal("0.75"), 5, 2, 1, 1, Decimal("25"), Decimal("1.5"))
            recorded: list[dict] = []
            position_until: datetime | None = None
            daily: dict[date, dict[str, int | Decimal]] = {}
            consecutive_losses = 0
            for event in events:
                if position_until and event["time"] <= position_until: continue
                row, decision = event["row"], event["decision"]
                local_day = event["time"].astimezone(ZoneInfo("Asia/Kolkata")).date()
                day_state = daily.setdefault(local_day, {"trades": 0, "pnl": Decimal("0")})
                permitted, _ = can_enter(limits=limits, open_positions=0, trades_today=int(day_state["trades"]), consecutive_losses=consecutive_losses, realized_pnl_today=Decimal(day_state["pnl"]), quote_fresh=True, broker_connected=True, reconciled=True)
                if not permitted:
                    continue
                cutoff = datetime.combine(local_day, time(15, 20), tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
                contract_choices = _historical_contract(session, provider, event["code"], decision.direction, local_day, Decimal(str(row.close)), strike_policy)
                chosen = None
                for instrument, moneyness in contract_choices:
                    bars = session.scalars(select(Candle).where(Candle.provider == provider, Candle.instrument_key == instrument.instrument_key, Candle.data_role == "OPTION", Candle.timeframe == "5minute", Candle.timestamp > event["time"], Candle.timestamp <= cutoff, Candle.source_verified.is_(True)).order_by(Candle.timestamp)).all()
                    if len(bars) >= entry_delay_bars:
                        bars = bars[entry_delay_bars - 1 :]
                        if bars[0].volume < minimum_option_volume:
                            warnings.add(f"{event['code']}: option entry rejected below minimum volume")
                            continue
                        has_bid_ask = bars[0].ask is not None and bars[0].bid is not None and all(bar.bid is not None for bar in bars)
                        if has_bid_ask:
                            midpoint = (Decimal(bars[0].ask) + Decimal(bars[0].bid)) / 2
                            spread_pct = (Decimal(bars[0].ask) - Decimal(bars[0].bid)) / midpoint * 100 if midpoint else Decimal("999")
                            if spread_pct > maximum_spread_pct:
                                warnings.add(f"{event['code']}: option entry rejected above maximum spread")
                                continue
                        chosen = instrument, moneyness, bars, has_bid_ask
                        break
                if not chosen:
                    warnings.add(f"{event['code']}: verified option candles unavailable for some signals")
                    continue
                instrument, moneyness, bars, has_bid_ask = chosen
                base_slip_pct = Decimal(str(config["slippage_pct"]))
                liquidity_multiplier = Decimal("1")
                if minimum_option_volume and bars[0].volume < minimum_option_volume * 5:
                    liquidity_multiplier += Decimal(minimum_option_volume * 5 - bars[0].volume) / Decimal(minimum_option_volume * 5)
                effective_slip_pct = min(base_slip_pct * liquidity_multiplier, base_slip_pct * 3)
                slip = effective_slip_pct / 100
                entry_base = Decimal(bars[0].ask) if has_bid_ask else Decimal(bars[0].open)
                entry = entry_base * (1 + slip)
                if not has_bid_ask:
                    warnings.add("Real option OHLC was used because historical bid/ask was unavailable; configured slippage was applied on both sides")
                plan = position_size(entry, instrument.lot_size, limits, "BACKTEST")
                if not plan.allowed: continue
                underlying_risk = Decimal(str(row.atr))
                underlying_stop = Decimal(str(row.close)) - underlying_risk if decision.direction == "CE" else Decimal(str(row.close)) + underlying_risk
                underlying_target = Decimal(str(row.close)) + underlying_risk * limits.reward_risk if decision.direction == "CE" else Decimal(str(row.close)) - underlying_risk * limits.reward_risk
                exit_bar, exit_reason = bars[-1], "TIME_EXIT"
                exit_base = Decimal(exit_bar.bid) if has_bid_ask else Decimal(exit_bar.close)
                exit_price = exit_base * (1 - slip)
                underlying_bars = session.scalars(
                    select(Candle)
                    .where(
                        Candle.provider == provider,
                        Candle.underlying_code == event["code"],
                        Candle.data_role == "SPOT",
                        Candle.timeframe == "5minute",
                        Candle.timestamp >= bars[0].timestamp,
                        Candle.timestamp <= bars[-1].timestamp,
                    )
                    .order_by(Candle.timestamp)
                ).all()
                underlying_by_time = {bar.timestamp.replace(tzinfo=UTC) if bar.timestamp.tzinfo is None else bar.timestamp.astimezone(UTC): bar for bar in underlying_bars}
                resolution_timeframe = "5minute"
                for bar in bars[1:]:
                    bar_time = bar.timestamp.replace(tzinfo=UTC) if bar.timestamp.tzinfo is None else bar.timestamp.astimezone(UTC)
                    underlying = underlying_by_time.get(bar_time)
                    premium_stop_hit = Decimal(bar.low) <= plan.premium_stop
                    stop_hit = underlying and ((decision.direction == "CE" and underlying.low <= underlying_stop) or (decision.direction == "PE" and underlying.high >= underlying_stop))
                    target_hit = underlying and ((decision.direction == "CE" and underlying.high >= underlying_target) or (decision.direction == "PE" and underlying.low <= underlying_target))
                    if underlying and stop_hit and target_hit:
                        outcome = _intrabar_underlying_outcome(session, provider, event["code"], underlying, decision.direction, underlying_stop, underlying_target)
                        if outcome:
                            resolution_timeframe = "1minute"
                            stop_hit = outcome == "UNDERLYING_STOP"
                            target_hit = outcome == "UNDERLYING_TARGET"
                        else:
                            warnings.add("Stop and target touched in one five-minute candle; conservative stop-first fallback applied")
                    if premium_stop_hit or stop_hit:
                        exit_bar, exit_reason = bar, "PREMIUM_STOP" if premium_stop_hit else "UNDERLYING_STOP"
                        observed_exit = Decimal(bar.bid) if has_bid_ask else Decimal(bar.close)
                        exit_base = min(observed_exit, plan.premium_stop) if premium_stop_hit else observed_exit
                        exit_price = exit_base * (1 - slip)
                        break
                    if target_hit:
                        exit_bar, exit_reason = bar, "UNDERLYING_TARGET"
                        exit_base = Decimal(bar.bid) if has_bid_ask else Decimal(bar.close)
                        exit_price = exit_base * (1 - slip)
                        break
                gross = (exit_price - entry) * plan.quantity
                charges = option_charges(
                    entry, exit_price, plan.quantity, cost_model, local_day, instrument.exchange
                )
                net = gross - charges
                session.add(BacktestTrade(provider=provider, run_id=run.id, signal_time=event["time"], entry_time=bars[0].timestamp, exit_time=exit_bar.timestamp, underlying_code=event["code"], instrument_key=instrument.instrument_key, option_type=decision.direction, moneyness=moneyness, quantity=plan.quantity, entry_price=entry, exit_price=exit_price, gross_pnl=gross, charges=charges, net_pnl=net, exit_reason=exit_reason, effective_slippage_pct=effective_slip_pct, metadata_source=instrument.metadata_source, resolution_timeframe=resolution_timeframe))
                doubled_entry = entry_base * (1 + slip * 2)
                doubled_exit = exit_base * (1 - slip * 2)
                stress_net = (doubled_exit - doubled_entry) * plan.quantity - option_charges(
                    doubled_entry,
                    doubled_exit,
                    plan.quantity,
                    cost_model,
                    local_day,
                    instrument.exchange,
                )
                recorded.append({"exit_time": exit_bar.timestamp, "underlying_code": event["code"], "option_type": decision.direction, "moneyness": moneyness, "net_pnl": net, "stress_net_pnl": stress_net, "exit_reason": exit_reason})
                day_state["trades"] = int(day_state["trades"]) + 1
                day_state["pnl"] = Decimal(day_state["pnl"]) + net
                consecutive_losses = consecutive_losses + 1 if net < 0 else 0
                position_until = exit_bar.timestamp
                if events:
                    run.progress_percent = min(95, 35 + int(len(recorded) / len(events) * 60))
                    session.commit()
            summary = metrics(recorded, capital)
            summary["double_slippage_net_pnl"] = float(sum((row["stress_net_pnl"] for row in recorded), Decimal("0")))
            run.summary_json = json.dumps(summary)
            run.warnings_json = json.dumps(sorted(warnings))
            run.status, run.progress_percent, run.completed_at = "COMPLETE", 100, datetime.now(UTC)
            session.commit()
        except Exception as exc:
            run.status, run.error_message, run.completed_at = "FAILED", str(exc), datetime.now(UTC)
            session.commit()


class BacktestService:
    def __init__(self, db_url: str, model_path: Path) -> None:
        self.db_url, self.model_path = db_url, model_path
        self.pool = ProcessPoolExecutor(max_workers=1)
        self.futures: set[Future] = set()

    def submit(self, run_id: int) -> None:
        future = self.pool.submit(run_backtest_job, self.db_url, run_id, str(self.model_path))
        self.futures.add(future)
        future.add_done_callback(self.futures.discard)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def trades_csv(rows: list[BacktestTrade]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["provider", "signal_time", "entry_time", "exit_time", "index", "instrument", "option_type", "moneyness", "quantity", "entry_price", "exit_price", "gross_pnl", "charges", "net_pnl", "exit_reason", "slippage_pct", "metadata_source", "resolution_timeframe"])
    for row in rows:
        writer.writerow([row.provider, row.signal_time, row.entry_time, row.exit_time, row.underlying_code, row.instrument_key, row.option_type, row.moneyness, row.quantity, row.entry_price, row.exit_price, row.gross_pnl, row.charges, row.net_pnl, row.exit_reason, row.effective_slippage_pct, row.metadata_source, row.resolution_timeframe])
    return output.getvalue()

