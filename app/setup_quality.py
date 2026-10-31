from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from .models import Candle, Instrument
from .risk import option_charges
from .strategy import with_futures_features
from .training import front_future_frame, model_metadata_path


SETUP_FEATURE_SCHEMA = "setup-quality-v2"
SETUP_INDICES = ("NIFTY", "BANK_NIFTY", "SENSEX")
RELATIVE_INDEX = {
    "NIFTY": "BANK_NIFTY",
    "BANK_NIFTY": "NIFTY",
    "SENSEX": "NIFTY",
}
SETUP_FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ema9_slope",
    "ema20_slope",
    "ema50_slope",
    "ema9_20_atr",
    "vwap_distance_atr",
    "opening_range_distance_atr",
    "day_high_distance_atr",
    "day_low_distance_atr",
    "atr_pct",
    "atr_change",
    "futures_basis_pct",
    "basis_change_1",
    "basis_change_3",
    "opening_gap_pct",
    "india_vix",
    "india_vix_change_3",
    "india_vix_available",
    "relative_strength_15m",
    "relative_strength_30m",
    "previous_day_range_position",
    "days_to_futures_expiry",
    "day_of_week_sin",
    "day_of_week_cos",
    "time_sin",
    "time_cos",
    "minutes_to_close",
    "time_bucket_open",
    "time_bucket_morning",
    "time_bucket_midday",
    "time_bucket_afternoon",
    "direction_ce",
]


def relative_index_code(underlying_code: str) -> str:
    """Return the reference index used for cross-index momentum features."""
    return RELATIVE_INDEX[underlying_code]


@dataclass(frozen=True)
class SetupContract:
    stop_atr: float = 1.0
    target_atr: float = 1.5
    timeout_bars: int = 9
    estimated_round_trip_cost_bps: float = 10.0
    entry_delay_bars: int = 1


def _as_utc_frame(rows: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": row.timestamp,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": row.volume,
                "india_vix": row.india_vix,
            }
            for row in rows
        ]
    )


def _setup_feature_frame(
    spot_frame: pd.DataFrame,
    futures_frame: pd.DataFrame,
    relative_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = with_futures_features(spot_frame, futures_frame)
    local = frame.timestamp.dt.tz_convert("Asia/Kolkata")
    day = local.dt.date
    close = frame.close.replace(0, np.nan)
    atr = frame.atr.replace(0, np.nan)

    frame["ema20_slope"] = frame.ema20.groupby(day).pct_change(3).fillna(0.0)
    frame["ema50_slope"] = frame.ema50.groupby(day).pct_change(3).fillna(0.0)
    frame["ema9_20_atr"] = (frame.ema9 - frame.ema20) / atr
    frame["vwap_distance_atr"] = (frame.futures_close - frame.futures_vwap) / atr
    frame["day_high_so_far"] = frame.high.groupby(day).cummax()
    frame["day_low_so_far"] = frame.low.groupby(day).cummin()
    frame["day_high_distance_atr"] = (frame.day_high_so_far - frame.close) / atr
    frame["day_low_distance_atr"] = (frame.close - frame.day_low_so_far) / atr
    frame["atr_pct"] = frame.atr / close
    frame["basis_change_1"] = frame.futures_basis_pct.groupby(day).diff().fillna(0.0)
    frame["basis_change_3"] = frame.futures_basis_pct.groupby(day).diff(3).fillna(0.0)

    daily_open = frame.open.groupby(day).transform("first")
    frame["opening_gap_pct"] = (daily_open - frame.previous_day_close) / frame.previous_day_close.replace(0, np.nan)
    previous_range = (frame.previous_day_high - frame.previous_day_low).replace(0, np.nan)
    frame["previous_day_range_position"] = (frame.close - frame.previous_day_low) / previous_range
    raw_vix = (
        pd.to_numeric(frame["india_vix"], errors="coerce")
        if "india_vix" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    frame["india_vix_available"] = raw_vix.notna().astype(float)
    causal_vix = raw_vix.groupby(day).ffill()
    frame["india_vix"] = causal_vix.fillna(0.0)
    frame["india_vix_change_3"] = causal_vix.groupby(day).pct_change(3).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    if "expiry" in futures_frame.columns:
        expiry = futures_frame[["timestamp", "expiry"]].copy()
        expiry["timestamp"] = pd.to_datetime(expiry.timestamp, utc=True)
        frame = frame.merge(expiry, on="timestamp", how="left", validate="one_to_one")
        frame["days_to_futures_expiry"] = [
            max(0, (value - trading_day).days) if pd.notna(value) else 0
            for value, trading_day in zip(frame.expiry, day)
        ]
    else:
        frame["days_to_futures_expiry"] = 0.0

    if relative_frame is not None and not relative_frame.empty:
        relative = relative_frame.copy().sort_values("timestamp")
        relative["timestamp"] = pd.to_datetime(relative.timestamp, utc=True)
        relative_local = relative.timestamp.dt.tz_convert("Asia/Kolkata")
        relative_day = relative_local.dt.date
        relative["relative_return_3"] = relative.close.groupby(relative_day).pct_change(3)
        relative["relative_return_6"] = relative.close.groupby(relative_day).pct_change(6)
        frame = frame.merge(
            relative[["timestamp", "relative_return_3", "relative_return_6"]],
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
        frame["relative_strength_15m"] = frame.return_3 - frame.relative_return_3
        frame["relative_strength_30m"] = frame.return_6 - frame.relative_return_6
    else:
        frame["relative_strength_15m"] = 0.0
        frame["relative_strength_30m"] = 0.0

    minutes_from_open = local.dt.hour * 60 + local.dt.minute - 555
    session_fraction = minutes_from_open.clip(lower=0, upper=370) / 370.0
    frame["minutes_to_close"] = (930 - (local.dt.hour * 60 + local.dt.minute)).clip(lower=0)
    frame["day_of_week_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 5.0)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 5.0)
    frame["time_sin"] = np.sin(2 * np.pi * session_fraction)
    frame["time_cos"] = np.cos(2 * np.pi * session_fraction)
    frame["time_bucket_open"] = (minutes_from_open < 60).astype(float)
    frame["time_bucket_morning"] = ((minutes_from_open >= 60) & (minutes_from_open < 150)).astype(float)
    frame["time_bucket_midday"] = ((minutes_from_open >= 150) & (minutes_from_open < 270)).astype(float)
    frame["time_bucket_afternoon"] = (minutes_from_open >= 270).astype(float)
    return frame


def _candidate_feature_values(row: pd.Series, direction: str) -> dict[str, float]:
    opening_level = float(
        row.opening_range_high if direction == "CE" else row.opening_range_low
    )
    return {
        "return_1": float(row.return_1),
        "return_3": float(row.return_3),
        "return_6": float(row.return_6),
        "return_12": float(row.return_12),
        "ema9_slope": float(row.ema9_slope),
        "ema20_slope": float(row.ema20_slope),
        "ema50_slope": float(row.ema50_slope),
        "ema9_20_atr": float(row.ema9_20_atr),
        "vwap_distance_atr": float(row.vwap_distance_atr),
        "opening_range_distance_atr": (float(row.close) - opening_level) / float(row.atr),
        "day_high_distance_atr": float(row.day_high_distance_atr),
        "day_low_distance_atr": float(row.day_low_distance_atr),
        "atr_pct": float(row.atr_pct),
        "atr_change": float(row.atr_change),
        "futures_basis_pct": float(row.futures_basis_pct),
        "basis_change_1": float(row.basis_change_1),
        "basis_change_3": float(row.basis_change_3),
        "opening_gap_pct": float(row.opening_gap_pct),
        "india_vix": float(row.india_vix),
        "india_vix_change_3": float(row.india_vix_change_3),
        "india_vix_available": float(row.india_vix_available),
        "relative_strength_15m": float(row.relative_strength_15m),
        "relative_strength_30m": float(row.relative_strength_30m),
        "previous_day_range_position": float(row.previous_day_range_position),
        "days_to_futures_expiry": float(row.days_to_futures_expiry),
        "day_of_week_sin": float(row.day_of_week_sin),
        "day_of_week_cos": float(row.day_of_week_cos),
        "time_sin": float(row.time_sin),
        "time_cos": float(row.time_cos),
        "minutes_to_close": float(row.minutes_to_close),
        "time_bucket_open": float(row.time_bucket_open),
        "time_bucket_morning": float(row.time_bucket_morning),
        "time_bucket_midday": float(row.time_bucket_midday),
        "time_bucket_afternoon": float(row.time_bucket_afternoon),
        "direction_ce": 1.0 if direction == "CE" else 0.0,
    }


def _rule_candidates_from_frame(frame: pd.DataFrame, underlying_code: str) -> pd.DataFrame:
    local_days = frame.timestamp.dt.tz_convert("Asia/Kolkata").dt.date.to_numpy()
    records: list[dict] = []
    active_day: date | None = None
    bullish_active = False
    bearish_active = False
    for position, row in frame.iterrows():
        row_day = local_days[position]
        if row_day != active_day:
            active_day = row_day
            bullish_active = False
            bearish_active = False
        bullish_now = (
            float(row.futures_close) > float(row.futures_vwap)
            and float(row.ema9) > float(row.ema20)
            and float(row.close) > float(row.opening_range_high)
        )
        bearish_now = (
            float(row.futures_close) < float(row.futures_vwap)
            and float(row.ema9) < float(row.ema20)
            and float(row.close) < float(row.opening_range_low)
        )
        bullish = bullish_now and not bullish_active
        bearish = bearish_now and not bearish_active
        bullish_active = bullish_now
        bearish_active = bearish_now
        if not bullish and not bearish:
            continue
        if not np.isfinite(float(row.get("atr", np.nan))) or float(row.atr) <= 0:
            continue
        direction = "CE" if bullish else "PE"
        features = _candidate_feature_values(row, direction)
        if not np.isfinite(np.asarray(list(features.values()), dtype=float)).all():
            continue
        records.append(
            {
                **features,
                "timestamp": pd.Timestamp(row.timestamp),
                "underlying_code": underlying_code,
                "direction": direction,
                "setup_type": (
                    "OPENING_RANGE_BREAKOUT" if direction == "CE" else "OPENING_RANGE_BREAKDOWN"
                ),
                "source_index": position,
                "signal_close": float(row.close),
                "signal_atr": float(row.atr),
            }
        )
    return pd.DataFrame(records)


def setup_rule_candidates(
    spot_frame: pd.DataFrame,
    futures_frame: pd.DataFrame,
    underlying_code: str,
    *,
    relative_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return causal CE/PE candidates without inspecting any future candle."""
    return _rule_candidates_from_frame(
        _setup_feature_frame(spot_frame, futures_frame, relative_frame), underlying_code
    )


def setup_candidates(
    spot_frame: pd.DataFrame,
    futures_frame: pd.DataFrame,
    underlying_code: str,
    *,
    relative_frame: pd.DataFrame | None = None,
    contract: SetupContract | None = None,
) -> pd.DataFrame:
    """Create causal rule candidates and conservative triple-barrier labels."""
    contract = contract or SetupContract()
    if contract.entry_delay_bars < 1 or contract.timeout_bars < 1:
        raise ValueError("Entry delay and timeout must be positive")
    frame = _setup_feature_frame(spot_frame, futures_frame, relative_frame)
    local_days = frame.timestamp.dt.tz_convert("Asia/Kolkata").dt.date.to_numpy()
    candidates = _rule_candidates_from_frame(frame, underlying_code)
    records: list[dict] = []
    for candidate in candidates.to_dict("records"):
        position = int(candidate["source_index"])
        direction = str(candidate["direction"])
        entry_position = position + contract.entry_delay_bars
        final_position = entry_position + contract.timeout_bars - 1
        if entry_position >= len(frame) or local_days[entry_position] != local_days[position]:
            continue
        path = frame.iloc[entry_position : min(final_position + 1, len(frame))]
        path = path[path.timestamp.dt.tz_convert("Asia/Kolkata").dt.date == local_days[position]]
        if path.empty:
            continue
        entry_price = float(path.iloc[0].open)
        risk = float(candidate["signal_atr"]) * contract.stop_atr
        reward = float(candidate["signal_atr"]) * contract.target_atr
        stop = entry_price - risk if direction == "CE" else entry_price + risk
        target = entry_price + reward if direction == "CE" else entry_price - reward
        exit_price = float(path.iloc[-1].close)
        exit_timestamp = pd.Timestamp(path.iloc[-1].timestamp)
        exit_reason = "TIMEOUT"
        bars_held = len(path)
        ambiguous = False
        for offset, (_, future) in enumerate(path.iterrows(), start=1):
            stop_hit = float(future.low) <= stop if direction == "CE" else float(future.high) >= stop
            target_hit = float(future.high) >= target if direction == "CE" else float(future.low) <= target
            if stop_hit and target_hit:
                ambiguous = True
                break
            if stop_hit:
                exit_price, exit_reason, bars_held = stop, "STOP", offset
                exit_timestamp = pd.Timestamp(future.timestamp)
                break
            if target_hit:
                exit_price, exit_reason, bars_held = target, "TARGET", offset
                exit_timestamp = pd.Timestamp(future.timestamp)
                break
        if ambiguous:
            continue
        signed_return = (
            (exit_price - entry_price) / entry_price
            if direction == "CE"
            else (entry_price - exit_price) / entry_price
        )
        net_return = signed_return - contract.estimated_round_trip_cost_bps / 10_000.0
        candidate.update(
            {
                "entry_timestamp": pd.Timestamp(path.iloc[0].timestamp),
                "exit_timestamp": exit_timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop,
                "target_price": target,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "gross_return": signed_return,
                "net_return": net_return,
                "label": "TAKE" if exit_reason == "TARGET" and net_return > 0 else "SKIP",
            }
        )
        records.append(candidate)
    return pd.DataFrame(records)


def _naive_utc(value) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.tz_localize(None).to_pydatetime()


def option_price_labels(
    session: Session,
    dataset: pd.DataFrame,
    *,
    provider: str,
    cost_model: str,
    slippage_pct: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """Replace proxy returns with verified ATM→ITM1 option-candle outcomes.

    The underlying triple barrier still determines exit time. Option entry uses
    the next option candle and option exit uses the candle aligned to the barrier
    event. Bid/ask is preferred; otherwise OHLC receives two-sided slippage.
    """
    if dataset.empty:
        return dataset.copy(), {"candidates": 0, "option_labelled": 0, "coverage": 0.0}
    start = min(_naive_utc(value) for value in dataset.entry_timestamp)
    end = max(_naive_utc(value) for value in dataset.exit_timestamp)
    codes = sorted(set(dataset.underlying_code))
    instruments = session.scalars(
        select(Instrument).where(
            Instrument.provider == provider,
            Instrument.underlying_code.in_(codes),
            Instrument.instrument_kind == "OPTION",
            Instrument.metadata_verified.is_(True),
            Instrument.lot_size > 0,
        )
    ).all()
    instrument_by_key = {item.instrument_key: item for item in instruments}
    candles = session.scalars(
        select(Candle).where(
            Candle.provider == provider,
            Candle.underlying_code.in_(codes),
            Candle.data_role == "OPTION",
            Candle.timeframe == "5minute",
            Candle.timestamp >= start,
            Candle.timestamp <= end,
            Candle.source_verified.is_(True),
        )
    ).all()
    candle_by_key = {
        (item.instrument_key, _naive_utc(item.timestamp)): item
        for item in candles
        if item.instrument_key in instrument_by_key
    }
    slip = Decimal(str(slippage_pct)) / Decimal("100")
    labelled: list[dict] = []
    rejection_counts: Counter = Counter()
    missing_examples: list[dict[str, str]] = []
    for _, row in dataset.iterrows():
        trading_day = pd.Timestamp(row.entry_timestamp).tz_convert("Asia/Kolkata").date()
        chain = [
            item
            for item in instruments
            if item.underlying_code == row.underlying_code
            and item.option_type == row.direction
            and item.expiry > trading_day
            and item.valid_from <= trading_day
            and (item.valid_to is None or item.valid_to >= trading_day)
        ]
        if not chain:
            rejection_counts["NO_VERIFIED_CONTRACT"] += 1
            if len(missing_examples) < 25:
                missing_examples.append({"date": trading_day.isoformat(), "direction": str(row.direction), "reason": "NO_VERIFIED_CONTRACT", "entry_time": str(row.entry_timestamp)})
            continue
        expiry = min(item.expiry for item in chain)
        chain = [item for item in chain if item.expiry == expiry]
        strikes = sorted({item.strike for item in chain})
        spot = Decimal(str(row.entry_price))
        atm = min(strikes, key=lambda strike: abs(strike - spot))
        position = strikes.index(atm)
        itm = (
            strikes[max(0, position - 1)]
            if row.direction == "CE"
            else strikes[min(len(strikes) - 1, position + 1)]
        )
        selected = None
        for strike, moneyness in dict.fromkeys(((atm, "ATM"), (itm, "ITM1"))):
            instrument = next(item for item in chain if item.strike == strike)
            entry_bar = candle_by_key.get((instrument.instrument_key, _naive_utc(row.entry_timestamp)))
            exit_bar = candle_by_key.get((instrument.instrument_key, _naive_utc(row.exit_timestamp)))
            if entry_bar is not None and exit_bar is not None:
                selected = instrument, moneyness, entry_bar, exit_bar
                break
        if selected is None:
            rejection_counts["OPTION_ENTRY_OR_EXIT_BAR_MISSING"] += 1
            if len(missing_examples) < 25:
                missing_examples.append({"date": trading_day.isoformat(), "direction": str(row.direction), "reason": "OPTION_ENTRY_OR_EXIT_BAR_MISSING", "expiry": expiry.isoformat(), "atm": str(atm), "itm1": str(itm), "entry_time": str(row.entry_timestamp), "exit_time": str(row.exit_timestamp)})
            continue
        instrument, moneyness, entry_bar, exit_bar = selected
        has_bid_ask = (
            entry_bar.ask is not None
            and entry_bar.bid is not None
            and exit_bar.bid is not None
            and Decimal(entry_bar.ask) > 0
            and Decimal(entry_bar.bid) > 0
            and Decimal(exit_bar.bid) > 0
        )
        entry_base = Decimal(entry_bar.ask) if has_bid_ask else Decimal(entry_bar.open)
        exit_base = Decimal(exit_bar.bid) if has_bid_ask else Decimal(exit_bar.close)
        option_entry = entry_base * (Decimal("1") + slip)
        option_exit = exit_base * (Decimal("1") - slip)
        gross = (option_exit - option_entry) * instrument.lot_size
        charges = option_charges(
            option_entry,
            option_exit,
            instrument.lot_size,
            cost_model,
            trading_day,
            instrument.exchange,
        )
        net = gross - charges
        notional = option_entry * instrument.lot_size
        result = row.to_dict()
        result.update(
            {
                "option_instrument_key": instrument.instrument_key,
                "option_moneyness": moneyness,
                "option_expiry": instrument.expiry,
                "option_entry": float(option_entry),
                "option_exit": float(option_exit),
                "option_gross_pnl": float(gross),
                "option_charges": float(charges),
                "option_net_pnl": float(net),
                "option_bid_ask_available": has_bid_ask,
                "net_return": float(net / notional) if notional else 0.0,
                "label": "TAKE" if row.exit_reason == "TARGET" and net > 0 else "SKIP",
                "label_source": "VERIFIED_OPTION_CANDLES",
            }
        )
        labelled.append(result)
    result_frame = pd.DataFrame(labelled)
    return result_frame, {
        "candidates": len(dataset),
        "option_labelled": len(result_frame),
        "coverage": round(len(result_frame) / len(dataset), 4),
        "rejections": dict(rejection_counts),
        "missing_examples": missing_examples,
        "bid_ask_rows": int(result_frame.option_bid_ask_available.sum()) if not result_frame.empty else 0,
    }


def _profit_metrics(returns: np.ndarray) -> dict[str, float | int | None]:
    returns = np.asarray(returns, dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    equity = np.cumsum(returns)
    running_peak = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = running_peak[1:] - equity if len(equity) else np.array([], dtype=float)
    return {
        "trades": int(len(returns)),
        "net_return": round(float(returns.sum()), 6),
        "expectancy": round(float(returns.mean()), 6) if len(returns) else 0.0,
        "win_rate": round(float((returns > 0).mean()), 4) if len(returns) else 0.0,
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 4) if len(losses) else None,
        "maximum_drawdown": round(float(drawdown.max()), 6) if len(drawdown) else 0.0,
    }


def selection_metrics(dataset: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    accepted = probabilities >= threshold
    actual_take = dataset.label.to_numpy() == "TAKE"
    selected = dataset.loc[accepted]
    local_month = pd.to_datetime(selected.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m")
    monthly = {
        str(month): round(float(values.sum()), 6)
        for month, values in selected.net_return.groupby(local_month)
    }
    positive_month_profit = sum(max(0.0, value) for value in monthly.values())
    largest_month_profit_fraction = (
        max((max(0.0, value) for value in monthly.values()), default=0.0) / positive_month_profit
        if positive_month_profit > 0
        else 0.0
    )
    confidence_bands: list[dict] = []
    for low, high in ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        mask = (probabilities >= low) & (probabilities < high)
        confidence_bands.append(
            {
                "band": f"{int(low * 100)}-{int(min(high, 1.0) * 100)}%",
                "candidates": int(mask.sum()),
                "take_precision": round(float(actual_take[mask].mean()), 4) if mask.any() else None,
            }
        )
    return {
        "threshold": round(float(threshold), 4),
        "candidates": int(len(dataset)),
        "accepted": int(accepted.sum()),
        "coverage": round(float(accepted.mean()), 4) if len(accepted) else 0.0,
        "take_precision": round(float(actual_take[accepted].mean()), 4) if accepted.any() else None,
        "selected": _profit_metrics(selected.net_return.to_numpy(dtype=float)),
        "rules_only": _profit_metrics(dataset.net_return.to_numpy(dtype=float)),
        "skipped": _profit_metrics(dataset.loc[~accepted, "net_return"].to_numpy(dtype=float)),
        "monthly_net_return": monthly,
        "positive_month_fraction": round(
            sum(value > 0 for value in monthly.values()) / len(monthly), 4
        ) if monthly else 0.0,
        "largest_month_profit_fraction": round(largest_month_profit_fraction, 4),
        "by_direction": {
            direction: _profit_metrics(group.net_return.to_numpy(dtype=float))
            for direction, group in selected.groupby("direction")
        },
        "confidence_bands": confidence_bands,
    }


def select_probability_threshold(dataset: pd.DataFrame, probabilities: np.ndarray) -> float:
    minimum_trades = max(20, int(math.ceil(len(dataset) * 0.10)))
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.arange(0.50, 0.81, 0.02):
        result = selection_metrics(dataset, probabilities, float(threshold))
        selected = result["selected"]
        precision = result["take_precision"]
        if result["accepted"] < minimum_trades or precision is None:
            continue
        profit_factor = selected["profit_factor"]
        candidates.append(
            (
                float(profit_factor) if profit_factor is not None else 999.0,
                float(selected["net_return"]),
                float(threshold),
            )
        )
    return max(candidates)[2] if candidates else 0.60


def _apply_binary_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def _binary_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best_temperature, best_loss = 1.0, float("inf")
    labels = np.asarray(labels, dtype=float)
    for temperature in np.linspace(0.5, 3.0, 51):
        calibrated = _apply_binary_temperature(probabilities, float(temperature))
        loss = -float(
            (labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)).mean()
        )
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return best_temperature


def setup_quality_issues(metrics: dict) -> list[str]:
    issues: list[str] = []
    selected = metrics.get("selected") or {}
    if int(metrics.get("accepted") or 0) < 100:
        issues.append("fewer than 100 holdout candidates were accepted")
    if metrics.get("take_precision") is None or float(metrics["take_precision"]) < 0.55:
        issues.append("TAKE precision is below 55%")
    if selected.get("profit_factor") is None or float(selected["profit_factor"]) < 1.20:
        issues.append("selected-trade profit factor is below 1.20")
    if float(selected.get("expectancy") or 0.0) <= 0:
        issues.append("selected-trade expectancy is not positive")
    if float(metrics.get("positive_month_fraction") or 0.0) < 0.60:
        issues.append("fewer than 60% of active holdout months are profitable")
    if float(metrics.get("largest_month_profit_fraction") or 0.0) > 0.50:
        issues.append("one holdout month contributes more than 50% of positive profit")
    return issues


def walk_forward_validation(
    dataset: pd.DataFrame,
    *,
    model_params: dict,
    boosting_rounds: int,
) -> list[dict]:
    """Run expanding chronological folds with a fixed 60% TAKE threshold."""
    import xgboost as xgb

    local_days = pd.to_datetime(dataset.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    unique_days = np.asarray(sorted(set(local_days)))
    folds: list[dict] = []
    for train_fraction in (0.55, 0.70, 0.82):
        train_days = max(20, int(len(unique_days) * train_fraction))
        validation_days = max(5, int(len(unique_days) * 0.10))
        validation_end = min(len(unique_days), train_days + validation_days)
        if train_days >= len(unique_days) or validation_end - train_days < 3:
            continue
        train_mask = local_days <= unique_days[train_days - 1]
        validation_mask = (local_days >= unique_days[train_days]) & (
            local_days <= unique_days[validation_end - 1]
        )
        train = dataset.loc[train_mask]
        validation = dataset.loc[validation_mask]
        if len(train) < 100 or len(validation) < 20 or train.label.nunique() < 2:
            continue
        train_y = (train.label == "TAKE").to_numpy(dtype=int)
        counts = Counter(train_y.tolist())
        weights = np.asarray(
            [len(train_y) / (len(counts) * counts[value]) for value in train_y], dtype=float
        )
        model = xgb.train(
            model_params,
            xgb.DMatrix(
                train[SETUP_FEATURES].to_numpy(dtype=float),
                label=train_y,
                weight=weights,
                feature_names=SETUP_FEATURES,
            ),
            num_boost_round=boosting_rounds,
            verbose_eval=False,
        )
        probabilities = model.predict(
            xgb.DMatrix(
                validation[SETUP_FEATURES].to_numpy(dtype=float),
                feature_names=SETUP_FEATURES,
            )
        )
        metrics = selection_metrics(validation, probabilities, 0.60)
        folds.append(
            {
                "training_end": str(unique_days[train_days - 1]),
                "validation_start": str(unique_days[train_days]),
                "validation_end": str(unique_days[validation_end - 1]),
                "training_candidates": len(train),
                "validation_candidates": len(validation),
                "accepted": metrics["accepted"],
                "take_precision": metrics["take_precision"],
                "profit_factor": metrics["selected"]["profit_factor"],
                "expectancy": metrics["selected"]["expectancy"],
            }
        )
    return folds


class SetupQualityModel:
    def __init__(self, manifest_path: Path) -> None:
        self.models: dict[str, object] = {}
        self.thresholds: dict[str, float] = {}
        self.temperatures: dict[str, float] = {}
        self.metadata: dict = {}
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if manifest.get("feature_schema") != SETUP_FEATURE_SCHEMA:
            return
        import xgboost as xgb

        self.metadata = manifest
        for code, detail in (manifest.get("index_models") or {}).items():
            path = manifest_path.parent / str(detail["file"])
            if not path.exists():
                self.models.clear()
                return
            booster = xgb.Booster()
            booster.load_model(path)
            self.models[code] = booster
            self.thresholds[code] = float(detail["threshold"])
            self.temperatures[code] = float(detail.get("probability_temperature", 1.0))

    @property
    def available(self) -> bool:
        return bool(self.models)

    def predict(self, underlying_code: str, values: dict[str, float]) -> tuple[str, float]:
        model = self.models.get(underlying_code)
        if model is None:
            return "SKIP", 0.0
        matrix = np.asarray([[values[name] for name in SETUP_FEATURES]], dtype=float)
        if not np.isfinite(matrix).all():
            return "SKIP", 0.0
        import xgboost as xgb

        raw = float(model.predict(xgb.DMatrix(matrix, feature_names=SETUP_FEATURES))[0])
        probability = float(
            _apply_binary_temperature(np.asarray([raw]), self.temperatures[underlying_code])[0]
        )
        return ("TAKE" if probability >= self.thresholds[underlying_code] else "SKIP", probability)


def _load_setup_frames(
    session: Session, indices: list[str], fit_start: date, holdout_end: date, provider: str
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], dict[str, dict[str, int]]]:
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    coverage: dict[str, dict[str, int]] = {}
    requested_codes = list(dict.fromkeys([*indices, *(relative_index_code(code) for code in indices)]))
    for code in requested_codes:
        start_stamp = datetime.combine(fit_start, time.min, tzinfo=UTC)
        end_stamp = datetime.combine(holdout_end, time.max, tzinfo=UTC)
        spot_rows = session.scalars(
            select(Candle).where(
                Candle.provider == provider,
                Candle.underlying_code == code,
                Candle.data_role == "SPOT",
                Candle.timeframe == "5minute",
                Candle.timestamp >= start_stamp,
                Candle.timestamp <= end_stamp,
                Candle.source_verified.is_(True),
            ).order_by(Candle.timestamp)
        ).all()
        future_rows = session.scalars(
            select(Candle).where(
                Candle.provider == provider,
                Candle.underlying_code == code,
                Candle.data_role == "FUTURE",
                Candle.timeframe == "5minute",
                Candle.timestamp >= start_stamp,
                Candle.timestamp <= end_stamp,
                Candle.source_verified.is_(True),
            ).order_by(Candle.timestamp)
        ).all()
        instruments = session.scalars(
            select(Instrument).where(
                Instrument.provider == provider,
                Instrument.underlying_code == code,
                Instrument.instrument_kind == "FUTURE",
            )
        ).all()
        spot_frame = _as_utc_frame(spot_rows)
        futures_frame = front_future_frame(future_rows, instruments, include_expiry=True)
        coverage[code] = {"spot_candles": len(spot_frame), "futures_candles": len(futures_frame)}
        if len(spot_frame) >= 100 and len(futures_frame) >= 100:
            frames[code] = (spot_frame, futures_frame)
    return frames, coverage


def inspect_setup_training_data(
    db_url: str,
    indices: list[str],
    fit_start: date,
    fit_end: date,
    tuning_end: date,
    holdout_end: date,
    provider: str = "UPSTOX",
) -> dict:
    """Check option availability first, then inspect rule candidates when labels are possible."""
    if not fit_start <= fit_end < tuning_end < holdout_end:
        raise ValueError("Expected fit_start <= fit_end < tuning_end < holdout_end")
    indices = list(dict.fromkeys(indices))
    if not indices or set(indices) - set(SETUP_INDICES):
        raise ValueError("Select NIFTY, BANK NIFTY or SENSEX")
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {},
    )
    results = []
    periods = {
        "Fit": (fit_start, fit_end),
        "Tune": (fit_end + timedelta(days=1), tuning_end),
        "Holdout": (tuning_end + timedelta(days=1), holdout_end),
    }
    try:
        with Session(engine) as session:
            start_stamp = datetime.combine(fit_start, time.min, tzinfo=UTC)
            end_stamp = datetime.combine(holdout_end, time.max, tzinfo=UTC)
            stored_counts = {
                (code, role): count
                for code, role, count in session.execute(
                    select(Candle.underlying_code, Candle.data_role, func.count(Candle.id))
                    .where(
                        Candle.provider == provider,
                        Candle.underlying_code.in_(indices),
                        Candle.data_role.in_(("SPOT", "FUTURE", "OPTION")),
                        Candle.timeframe == "5minute",
                        Candle.timestamp >= start_stamp,
                        Candle.timestamp <= end_stamp,
                        Candle.source_verified.is_(True),
                    )
                    .group_by(Candle.underlying_code, Candle.data_role)
                )
            }
            option_indices = [code for code in indices if stored_counts.get((code, "OPTION"), 0)]
            frames, candle_counts = (
                _load_setup_frames(session, option_indices, fit_start, holdout_end, provider)
                if option_indices else ({}, {})
            )
            for code in indices:
                stored_spot = stored_counts.get((code, "SPOT"), 0)
                stored_futures = stored_counts.get((code, "FUTURE"), 0)
                front_counts = candle_counts.get(code)
                summary = {
                    "index": code,
                    "spot_candles": front_counts["spot_candles"] if front_counts else stored_spot,
                    "futures_candles": front_counts["futures_candles"] if front_counts else stored_futures,
                    "futures_count_label": "front-contract futures candles" if front_counts else "verified futures candles",
                    "periods": [], "missing_examples": [], "rejections": {},
                }
                if code not in option_indices:
                    summary["issue"] = (
                        "Fewer than 100 verified spot or futures candles"
                        if summary["spot_candles"] < 100 or summary["futures_candles"] < 100
                        else (
                            "No verified five-minute option candles in these dates. "
                            "Rule setups were not counted because option outcomes cannot be checked."
                        )
                    )
                    summary["periods"] = [
                        {
                            "name": name, "start": start, "end": end,
                            "candidates": None, "labelled": 0,
                            "coverage_pct": None, "ready": False,
                        }
                        for name, (start, end) in periods.items()
                    ]
                    results.append(summary)
                    continue
                if code not in frames:
                    summary["issue"] = "Fewer than 100 verified spot or futures candles"
                    results.append(summary)
                    continue
                spot_frame, futures_frame = frames[code]
                reference = frames.get(relative_index_code(code))
                dataset = setup_candidates(
                    spot_frame, futures_frame, code,
                    relative_frame=reference[0] if reference else None,
                )
                if dataset.empty:
                    summary["issue"] = "No completed rule candidates in this period"
                    results.append(summary)
                    continue
                labelled, details = option_price_labels(
                    session, dataset, provider=provider, cost_model=provider
                )
                summary["rejections"] = details["rejections"]
                summary["missing_examples"] = details["missing_examples"]
                candidate_days = pd.to_datetime(dataset.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
                labelled_days = (
                    pd.to_datetime(labelled.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
                    if not labelled.empty else pd.Series([], dtype=object)
                )
                for name, (start, end) in periods.items():
                    candidate_count = int(((candidate_days >= start) & (candidate_days <= end)).sum())
                    labelled_count = int(((labelled_days >= start) & (labelled_days <= end)).sum()) if len(labelled_days) else 0
                    required = {"Fit": 100, "Tune": 20, "Holdout": 20}[name]
                    summary["periods"].append({
                        "name": name, "start": start, "end": end,
                        "candidates": candidate_count, "labelled": labelled_count,
                        "coverage_pct": round(100 * labelled_count / candidate_count, 1) if candidate_count else 0.0,
                        "ready": candidate_count >= required and labelled_count / candidate_count >= 0.90 if candidate_count else False,
                    })
                results.append(summary)
    finally:
        engine.dispose()
    return {"provider": provider, "indices": results, "periods": periods}


def train_setup_quality_models(
    db_url: str,
    manifest_path: Path,
    indices: list[str],
    fit_start: date,
    fit_end: date,
    tuning_end: date,
    holdout_end: date,
    provider: str = "UPSTOX",
    contract: SetupContract | None = None,
) -> dict:
    if not fit_start <= fit_end < tuning_end < holdout_end:
        raise ValueError("Expected fit_start <= fit_end < tuning_end < holdout_end")
    indices = list(dict.fromkeys(indices))
    unsupported = set(indices) - set(SETUP_INDICES)
    if unsupported or not indices:
        raise ValueError("Setup-quality training supports NIFTY, BANK NIFTY and SENSEX only")
    contract = contract or SetupContract()
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {},
    )
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    coverage: dict[str, dict[str, object]] = {}
    try:
        with Session(engine) as session:
            frames, coverage = _load_setup_frames(session, indices, fit_start, holdout_end, provider)
    finally:
        engine.dispose()
    missing = [code for code in indices if code not in frames]
    if missing:
        detail = "; ".join(
            f"{code}: {coverage[code]['spot_candles']} spot, {coverage[code]['futures_candles']} futures"
            for code in missing
        )
        raise RuntimeError("Insufficient aligned spot/futures data. " + detail)

    datasets: dict[str, pd.DataFrame] = {}
    for code in indices:
        spot_frame, futures_frame = frames[code]
        relative_code = relative_index_code(code)
        relative_frame = frames.get(relative_code, (None, None))[0]
        dataset = setup_candidates(
            spot_frame,
            futures_frame,
            code,
            relative_frame=relative_frame,
            contract=contract,
        )
        if dataset.empty:
            raise RuntimeError(f"{code}: no rule-confirmed candidates could be labelled")
        datasets[code] = dataset.sort_values("timestamp").reset_index(drop=True)
        coverage[code]["candidates"] = len(dataset)

    label_engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {},
    )
    label_sources: dict[str, str] = {}
    try:
        with Session(label_engine) as session:
            for code, proxy_dataset in list(datasets.items()):
                option_dataset, option_coverage = option_price_labels(
                    session,
                    proxy_dataset,
                    provider=provider,
                    cost_model=provider,
                )
                coverage[code]["option_labelled_candidates"] = option_coverage["option_labelled"]
                coverage[code]["option_label_coverage_pct"] = round(
                    float(option_coverage["coverage"]) * 100, 2
                )
                coverage[code]["option_label_rejections"] = option_coverage["rejections"]
                proxy_dates = pd.to_datetime(proxy_dataset.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
                option_dates = (
                    pd.to_datetime(option_dataset.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
                    if not option_dataset.empty
                    else pd.Series([], dtype=object)
                )
                periods = {
                    "fit": (fit_start, fit_end),
                    "tuning": (fit_end + timedelta(days=1), tuning_end),
                    "holdout": (tuning_end + timedelta(days=1), holdout_end),
                }
                period_coverage = {}
                for name, (period_start, period_end) in periods.items():
                    proxy_count = int(((proxy_dates >= period_start) & (proxy_dates <= period_end)).sum())
                    option_count = int(((option_dates >= period_start) & (option_dates <= period_end)).sum()) if len(option_dates) else 0
                    period_coverage[name] = round(option_count / proxy_count, 4) if proxy_count else 0.0
                coverage[code]["option_label_period_coverage"] = period_coverage
                if period_coverage and min(period_coverage.values()) >= 0.90:
                    datasets[code] = option_dataset.sort_values("timestamp").reset_index(drop=True)
                    label_sources[code] = "VERIFIED_OPTION_CANDLES"
                else:
                    label_sources[code] = "UNDERLYING_TRIPLE_BARRIER_PROXY"
    finally:
        label_engine.dispose()

    import xgboost as xgb

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(UTC)
    index_models: dict[str, dict] = {}
    all_issues: list[str] = []
    total_samples = 0
    for code, dataset in datasets.items():
        local_dates = pd.to_datetime(dataset.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.date
        fit = dataset[local_dates <= fit_end]
        tuning = dataset[(local_dates > fit_end) & (local_dates <= tuning_end)]
        holdout = dataset[(local_dates > tuning_end) & (local_dates <= holdout_end)]
        if len(fit) < 100 or len(tuning) < 20 or len(holdout) < 20:
            raise RuntimeError(
                f"{code}: setup candidates are insufficient for chronological validation "
                f"({len(fit)} fit, {len(tuning)} tuning, {len(holdout)} holdout)"
            )
        if fit.label.nunique() < 2:
            raise RuntimeError(f"{code}: fit period contains only one TAKE/SKIP class")
        total_samples += len(dataset)
        train_y = (fit.label == "TAKE").to_numpy(dtype=int)
        tuning_y = (tuning.label == "TAKE").to_numpy(dtype=int)
        counts = Counter(train_y.tolist())
        weights = np.asarray(
            [len(train_y) / (len(counts) * counts[value]) for value in train_y], dtype=float
        )
        train_matrix = xgb.DMatrix(
            fit[SETUP_FEATURES].to_numpy(dtype=float),
            label=train_y,
            weight=weights,
            feature_names=SETUP_FEATURES,
        )
        tuning_matrix = xgb.DMatrix(
            tuning[SETUP_FEATURES].to_numpy(dtype=float),
            label=tuning_y,
            feature_names=SETUP_FEATURES,
        )
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.04,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "lambda": 1.5,
            "seed": 42,
            "nthread": 1,
        }
        booster = xgb.train(
            params,
            train_matrix,
            num_boost_round=800,
            evals=[(tuning_matrix, "tuning")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        rounds = int(booster.best_iteration) + 1
        tuning_raw = booster.predict(tuning_matrix, iteration_range=(0, rounds))
        temperature = _binary_temperature(tuning_raw, tuning_y)
        tuning_probabilities = _apply_binary_temperature(tuning_raw, temperature)
        threshold = select_probability_threshold(tuning, tuning_probabilities)
        holdout_matrix = xgb.DMatrix(
            holdout[SETUP_FEATURES].to_numpy(dtype=float), feature_names=SETUP_FEATURES
        )
        holdout_probabilities = _apply_binary_temperature(
            booster.predict(holdout_matrix, iteration_range=(0, rounds)), temperature
        )
        holdout_metrics = selection_metrics(holdout, holdout_probabilities, threshold)
        quality_issues = setup_quality_issues(holdout_metrics)
        pre_holdout = pd.concat([fit, tuning], ignore_index=True)
        walk_forward = walk_forward_validation(
            pre_holdout,
            model_params=params,
            boosting_rounds=rounds,
        )
        stable_folds = sum(
            1
            for fold in walk_forward
            if int(fold.get("accepted") or 0) >= 20
            and fold.get("profit_factor") is not None
            and float(fold["profit_factor"]) >= 1.20
            and float(fold.get("expectancy") or 0.0) > 0
        )
        if stable_folds < 2:
            quality_issues.append("fewer than two walk-forward folds passed stability gates")
        all_issues.extend(f"{code}: {issue}" for issue in quality_issues)

        full = pre_holdout
        full_y = (full.label == "TAKE").to_numpy(dtype=int)
        full_counts = Counter(full_y.tolist())
        full_weights = np.asarray(
            [len(full_y) / (len(full_counts) * full_counts[value]) for value in full_y], dtype=float
        )
        final_model = xgb.train(
            params,
            xgb.DMatrix(
                full[SETUP_FEATURES].to_numpy(dtype=float),
                label=full_y,
                weight=full_weights,
                feature_names=SETUP_FEATURES,
            ),
            num_boost_round=rounds,
            verbose_eval=False,
        )
        model_file = f"{manifest_path.stem}-{code.lower()}.json"
        temporary_model = manifest_path.parent / f".{manifest_path.stem}-{code.lower()}.tmp.json"
        final_model.save_model(temporary_model)
        os.replace(temporary_model, manifest_path.parent / model_file)
        importance = final_model.get_score(importance_type="gain")
        index_models[code] = {
            "file": model_file,
            "threshold": round(float(threshold), 4),
            "probability_temperature": round(float(temperature), 4),
            "boosting_rounds": rounds,
            "samples": len(dataset),
            "fit_samples": len(fit),
            "tuning_samples": len(tuning),
            "holdout_samples": len(holdout),
            "class_counts": dict(Counter(dataset.label.tolist())),
            "holdout": holdout_metrics,
            "walk_forward_validation": walk_forward,
            "stable_walk_forward_folds": stable_folds,
            "quality_issues": quality_issues,
            "feature_importance": {
                feature: round(float(importance.get(feature, 0.0)), 6)
                for feature in SETUP_FEATURES
            },
        }

    version = f"setup-quality-{trained_at.strftime('%Y%m%dT%H%M%SZ')}"
    overall_label_source = (
        "VERIFIED_OPTION_CANDLES"
        if set(label_sources.values()) == {"VERIFIED_OPTION_CANDLES"}
        else "UNDERLYING_TRIPLE_BARRIER_PROXY"
    )
    activation_issues = [*all_issues]
    if overall_label_source != "VERIFIED_OPTION_CANDLES":
        activation_issues.insert(
            0,
            "research-only: replace underlying proxy labels with verified option-price labels before activation",
        )
    metadata = {
        "version": version,
        "model_type": "SETUP_QUALITY",
        "trained_at": trained_at.isoformat(),
        "training_start": fit_start.isoformat(),
        "training_end": fit_end.isoformat(),
        "tuning_start": (fit_end + timedelta(days=1)).isoformat(),
        "tuning_end": tuning_end.isoformat(),
        "validation_start": (tuning_end + timedelta(days=1)).isoformat(),
        "validation_end": holdout_end.isoformat(),
        "indices": indices,
        "provider": provider,
        "training_data_provider": provider,
        "feature_schema": SETUP_FEATURE_SCHEMA,
        "features": SETUP_FEATURES,
        "samples": total_samples,
        "classes": ["SKIP", "TAKE"],
        "label_source": overall_label_source,
        "label_sources_by_index": label_sources,
        "training_contract": asdict(contract),
        "coverage": coverage,
        "index_models": index_models,
        "research_quality": {
            "ready_for_option_replay": not all_issues,
            "option_label_ready": overall_label_source == "VERIFIED_OPTION_CANDLES",
            "issues": all_issues,
        },
        "activation_quality": {
            "ready": overall_label_source == "VERIFIED_OPTION_CANDLES" and not activation_issues,
            "issues": activation_issues,
        },
    }
    manifest = {
        "version": version,
        "model_type": "SETUP_QUALITY",
        "feature_schema": SETUP_FEATURE_SCHEMA,
        "features": SETUP_FEATURES,
        "training_end": fit_end.isoformat(),
        "tuning_end": tuning_end.isoformat(),
        "validation_end": holdout_end.isoformat(),
        "label_source": overall_label_source,
        "index_models": {
            code: {
                "file": detail["file"],
                "threshold": detail["threshold"],
                "probability_temperature": detail["probability_temperature"],
            }
            for code, detail in index_models.items()
        },
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_metadata = model_metadata_path(manifest_path).with_name(
        f".{model_metadata_path(manifest_path).name}.tmp"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(temporary_manifest, manifest_path)
    os.replace(temporary_metadata, model_metadata_path(manifest_path))
    return metadata
