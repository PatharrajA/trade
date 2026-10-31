from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "vwap_distance", "ema9_20_distance", "ema20_50_distance", "atr_pct", "rsi", "volume_ratio",
    "previous_day_high_distance", "previous_day_low_distance", "previous_day_close_distance",
    "india_vix", "futures_basis_pct", "pcr", "minutes_from_open",
]

# The first locally-trained model deliberately uses only fields that can be
# reconstructed from historical candles. Live-only context can be added to a
# later model by including those fields in its metadata.
PRICE_FEATURES = [
    "vwap_distance", "ema9_20_distance", "ema20_50_distance", "atr_pct", "rsi", "volume_ratio",
    "previous_day_high_distance", "previous_day_low_distance", "previous_day_close_distance",
    "return_1", "return_3", "return_6", "return_12", "ema9_slope", "atr_change",
    "opening_range_high_distance", "opening_range_low_distance", "futures_basis_pct",
    "minutes_from_open",
]

FEATURE_SCHEMA = "spot-futures-v3"
SUPPORTED_FEATURE_SCHEMAS = {"spot-futures-v2", FEATURE_SCHEMA}


@dataclass(frozen=True)
class Decision:
    regime: str
    confidence: float
    direction: str
    reason: str
    features: dict[str, float]


def indicators(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ema9"] = frame.close.ewm(span=9, adjust=False).mean()
    frame["ema20"] = frame.close.ewm(span=20, adjust=False).mean()
    frame["ema50"] = frame.close.ewm(span=50, adjust=False).mean()
    previous = frame.close.shift(1)
    tr = pd.concat(((frame.high - frame.low), (frame.high - previous).abs(), (frame.low - previous).abs()), axis=1).max(axis=1)
    frame["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    delta = frame.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    typical = (frame.high + frame.low + frame.close) / 3
    local = frame.timestamp.dt.tz_convert("Asia/Kolkata")
    day = local.dt.date
    frame["vwap"] = ((typical * frame.volume).groupby(day).cumsum() / frame.volume.groupby(day).cumsum().replace(0, np.nan)).fillna(frame.close)
    frame["volume_ratio"] = frame.volume / frame.volume.rolling(20).mean().replace(0, np.nan)
    opening = local.dt.time <= pd.Timestamp("09:30").time()
    frame["opening_range_high"] = frame.high.where(opening).groupby(day).transform("max")
    frame["opening_range_low"] = frame.low.where(opening).groupby(day).transform("min")
    for bars in (1, 3, 6, 12):
        frame[f"return_{bars}"] = frame.close.groupby(day).pct_change(bars).fillna(0.0)
    frame["ema9_slope"] = frame.ema9.groupby(day).pct_change(3).fillna(0.0)
    frame["atr_change"] = frame.atr.groupby(day).pct_change(6).fillna(0.0)
    frame["opening_range_high_distance"] = (
        frame.close - frame.opening_range_high
    ) / frame.close.replace(0, np.nan)
    frame["opening_range_low_distance"] = (
        frame.close - frame.opening_range_low
    ) / frame.close.replace(0, np.nan)
    daily = frame.assign(trading_day=day).groupby("trading_day").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")).shift(1)
    frame["previous_day_high"] = day.map(daily.day_high)
    frame["previous_day_low"] = day.map(daily.day_low)
    frame["previous_day_close"] = day.map(daily.day_close)
    return frame


def with_futures_features(spot_candles: pd.DataFrame, futures_candles: pd.DataFrame) -> pd.DataFrame:
    """Build causal spot indicators and attach exact-timestamp futures liquidity features."""
    spot = indicators(spot_candles)
    futures = indicators(futures_candles)
    futures = futures[["timestamp", "close", "vwap", "volume_ratio"]].rename(
        columns={
            "close": "futures_close",
            "vwap": "futures_vwap",
            "volume_ratio": "futures_volume_ratio",
        }
    )
    combined = spot.merge(futures, on="timestamp", how="left", validate="one_to_one")
    combined["feature_vwap_distance"] = (
        combined.futures_close - combined.futures_vwap
    ) / combined.futures_close.replace(0, np.nan)
    combined["feature_volume_ratio"] = combined.futures_volume_ratio
    combined["futures_basis_pct"] = (
        (combined.futures_close - combined.close) / combined.close.replace(0, np.nan) * 100
    )
    return combined


class RegimeModel:
    def __init__(self, path: Path) -> None:
        self.model = None
        self.classes = ["UPTREND", "DOWNTREND", "RANGE", "HIGH_VOLATILITY", "UNCERTAIN"]
        self.required_features = FEATURES
        self.metadata: dict = {}
        metadata_path = path.with_name(f"{path.stem}.metadata.json")
        if path.exists() and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            if self.metadata.get("feature_schema") not in SUPPORTED_FEATURE_SCHEMAS:
                return
            import xgboost as xgb

            self.classes = list(self.metadata.get("classes") or self.classes)
            self.required_features = list(self.metadata.get("features") or FEATURES)
            model = xgb.Booster()
            model.load_model(path)
            self.model = model
        self.probability_temperature = max(
            0.05, float(self.metadata.get("probability_temperature", 1.0))
        )

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, values: dict[str, float]) -> tuple[str, float]:
        if not self.model:
            return "UNCERTAIN", 0.0
        matrix = np.array([[values[key] for key in self.required_features]], dtype=float)
        if not np.isfinite(matrix).all():
            return "UNCERTAIN", 0.0
        import xgboost as xgb

        probabilities = self.model.predict(xgb.DMatrix(matrix, feature_names=self.required_features))[0]
        if self.probability_temperature != 1.0:
            powered = np.power(
                np.clip(probabilities, 1e-12, 1.0),
                1.0 / self.probability_temperature,
            )
            probabilities = powered / powered.sum()
        best = int(np.argmax(probabilities))
        if best >= len(self.classes):
            return "UNCERTAIN", 0.0
        return self.classes[best], float(probabilities[best])


def feature_values(row: pd.Series, context: dict[str, float | None] | None = None) -> dict[str, float]:
    context = context or {}
    close = float(row.close)
    stamp = pd.Timestamp(row.timestamp)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    stamp = stamp.tz_convert("Asia/Kolkata")
    return {
        "vwap_distance": float(row.get("feature_vwap_distance", (close - float(row.vwap)) / close)),
        "ema9_20_distance": (float(row.ema9) - float(row.ema20)) / close,
        "ema20_50_distance": (float(row.ema20) - float(row.ema50)) / close,
        "atr_pct": float(row.atr) / close,
        "rsi": float(row.rsi),
        "volume_ratio": float(row.get("feature_volume_ratio", row.volume_ratio)),
        "previous_day_high_distance": (close - float(row.previous_day_high)) / close,
        "previous_day_low_distance": (close - float(row.previous_day_low)) / close,
        "previous_day_close_distance": (close - float(row.previous_day_close)) / close,
        "return_1": float(row.get("return_1", np.nan)),
        "return_3": float(row.get("return_3", np.nan)),
        "return_6": float(row.get("return_6", np.nan)),
        "return_12": float(row.get("return_12", np.nan)),
        "ema9_slope": float(row.get("ema9_slope", np.nan)),
        "atr_change": float(row.get("atr_change", np.nan)),
        "opening_range_high_distance": float(row.get("opening_range_high_distance", np.nan)),
        "opening_range_low_distance": float(row.get("opening_range_low_distance", np.nan)),
        "india_vix": float(context.get("india_vix") or np.nan),
        "futures_basis_pct": float(row.get("futures_basis_pct", context.get("futures_basis_pct") or np.nan)),
        "pcr": float(context.get("pcr") or np.nan),
        "minutes_from_open": float(stamp.hour * 60 + stamp.minute - 555),
    }


def decide(row: pd.Series, context: dict[str, float | None], model: RegimeModel) -> Decision:
    close = float(row.close)
    features = feature_values(row, context)
    regime, confidence = model.predict(features)
    if not model.available:
        return Decision("UNCERTAIN", 0, "NO_TRADE", "Model unavailable", features)
    if not all(np.isfinite(features[key]) for key in model.required_features):
        return Decision("UNCERTAIN", 0, "NO_TRADE", "Required market context unavailable", features)
    futures_close = float(row.get("futures_close", close))
    futures_vwap = float(row.get("futures_vwap", row.vwap))
    if regime == "UPTREND" and confidence >= 0.70 and futures_close > futures_vwap and row.ema9 > row.ema20 and close > row.opening_range_high:
        return Decision(regime, confidence, "CE", "Confirmed opening-range breakout", features)
    if regime == "DOWNTREND" and confidence >= 0.70 and futures_close < futures_vwap and row.ema9 < row.ema20 and close < row.opening_range_low:
        return Decision(regime, confidence, "PE", "Confirmed opening-range breakdown", features)
    return Decision(regime, confidence, "NO_TRADE", "Rules not aligned", features)


def rule_only_decide(row: pd.Series) -> Decision:
    """Deterministic comparison strategy using the same causal confirmation rules."""
    features = feature_values(row)
    close = float(row.close)
    futures_close = float(row.get("futures_close", close))
    futures_vwap = float(row.get("futures_vwap", row.vwap))
    if futures_close > futures_vwap and row.ema9 > row.ema20 and close > row.opening_range_high:
        return Decision("UPTREND", 1.0, "CE", "Rule-only opening-range breakout", features)
    if futures_close < futures_vwap and row.ema9 < row.ema20 and close < row.opening_range_low:
        return Decision("DOWNTREND", 1.0, "PE", "Rule-only opening-range breakdown", features)
    return Decision("RANGE", 0.0, "NO_TRADE", "Rule-only conditions not aligned", features)

