from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import Candle, Instrument
from .strategy import FEATURE_SCHEMA, PRICE_FEATURES, feature_values, indicators, with_futures_features


def model_metadata_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.metadata.json")


def load_model_metadata(model_path: Path) -> dict:
    metadata_path = model_metadata_path(model_path)
    if not model_path.exists() or not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def labelled_samples(
    frame: pd.DataFrame,
    horizon: int = 6,
    futures_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create forward-looking labels while keeping all model features causal."""
    enriched = with_futures_features(frame, futures_frame) if futures_frame is not None else indicators(frame)
    local_day = enriched.timestamp.dt.tz_convert("Asia/Kolkata").dt.date
    future_closes = enriched.close.groupby(local_day).shift(-horizon)
    future_high = pd.concat(
        [enriched.high.groupby(local_day).shift(-step) for step in range(1, horizon + 1)], axis=1
    ).max(axis=1)
    future_low = pd.concat(
        [enriched.low.groupby(local_day).shift(-step) for step in range(1, horizon + 1)], axis=1
    ).min(axis=1)
    move = future_closes - enriched.close
    threshold = enriched.atr * 0.65
    future_range = future_high - future_low
    labels = np.select(
        [
            move >= threshold,
            move <= -threshold,
            (move.abs() < threshold) & (future_range >= enriched.atr * 2.0),
        ],
        ["UPTREND", "DOWNTREND", "HIGH_VOLATILITY"],
        default="RANGE",
    )
    feature_rows = [feature_values(row) for _, row in enriched.iterrows()]
    samples = pd.DataFrame(feature_rows, index=enriched.index)
    samples["label"] = labels
    samples["timestamp"] = enriched.timestamp
    samples["future_close_available"] = future_closes.notna()
    finite = np.isfinite(samples[PRICE_FEATURES].to_numpy(dtype=float)).all(axis=1) & samples["future_close_available"]
    return samples.loc[finite].reset_index(drop=True)


def front_future_frame(candles: list[Candle], instruments: list[Instrument]) -> pd.DataFrame:
    expiries = {item.instrument_key: item.expiry for item in instruments}
    rows = []
    for candle in candles:
        expiry = expiries.get(candle.instrument_key)
        if expiry is None:
            continue
        timestamp = pd.Timestamp(candle.timestamp)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        trading_day = timestamp.tz_convert("Asia/Kolkata").date()
        # Roll at the start of expiry day so training and live inference use the
        # same deterministic contract-selection policy.
        if expiry <= trading_day:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "expiry": expiry,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": candle.volume,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows).sort_values(["timestamp", "expiry"])
    return frame.drop_duplicates("timestamp", keep="first").drop(columns=["expiry"]).reset_index(drop=True)


def train_regime_model(
    db_url: str,
    model_path: Path,
    indices: list[str],
    start_date: date,
    end_date: date,
    provider: str = "UPSTOX",
) -> dict:
    if start_date >= end_date:
        raise ValueError("Training end date must be after the start date")

    engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {})
    samples: list[pd.DataFrame] = []
    diagnostics: dict[str, dict[str, int]] = {}
    try:
        with Session(engine) as session:
            for code in indices:
                candles = session.scalars(
                    select(Candle).where(
                        Candle.provider == provider,
                        Candle.underlying_code == code,
                        Candle.data_role == "SPOT",
                        Candle.timeframe == "5minute",
                        Candle.timestamp >= datetime.combine(start_date, time.min, tzinfo=UTC),
                        Candle.timestamp <= datetime.combine(end_date, time.max, tzinfo=UTC),
                        Candle.source_verified.is_(True),
                    ).order_by(Candle.timestamp)
                ).all()
                if len(candles) < 100:
                    diagnostics[code] = {"spot_candles": len(candles), "futures_candles": 0, "usable_samples": 0}
                    continue
                future_candles = session.scalars(
                    select(Candle).where(
                        Candle.provider == provider,
                        Candle.underlying_code == code,
                        Candle.data_role == "FUTURE",
                        Candle.timeframe == "5minute",
                        Candle.timestamp >= datetime.combine(start_date, time.min, tzinfo=UTC),
                        Candle.timestamp <= datetime.combine(end_date, time.max, tzinfo=UTC),
                        Candle.source_verified.is_(True),
                    ).order_by(Candle.timestamp)
                ).all()
                future_instruments = session.scalars(
                    select(Instrument).where(
                        Instrument.provider == provider,
                        Instrument.underlying_code == code,
                        Instrument.instrument_kind == "FUTURE",
                    )
                ).all()
                futures_frame = front_future_frame(future_candles, future_instruments)
                if len(futures_frame) < 100:
                    diagnostics[code] = {
                        "spot_candles": len(candles),
                        "futures_candles": len(futures_frame),
                        "usable_samples": 0,
                    }
                    continue
                frame = pd.DataFrame([
                    {"timestamp": row.timestamp, "open": float(row.open), "high": float(row.high), "low": float(row.low), "close": float(row.close), "volume": row.volume}
                    for row in candles
                ])
                index_samples = labelled_samples(frame, futures_frame=futures_frame)
                index_samples["underlying_code"] = code
                diagnostics[code] = {
                    "spot_candles": len(candles),
                    "futures_candles": len(futures_frame),
                    "usable_samples": len(index_samples),
                }
                if not index_samples.empty:
                    samples.append(index_samples)
    finally:
        engine.dispose()

    if not samples:
        detail = "; ".join(
            f"{code}: {values['spot_candles']} spot, {values['futures_candles']} futures, {values['usable_samples']} usable"
            for code, values in diagnostics.items()
        )
        raise RuntimeError(
            "No usable spot/futures training samples. Download verified futures candles; "
            "index candles have no tradable volume."
            + (f" Coverage: {detail}" if detail else "")
        )
    dataset = pd.concat(samples, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if len(dataset) < 500:
        raise RuntimeError(f"Only {len(dataset)} usable samples found; at least 500 are required")

    import xgboost as xgb

    classes = sorted(dataset.label.unique().tolist())
    if len(classes) < 3:
        raise RuntimeError("Training data contains fewer than three market regimes; use a wider date range")
    class_to_id = {name: index for index, name in enumerate(classes)}
    labels = dataset.label.map(class_to_id).to_numpy(dtype=int)
    matrix = dataset[PRICE_FEATURES].to_numpy(dtype=float)
    split = max(1, min(len(dataset) - 1, int(len(dataset) * 0.8)))
    train_x, validation_x = matrix[:split], matrix[split:]
    train_y, validation_y = labels[:split], labels[split:]

    counts = Counter(train_y.tolist())
    weights = np.array([len(train_y) / (len(counts) * counts[value]) for value in train_y], dtype=float)
    train_matrix = xgb.DMatrix(train_x, label=train_y, weight=weights, feature_names=PRICE_FEATURES)
    validation_matrix = xgb.DMatrix(validation_x, label=validation_y, feature_names=PRICE_FEATURES)
    booster = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": len(classes),
            "eval_metric": "mlogloss",
            "eta": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "nthread": 1,
        },
        train_matrix,
        num_boost_round=220,
        evals=[(validation_matrix, "validation")],
        verbose_eval=False,
    )
    validation_probabilities = booster.predict(validation_matrix)
    predictions = np.argmax(validation_probabilities, axis=1)
    accuracy = float(np.mean(predictions == validation_y)) if len(validation_y) else 0.0
    per_class_accuracy = []
    for class_id in range(len(classes)):
        mask = validation_y == class_id
        if mask.any():
            per_class_accuracy.append(float(np.mean(predictions[mask] == validation_y[mask])))
    balanced_accuracy = float(np.mean(per_class_accuracy)) if per_class_accuracy else 0.0
    confusion = [[0 for _ in classes] for _ in classes]
    per_class_metrics: dict[str, dict[str, float]] = {}
    for actual, predicted in zip(validation_y.tolist(), predictions.tolist()):
        confusion[int(actual)][int(predicted)] += 1
    for class_id, class_name in enumerate(classes):
        true_positive = confusion[class_id][class_id]
        predicted_total = sum(row[class_id] for row in confusion)
        actual_total = sum(confusion[class_id])
        per_class_metrics[class_name] = {
            "precision": round(true_positive / predicted_total, 4) if predicted_total else 0.0,
            "recall": round(true_positive / actual_total, 4) if actual_total else 0.0,
        }
    importance = booster.get_score(importance_type="gain")
    confidence = np.max(validation_probabilities, axis=1) if len(validation_probabilities) else np.array([])
    walk_forward: list[dict[str, float | int | str]] = []
    for train_fraction in (0.6, 0.7, 0.8):
        fold_end = int(len(dataset) * train_fraction)
        validation_end = min(len(dataset), fold_end + max(1, int(len(dataset) * 0.1)))
        if fold_end < 300 or validation_end - fold_end < 50:
            continue
        fold_y = labels[:fold_end]
        fold_counts = Counter(fold_y.tolist())
        fold_weights = np.array(
            [len(fold_y) / (len(fold_counts) * fold_counts[value]) for value in fold_y],
            dtype=float,
        )
        fold_train = xgb.DMatrix(matrix[:fold_end], label=fold_y, weight=fold_weights, feature_names=PRICE_FEATURES)
        fold_validation = xgb.DMatrix(matrix[fold_end:validation_end], label=labels[fold_end:validation_end], feature_names=PRICE_FEATURES)
        fold_model = xgb.train(
            {
                "objective": "multi:softprob",
                "num_class": len(classes),
                "eval_metric": "mlogloss",
                "eta": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42,
                "nthread": 1,
            },
            fold_train,
            num_boost_round=120,
            verbose_eval=False,
        )
        fold_predictions = np.argmax(fold_model.predict(fold_validation), axis=1)
        expected = labels[fold_end:validation_end]
        fold_scores = []
        for class_id in range(len(classes)):
            mask = expected == class_id
            if mask.any():
                fold_scores.append(float(np.mean(fold_predictions[mask] == expected[mask])))
        walk_forward.append(
            {
                "training_end": pd.Timestamp(dataset.iloc[fold_end - 1].timestamp).isoformat(),
                "validation_end": pd.Timestamp(dataset.iloc[validation_end - 1].timestamp).isoformat(),
                "training_samples": fold_end,
                "validation_samples": validation_end - fold_end,
                "accuracy": round(float(np.mean(fold_predictions == expected)), 4),
                "balanced_accuracy": round(float(np.mean(fold_scores)), 4) if fold_scores else 0.0,
            }
        )

    trained_at = datetime.now(UTC)
    metadata = {
        "version": f"regime-{trained_at.strftime('%Y%m%dT%H%M%SZ')}",
        "trained_at": trained_at.isoformat(),
        "training_start": start_date.isoformat(),
        "training_end": end_date.isoformat(),
        "indices": indices,
        "provider": provider,
        "training_data_provider": provider,
        "feature_schema": FEATURE_SCHEMA,
        "compatible_live_providers": ["UPSTOX", "GROWW"],
        "features": PRICE_FEATURES,
        "classes": classes,
        "samples": len(dataset),
        "training_samples": split,
        "validation_samples": len(dataset) - split,
        "validation_accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "confusion_matrix": confusion,
        "per_class_metrics": per_class_metrics,
        "feature_importance": {feature: round(float(importance.get(feature, 0.0)), 6) for feature in PRICE_FEATURES},
        "validation_confidence": {
            "mean": round(float(np.mean(confidence)), 4) if len(confidence) else 0.0,
            "above_70_percent": round(float(np.mean(confidence >= 0.70) * 100), 2) if len(confidence) else 0.0,
        },
        "walk_forward_validation": walk_forward,
        "class_counts": dict(Counter(dataset.label.tolist())),
        "label_horizon_bars": 6,
        "uses_live_context": False,
        "coverage": diagnostics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_name(f".{model_path.stem}.tmp.json")
    temporary_metadata = model_metadata_path(model_path).with_name(f".{model_metadata_path(model_path).name}.tmp")
    booster.save_model(temporary_model)
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(temporary_model, model_path)
    os.replace(temporary_metadata, model_metadata_path(model_path))
    return metadata
