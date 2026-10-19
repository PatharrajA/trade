from datetime import UTC, datetime, timedelta

import pandas as pd

from app.strategy import PRICE_FEATURES, indicators
from app.training import labelled_samples


def test_previous_day_levels_do_not_use_current_day_data():
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    rows = []
    for day in range(2):
        for bar in range(6):
            price = 100 + day * 20 + bar
            rows.append({"timestamp": start + timedelta(days=day, minutes=bar * 5), "open": price, "high": price + 2, "low": price - 2, "close": price + 1, "volume": 100})
    frame = indicators(pd.DataFrame(rows))
    first_second_day = frame.iloc[6]
    assert first_second_day.previous_day_high == 107
    assert first_second_day.previous_day_low == 98
    assert first_second_day.previous_day_close == 106


def test_training_labels_never_cross_market_days():
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    rows = []
    for day in range(3):
        for bar in range(20):
            price = 100 + day * 10 + bar
            rows.append({"timestamp": start + timedelta(days=day, minutes=bar * 5), "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000})
    samples = labelled_samples(pd.DataFrame(rows), horizon=6)
    local = pd.to_datetime(samples.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    counts = samples.assign(day=local.dt.date).groupby("day").size()
    assert all(counts <= 14)


def test_zero_volume_index_candles_use_futures_liquidity_features():
    start = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
    spot_rows = []
    future_rows = []
    for day in range(4):
        for bar in range(40):
            stamp = start + timedelta(days=day, minutes=bar * 5)
            spot_price = 24_000 + day * 25 + bar * 2
            future_price = spot_price + 18 + (bar % 5)
            spot_rows.append(
                {
                    "timestamp": stamp,
                    "open": spot_price,
                    "high": spot_price + 4,
                    "low": spot_price - 4,
                    "close": spot_price + 1,
                    "volume": 0,
                }
            )
            future_rows.append(
                {
                    "timestamp": stamp,
                    "open": future_price,
                    "high": future_price + 5,
                    "low": future_price - 5,
                    "close": future_price + 1,
                    "volume": 10_000 + bar * 125,
                }
            )

    samples = labelled_samples(
        pd.DataFrame(spot_rows),
        horizon=6,
        futures_frame=pd.DataFrame(future_rows),
    )

    assert not samples.empty
    assert samples[PRICE_FEATURES].notna().all().all()
    assert (samples["volume_ratio"] > 0).all()
