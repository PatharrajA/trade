from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, Candle
import pandas as pd

from app.training import day_split_indices, train_regime_model


def test_day_split_keeps_indices_together_and_purges_prior_day():
    stamps = pd.Series([
        datetime(2025, 1, day, 4, index * 5, tzinfo=UTC)
        for day in range(1, 21) for index in range(2)
    ])
    train_end, validation_start, validation_end = day_split_indices(stamps, 0.8)
    assert train_end == 30
    assert validation_start == 32
    assert validation_end == len(stamps)


def test_training_reports_missing_futures_before_loading_xgboost(tmp_path):
    database = tmp_path / "training.db"
    engine = create_engine(f"sqlite:///{database}")
    Base.metadata.create_all(engine)
    start = datetime(2026, 1, 2, 3, 45, tzinfo=UTC)
    with Session(engine) as session:
        for offset in range(120):
            price = Decimal("24000") + offset
            session.add(
                Candle(
                    provider="UPSTOX",
                    instrument_key="NSE_INDEX|Nifty 50",
                    underlying_code="NIFTY",
                    timeframe="5minute",
                    timestamp=start + timedelta(minutes=offset * 5),
                    open=price,
                    high=price + 2,
                    low=price - 2,
                    close=price + 1,
                    volume=0,
                    open_interest=0,
                    source_verified=True,
                    data_role="SPOT",
                )
            )
        session.commit()
    engine.dispose()

    with pytest.raises(RuntimeError, match=r"120 spot, 0 futures, 0 usable"):
        train_regime_model(
            f"sqlite:///{database}",
            tmp_path / "regime.json",
            ["NIFTY"],
            date(2026, 1, 1),
            date(2026, 1, 31),
            "UPSTOX",
        )
