from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, Candle
from app.training import train_regime_model


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
