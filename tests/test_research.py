from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Candle
from app.research import _parse, _store_candles


def test_parse_groww_candle_accepts_null_open_interest():
    candle = _parse(["2025-09-24T10:30:00", 245.95, 246.15, 245.05, 245.6, 735060, None])

    assert candle == (
        datetime(2025, 9, 24, 5, 0, tzinfo=UTC),
        Decimal("245.95"),
        Decimal("246.15"),
        Decimal("245.05"),
        Decimal("245.6"),
        735060,
        0,
    )


def test_parse_groww_candle_rejects_non_numeric_price():
    assert _parse(["2025-09-24T10:30:00", "--", 246.15, 245.05, 245.6, 735060, None]) is None


def test_parse_groww_candle_rejects_non_finite_or_invalid_ohlc():
    assert _parse(["2025-09-24T10:30:00", "NaN", 246.15, 245.05, 245.6, 735060, 0]) is None
    assert _parse(["2025-09-24T10:30:00", 245.95, 244, 245.05, 245.6, 735060, 0]) is None


@pytest.mark.asyncio
async def test_research_redownload_deduplicates_rows_within_response(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'candles.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    class DuplicateClient:
        async def historical_candles(self, instrument_key, start, end, minutes):
            candle = ["2025-11-21T09:00:00", 100, 102, 99, 101, 0, 0]
            return [candle, candle]

    kwargs = dict(
        client=DuplicateClient(), session_factory=factory,
        instrument_key="NSE-NIFTY", underlying_code="NIFTY",
        start=datetime(2025, 11, 21).date(), end=datetime(2025, 11, 21).date(),
        provider="GROWW", minutes=1,
    )
    assert await _store_candles(**kwargs) == 1
    assert await _store_candles(**kwargs) == 0
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Candle)) == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_repair_only_requests_detected_intraday_gap(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'repair.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    trading_day = date(2025, 11, 21)
    with factory() as session:
        session.add_all([
            Candle(provider="GROWW", instrument_key="NSE-NIFTY", underlying_code="NIFTY", timeframe="5minute", timestamp=datetime(2025, 11, 21, 3, 45, tzinfo=UTC), open=100, high=102, low=99, close=101, volume=0, open_interest=0, source_verified=True, data_role="SPOT"),
            Candle(provider="GROWW", instrument_key="NSE-NIFTY", underlying_code="NIFTY", timeframe="5minute", timestamp=datetime(2025, 11, 21, 3, 55, tzinfo=UTC), open=101, high=103, low=100, close=102, volume=0, open_interest=0, source_verified=True, data_role="SPOT"),
        ])
        session.commit()

    class GapClient:
        def __init__(self):
            self.calls = []

        async def historical_candles(self, instrument_key, start, end, minutes):
            self.calls.append((start, end, minutes))
            return [["2025-11-21T09:20:00", 100, 102, 99, 101, 0, 0]]

    client = GapClient()
    inserted = await _store_candles(
        client=client,
        session_factory=factory,
        instrument_key="NSE-NIFTY",
        underlying_code="NIFTY",
        start=trading_day,
        end=trading_day,
        provider="GROWW",
        repair_only=True,
    )

    assert inserted == 1
    assert client.calls == [(trading_day, trading_day, 5)]
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Candle)) == 3
    engine.dispose()
