from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from app.models import Candle
from app.runtime import RuntimeService


def _row(stamp: datetime, close: str):
    value = Decimal(close)
    return (
        stamp,
        value - Decimal("1"),
        value + Decimal("2"),
        value - Decimal("2"),
        value,
        1_000,
        500,
    )


def test_runtime_candle_upsert_normalizes_aware_and_naive_utc(session):
    runtime = object.__new__(RuntimeService)
    aware = datetime(2026, 10, 15, 4, 0, tzinfo=UTC)
    naive = aware.replace(tzinfo=None)

    runtime._store_runtime_rows(
        session,
        provider="UPSTOX",
        code="NIFTY",
        instrument_key="NSE_INDEX|Nifty 50",
        rows=[_row(aware, "25000")],
        data_role="SPOT",
    )
    session.commit()
    runtime._store_runtime_rows(
        session,
        provider="UPSTOX",
        code="NIFTY",
        instrument_key="NSE_INDEX|Nifty 50",
        rows=[_row(naive, "25005")],
        data_role="SPOT",
    )
    session.commit()

    rows = session.scalars(select(Candle)).all()
    assert len(rows) == 1
    assert rows[0].timestamp == naive
    assert rows[0].close == Decimal("25005")


def test_runtime_parse_rejects_malformed_market_candles():
    valid = ["2026-10-15T09:30:00", "100", "102", "99", "101", 1000, 0]
    assert RuntimeService._parse(valid) is not None
    for broken in (
        [*valid[:2], "--", *valid[3:]],
        [*valid[:4], "NaN", *valid[5:]],
        [valid[0], "100", "99", "98", "101", 1000, 0],
        valid[:3],
    ):
        assert RuntimeService._parse(broken) is None
