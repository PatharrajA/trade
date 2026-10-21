from datetime import UTC, datetime
from decimal import Decimal

from app.research import _parse


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
