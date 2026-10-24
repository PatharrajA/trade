from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.data_quality import _missing_intervals, research_readiness
from app.models import Candle, Instrument


def _candle(stamp, role, key, volume=100):
    return Candle(
        provider="GROWW",
        instrument_key=key,
        underlying_code="NIFTY",
        timeframe="5minute",
        timestamp=stamp,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("98"),
        close=Decimal("101"),
        volume=volume,
        open_interest=10,
        source_verified=True,
        data_role=role,
    )


def test_research_readiness_reports_alignment_gaps_and_metadata(session):
    start = datetime(2026, 1, 2, 3, 45, tzinfo=UTC)
    session.add_all(
        [
            _candle(start, "SPOT", "spot", 0),
            _candle(start + timedelta(minutes=5), "SPOT", "spot", 0),
            _candle(start + timedelta(minutes=15), "SPOT", "spot", 0),
            _candle(start, "FUTURE", "future"),
            _candle(start + timedelta(minutes=5), "FUTURE", "future"),
            _candle(start, "OPTION", "option"),
            Instrument(
                provider="GROWW",
                instrument_key="option",
                underlying_code="NIFTY",
                exchange="NSE",
                trading_symbol="NIFTY TEST CE",
                option_type="CE",
                strike=Decimal("25000"),
                expiry=date(2026, 1, 29),
                lot_size=65,
                tick_size=Decimal("0.05"),
                valid_from=date(2026, 1, 1),
                instrument_kind="OPTION",
                metadata_verified=True,
                metadata_source="CSV_IMPORT",
            ),
        ]
    )
    session.commit()

    row = next(item for item in research_readiness(session) if item.provider == "GROWW" and item.underlying_code == "NIFTY")
    assert row.spot_count == 3
    assert row.futures_count == 2
    assert row.aligned_count == 2
    assert row.missing_intervals == 1
    assert row.spot_missing_intervals == 1
    assert row.futures_missing_intervals == 0
    assert row.first_missing_timestamp == start + timedelta(minutes=10)
    assert row.last_missing_timestamp == start + timedelta(minutes=10)
    assert row.zero_volume_percent == 100
    assert row.option_count == 1
    assert row.verified_contracts == 1
    assert row.metadata_gap_contracts == 0


def test_missing_bars_excludes_preopen_and_special_session_breaks():
    # 03:35 UTC is 09:05 IST; the 09:10 pre-open interval is not a market bar.
    assert _missing_intervals([
        datetime(2025, 1, 2, 3, 35, tzinfo=UTC),
        datetime(2025, 1, 2, 3, 45, tzinfo=UTC),
    ]) == 0
    # NSE closed between the two disaster-recovery sessions on these Saturdays.
    for day in (date(2024, 3, 2), date(2024, 5, 18)):
        assert _missing_intervals([
            datetime(day.year, day.month, day.day, 4, 30, tzinfo=UTC),
            datetime(day.year, day.month, day.day, 6, 0, tzinfo=UTC),
        ]) == 0
    assert _missing_intervals([
        datetime(2025, 1, 2, 3, 45, tzinfo=UTC),
        datetime(2025, 1, 2, 3, 55, tzinfo=UTC),
    ]) == 1
