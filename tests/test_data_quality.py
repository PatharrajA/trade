from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.data_quality import research_readiness
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
    assert row.zero_volume_percent == 100
    assert row.option_count == 1
    assert row.verified_contracts == 1
