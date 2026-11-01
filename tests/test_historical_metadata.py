from datetime import date
from decimal import Decimal

from sqlalchemy import select

from app.historical_metadata import _lot_rule, verify_historical_contract_metadata
from app.models import Instrument


def _option(
    session,
    *,
    key: str,
    code: str,
    expiry: date,
    valid_from: date,
    lot_size: int = 0,
    verified: bool = False,
) -> None:
    session.add(
        Instrument(
            provider="GROWW",
            instrument_key=key,
            underlying_code=code,
            exchange="BSE" if code == "SENSEX" else "NSE",
            trading_symbol=key,
            option_type="CE",
            strike=Decimal("22500"),
            expiry=expiry,
            lot_size=lot_size,
            tick_size=Decimal("0.05"),
            valid_from=valid_from,
            valid_to=expiry,
            active=False,
            instrument_kind="OPTION",
            metadata_verified=verified,
            metadata_source="CSV_IMPORT" if verified else "GROWW_HISTORICAL_SYMBOL",
        )
    )


def test_repair_splits_nifty_lot_change_and_is_idempotent(session):
    _option(session, key="NIFTY-2023", code="NIFTY",
            expiry=date(2023, 10, 5), valid_from=date(2023, 9, 28))
    _option(session, key="NIFTY-MAY2", code="NIFTY",
            expiry=date(2024, 5, 2), valid_from=date(2024, 4, 25))
    _option(session, key="NIFTY-MAY9", code="NIFTY",
            expiry=date(2024, 5, 9), valid_from=date(2024, 5, 2),
            lot_size=50, verified=True)
    session.flush()

    result = verify_historical_contract_metadata(session)
    assert result.split_contracts == 1
    records = session.scalars(
        select(Instrument).order_by(Instrument.instrument_key, Instrument.valid_from)
    ).all()
    by_key: dict[str, list[Instrument]] = {}
    for record in records:
        by_key.setdefault(record.instrument_key, []).append(record)
    assert by_key["NIFTY-2023"][0].lot_size == 50
    assert [(row.valid_from, row.valid_to, row.lot_size) for row in by_key["NIFTY-MAY2"]] == [
        (date(2024, 4, 25), date(2024, 4, 25), 50),
        (date(2024, 4, 26), date(2024, 5, 2), 25),
    ]
    assert by_key["NIFTY-MAY9"][0].lot_size == 25
    assert all(row.metadata_verified for row in by_key["NIFTY-MAY2"])

    again = verify_historical_contract_metadata(session)
    assert again.corrected_versions == 0
    assert again.split_contracts == 0


def test_repair_covers_bank_nifty_and_sensex_transition_dates(session):
    _option(session, key="BANK-JUNE23", code="BANK_NIFTY",
            expiry=date(2023, 6, 28), valid_from=date(2023, 6, 22))
    _option(session, key="BANK-JULY23", code="BANK_NIFTY",
            expiry=date(2023, 7, 27), valid_from=date(2023, 4, 27))
    _option(session, key="BANK-SEP25", code="BANK_NIFTY",
            expiry=date(2025, 9, 25), valid_from=date(2025, 4, 24))
    _option(session, key="BANK-JAN26", code="BANK_NIFTY",
            expiry=date(2026, 1, 27), valid_from=date(2025, 12, 24))
    _option(session, key="SENSEX-2023", code="SENSEX",
            expiry=date(2023, 11, 10), valid_from=date(2023, 11, 3))
    _option(session, key="SENSEX-MAR25", code="SENSEX",
            expiry=date(2025, 3, 28), valid_from=date(2024, 12, 27))
    _option(session, key="SENSEX-JAN27", code="SENSEX",
            expiry=date(2027, 1, 28), valid_from=date(2026, 10, 29))
    session.flush()

    result = verify_historical_contract_metadata(session)
    assert result.unresolved_versions == 0
    rows = session.scalars(select(Instrument).order_by(Instrument.valid_from)).all()
    by_key: dict[str, list[Instrument]] = {}
    for row in rows:
        by_key.setdefault(row.instrument_key, []).append(row)
    assert by_key["BANK-JUNE23"][0].lot_size == 25
    assert [row.lot_size for row in by_key["BANK-JULY23"]] == [25, 15]
    assert [row.lot_size for row in by_key["BANK-SEP25"]] == [30, 35]
    assert by_key["BANK-JAN26"][0].lot_size == 30
    assert by_key["SENSEX-2023"][0].lot_size == 10
    assert [row.lot_size for row in by_key["SENSEX-MAR25"]] == [10, 20]
    assert [row.lot_size for row in by_key["SENSEX-JAN27"]] == [20, 25]
    assert all(row.tick_size == Decimal("0.05") and row.metadata_verified for row in rows)


def test_known_first_2025_expiries_and_ambiguous_long_dated_series():
    assert _lot_rule("NIFTY", date(2025, 1, 2), date(2024, 12, 27))[0] == 75
    assert _lot_rule("NIFTY", date(2025, 1, 30), date(2025, 1, 23))[0] == 25
    assert _lot_rule("BANK_NIFTY", date(2025, 1, 29), date(2025, 1, 23))[0] == 15
    assert _lot_rule("BANK_NIFTY", date(2025, 2, 26), date(2025, 2, 20))[0] == 30
    assert _lot_rule("SENSEX", date(2025, 1, 3), date(2024, 12, 27))[0] == 10
    assert _lot_rule("SENSEX", date(2025, 1, 7), date(2025, 1, 3))[0] == 20
    assert _lot_rule("SENSEX", date(2025, 1, 28), date(2025, 1, 24))[0] == 10
    assert _lot_rule("NIFTY", date(2026, 3, 31), date(2025, 11, 1)) is None
    assert _lot_rule("BANK_NIFTY", date(2023, 9, 28), date(2023, 6, 29)) is None
    assert _lot_rule("SENSEX", date(2023, 6, 30), date(2023, 5, 12)) is None
