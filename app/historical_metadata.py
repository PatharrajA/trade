from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Instrument


REPAIR_START = date(2023, 1, 1)
REPAIR_END = date(2026, 10, 31)
OPTION_TICK = Decimal("0.05")

# Ambiguous old long-dated series remain unverified.
CHANGE_DATES = (
    date(2023, 4, 28), date(2023, 7, 1), date(2024, 4, 26),
    date(2024, 12, 25), date(2024, 12, 27), date(2024, 12, 28),
    date(2025, 4, 25), date(2025, 12, 31), date(2026, 10, 30),
)


@dataclass(frozen=True)
class MetadataRepair:
    corrected_versions: int
    split_contracts: int
    unresolved_versions: int


def _lot_rule(code: str, expiry: date, day: date) -> tuple[int, str] | None:
    if code == "NIFTY":
        if expiry.year <= 2023:
            return 50, "NSE_FAOP_56233"
        if expiry.year == 2024:
            return (50, "NSE_FAOP_56233") if day < date(2024, 4, 26) else (25, "NSE_FAOP_61415")
        if expiry.year == 2025:
            if expiry == date(2025, 1, 30):
                return 25, "NSE_FAOP_64672"
            if expiry >= date(2025, 3, 1) and day < date(2024, 12, 27):
                return 25, "NSE_FAOP_64672"
            return 75, "NSE_FAOP_64672"
        if day < date(2025, 12, 31) and expiry.month in {3, 6, 9, 12}:
            return None
        return 65, "NSE_FAOP_70616"

    if code == "BANK_NIFTY":
        if expiry < date(2023, 7, 1):
            return 25, "NSE_FAOP_56233"
        if expiry.year == 2023 and expiry.month == 7:
            if expiry == date(2023, 7, 27) and day >= date(2023, 4, 28):
                return 15, "NSE_FAOP_56233"
            return 25, "NSE_FAOP_56233"
        # Existing long-dated contracts kept the old lot until June 2023 EOD;
        # the expiry alone cannot identify whether an earlier version was new.
        if expiry.year == 2023 and expiry.month >= 9 and day < date(2023, 7, 1):
            return None
        if expiry.year <= 2024:
            return 15, "NSE_FAOP_56233"
        if expiry.year == 2025:
            if expiry.month == 1:
                return 15, "NSE_FAOP_64672"
            if expiry.month <= 6:
                if expiry.month in {3, 6} and day < date(2024, 12, 25):
                    return 15, "NSE_FAOP_64672"
                return 30, "NSE_FAOP_64672"
            if expiry.month in {9, 12} and day < date(2025, 4, 25):
                return 30, "NSE_FAOP_67372"
            return 35, "NSE_FAOP_67372"
        if day < date(2025, 12, 31) and expiry.month in {3, 6, 9, 12}:
            return None
        return 30, "NSE_FAOP_70616"

    if code == "SENSEX":
        if expiry < date(2023, 5, 15):
            return 15, "BSE_20230327_65"
        if day < date(2023, 5, 15):
            return None
        if expiry.year <= 2024:
            return 10, "BSE_20230327_65"
        if expiry.year == 2025:
            if expiry in {date(2025, 1, 3), date(2025, 1, 28)}:
                return 10, "BSE_20241021_13"
            if expiry >= date(2025, 3, 1) and day < date(2024, 12, 28):
                return 10, "BSE_20241021_13"
            return 20, "BSE_20241021_13"
        if expiry.year == 2026 or day < date(2026, 10, 30):
            return 20, "BSE_20241021_13"
        return 25, "BSE_20260930_58"
    return None


def _verified_value(instrument: Instrument, lot: int, source: str) -> bool:
    changed = (
        instrument.lot_size != lot
        or instrument.tick_size != OPTION_TICK
        or not instrument.metadata_verified
    )
    if changed:
        instrument.lot_size = lot
        instrument.tick_size = OPTION_TICK
        instrument.metadata_verified = True
        instrument.metadata_source = source
    return changed


def verify_historical_contract_metadata(session: Session) -> MetadataRepair:
    """Repair supported GROWW index-option metadata using dated exchange notices."""
    instruments = session.scalars(
        select(Instrument)
        .where(
            Instrument.provider == "GROWW",
            Instrument.underlying_code.in_(("NIFTY", "BANK_NIFTY", "SENSEX")),
            Instrument.instrument_kind == "OPTION",
            Instrument.valid_from <= REPAIR_END,
            (Instrument.valid_to.is_(None) | (Instrument.valid_to >= REPAIR_START)),
            Instrument.expiry > REPAIR_START,
        )
        .order_by(Instrument.instrument_key, Instrument.valid_from)
    ).all()
    versions = {(row.instrument_key, row.valid_from): row for row in instruments}
    corrected = split = unresolved = 0
    for item in instruments:
        if item.exchange != ("BSE" if item.underlying_code == "SENSEX" else "NSE"):
            unresolved += int(not item.metadata_verified)
            continue
        valid_end = min(item.valid_to or item.expiry, item.expiry)
        if item.valid_from > valid_end:
            unresolved += int(not item.metadata_verified)
            continue
        starts = [item.valid_from] + [
            boundary for boundary in CHANGE_DATES
            if item.valid_from < boundary <= min(valid_end, REPAIR_END)
        ]
        periods: list[tuple[date, date, int, str]] = []
        for index, period_start in enumerate(starts):
            period_end = (starts[index + 1] - timedelta(days=1)) if index + 1 < len(starts) else valid_end
            rule = _lot_rule(item.underlying_code, item.expiry, period_start)
            if rule is None:
                periods = []
                break
            lot, source = rule
            if periods and periods[-1][2] == lot:
                previous = periods[-1]
                periods[-1] = (previous[0], period_end, lot, previous[3])
            else:
                periods.append((period_start, period_end, lot, source))
        if not periods:
            unresolved += int(not item.metadata_verified)
            continue
        _, first_end, first_lot, first_source = periods[0]
        if len(periods) > 1 and item.valid_to != first_end:
            item.valid_to = first_end
            corrected += 1
        corrected += int(_verified_value(item, first_lot, first_source))
        for period_start, period_end, lot, source in periods[1:]:
            key = (item.instrument_key, period_start)
            existing = versions.get(key)
            if existing is None:
                existing = Instrument(
                    provider=item.provider,
                    instrument_key=item.instrument_key,
                    underlying_code=item.underlying_code,
                    exchange=item.exchange,
                    trading_symbol=item.trading_symbol,
                    option_type=item.option_type,
                    strike=item.strike,
                    expiry=item.expiry,
                    lot_size=lot,
                    tick_size=OPTION_TICK,
                    valid_from=period_start,
                    valid_to=period_end,
                    active=False,
                    instrument_kind="OPTION",
                    metadata_verified=True,
                    metadata_source=source,
                )
                session.add(existing)
                versions[key] = existing
                corrected += 1
                split += 1
            else:
                if existing.valid_to != period_end:
                    existing.valid_to = period_end
                    corrected += 1
                corrected += int(_verified_value(existing, lot, source))
    session.flush()
    return MetadataRepair(corrected, split, unresolved)
