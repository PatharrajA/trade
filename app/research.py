from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .data_quality import _missing_timestamps
from .domain import Contract
from .models import Candle, Instrument
from .providers import BrokerClient, provider_name


@dataclass(frozen=True)
class SyncResult:
    underlying_candles: int
    futures_candles: int
    option_candles: int
    futures_contracts: int
    option_contracts: int
    warnings: list[str]
    incomplete_reasons: list[str]


def _parse(raw: list) -> tuple[datetime, Decimal, Decimal, Decimal, Decimal, int, int] | None:
    try:
        if len(raw) < 6:
            return None
        stamp = datetime.fromisoformat(str(raw[0]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        prices = tuple(Decimal(str(raw[index])) for index in range(1, 5))
        if any(not value.is_finite() or value <= 0 for value in prices):
            return None
        open_, high, low, close = prices
        if high < max(open_, low, close) or low > min(open_, high, close):
            return None
        volume_value = Decimal(str(raw[5] or 0))
        oi_value = Decimal(str(raw[6] or 0)) if len(raw) > 6 else Decimal("0")
        if not volume_value.is_finite() or not oi_value.is_finite():
            return None
        return (
            stamp.astimezone(UTC),
            open_,
            high,
            low,
            close,
            max(int(volume_value), 0),
            max(int(oi_value), 0),
        )
    except (ArithmeticError, IndexError, TypeError, ValueError):
        return None


def _chunks(start: date, end: date, days: int = 28):
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _naive_utc(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).replace(tzinfo=None)


def _has_candles(
    session_factory,
    provider: str,
    underlying_code: str,
    data_role: str,
    start: date,
    end: date,
) -> bool:
    with session_factory() as session:
        return session.scalar(
            select(Candle.id).where(
                Candle.provider == provider,
                Candle.underlying_code == underlying_code,
                Candle.data_role == data_role,
                Candle.timeframe == "5minute",
                Candle.timestamp >= datetime.combine(start, datetime.min.time()),
                Candle.timestamp <= datetime.combine(end, datetime.max.time()),
                Candle.source_verified.is_(True),
            ).limit(1)
        ) is not None


def _known_gap_count(
    session_factory,
    provider: str,
    underlying_code: str,
    data_role: str,
    start: date,
    end: date,
) -> int:
    return len(_known_gap_timestamps(session_factory, provider, underlying_code, data_role, start, end))


def _known_gap_timestamps(
    session_factory,
    provider: str,
    underlying_code: str,
    data_role: str,
    start: date,
    end: date,
) -> list[datetime]:
    with session_factory() as session:
        values = session.scalars(
            select(Candle.timestamp)
            .where(
                Candle.provider == provider,
                Candle.underlying_code == underlying_code,
                Candle.data_role == data_role,
                Candle.timeframe == "5minute",
                Candle.timestamp >= datetime.combine(start, datetime.min.time()),
                Candle.timestamp <= datetime.combine(end, datetime.max.time()),
                Candle.source_verified.is_(True),
            )
            .order_by(Candle.timestamp)
        ).all()
    return _missing_timestamps(list(values))


def _known_gap_days(
    session_factory,
    provider: str,
    underlying_code: str,
    data_role: str,
    start: date,
    end: date,
) -> set[date]:
    return {
        stamp.astimezone(ZoneInfo("Asia/Kolkata")).date()
        for stamp in _known_gap_timestamps(
            session_factory, provider, underlying_code, data_role, start, end
        )
    }


async def _store_candles(
    client: BrokerClient,
    session_factory,
    instrument_key: str,
    underlying_code: str | None,
    start: date,
    end: date,
    *,
    provider: str,
    contract: Contract | None = None,
    data_role: str = "SPOT",
    minutes: int = 5,
    repair_only: bool = False,
    repair_dates: set[date] | None = None,
    warnings: list[str] | None = None,
) -> int:
    inserted = 0
    rejected = 0
    ranges = list(_chunks(start, end))
    if repair_only:
        with session_factory() as session:
            stored = session.scalars(
                select(Candle.timestamp)
                .where(
                    Candle.provider == provider,
                    Candle.instrument_key == instrument_key,
                    Candle.timeframe == f"{minutes}minute",
                    Candle.timestamp >= datetime.combine(start, datetime.min.time()),
                    Candle.timestamp <= datetime.combine(end, datetime.max.time()),
                    Candle.source_verified.is_(True),
                )
                .order_by(Candle.timestamp)
            ).all()
        if stored:
            gap_days = sorted(
                day
                for day in (
                    repair_dates
                    if repair_dates is not None
                    else {
                        stamp.astimezone(ZoneInfo("Asia/Kolkata")).date()
                        for stamp in _missing_timestamps(list(stored), minutes)
                    }
                )
                if start <= day <= end
            )
            ranges = [(day, day) for day in gap_days]
    for chunk_start, chunk_end in ranges:
        rows = (
            await client.historical_option_candles(contract, chunk_start, chunk_end, minutes)
            if contract
            else await client.historical_candles(instrument_key, chunk_start, chunk_end, minutes)
        )
        parsed = []
        for row in rows:
            candle = _parse(row)
            if candle is None:
                rejected += 1
                continue
            parsed.append(candle)
        if not parsed:
            continue
        first_stamp = min(_naive_utc(item[0]) for item in parsed)
        last_stamp = max(_naive_utc(item[0]) for item in parsed)
        with session_factory() as session:
            existing = {
                _naive_utc(value)
                for value in session.scalars(
                    select(Candle.timestamp).where(
                        Candle.provider == provider,
                        Candle.instrument_key == instrument_key,
                        Candle.timeframe == f"{minutes}minute",
                        Candle.timestamp >= first_stamp,
                        Candle.timestamp <= last_stamp,
                    )
                )
            }
            for stamp, open_, high, low, close, volume, oi in parsed:
                normalized = _naive_utc(stamp)
                if normalized in existing:
                    continue
                session.add(
                    Candle(
                        provider=provider,
                        instrument_key=instrument_key,
                        underlying_code=underlying_code,
                        timeframe=f"{minutes}minute",
                        timestamp=stamp,
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        open_interest=oi,
                        source_verified=True,
                        data_role=data_role,
                    )
                )
                existing.add(normalized)
                inserted += 1
            session.commit()
    if rejected and warnings is not None:
        warnings.append(
            f"{provider}: skipped {rejected} malformed {data_role.lower()} candles for {instrument_key}"
        )
    return inserted


def _monthly_expiries(expiries: list[date]) -> list[date]:
    values: dict[tuple[int, int], date] = {}
    for expiry in expiries:
        key = (expiry.year, expiry.month)
        values[key] = max(values.get(key, expiry), expiry)
    return sorted(values.values())


def _store_contract_version(
    session_factory,
    contract: Contract,
    underlying_code: str,
    first_candle: datetime,
) -> bool:
    valid_from = _naive_utc(first_candle).date()
    with session_factory() as session:
        existing = session.scalar(
            select(Instrument).where(
                Instrument.provider == contract.provider,
                Instrument.instrument_key == contract.instrument_key,
                Instrument.valid_from == valid_from,
            )
        )
        if existing is not None:
            return False
        session.add(
            Instrument(
                provider=contract.provider,
                instrument_key=contract.instrument_key,
                underlying_code=underlying_code,
                exchange=contract.exchange,
                trading_symbol=contract.trading_symbol,
                option_type=contract.option_type if contract.option_type in {"CE", "PE"} else "FU",
                strike=contract.strike,
                expiry=contract.expiry,
                lot_size=contract.lot_size,
                tick_size=contract.tick_size,
                valid_from=valid_from,
                valid_to=contract.expiry,
                active=False,
                instrument_kind="FUTURE" if contract.option_type == "FUT" else "OPTION",
                metadata_verified=contract.metadata_verified,
                metadata_source=contract.metadata_source,
            )
        )
        session.commit()
        return True


def _contract_choices(contracts: list[Contract], spot: Decimal) -> list[Contract]:
    selected: list[Contract] = []
    for option_type in ("CE", "PE"):
        side = [item for item in contracts if item.option_type == option_type]
        strikes = sorted({item.strike for item in side})
        if not strikes:
            continue
        atm = min(strikes, key=lambda strike: abs(strike - spot))
        position = strikes.index(atm)
        itm = strikes[max(0, position - 1)] if option_type == "CE" else strikes[min(len(strikes) - 1, position + 1)]
        selected.extend(item for item in side if item.strike in {atm, itm})
    return selected


def _repair_issues(
    session_factory,
    provider: str,
    indices: list[str],
    start: date,
    end: date,
    *,
    check_option_metadata: bool,
) -> list[str]:
    issues: list[str] = []
    for code in indices:
        for role in ("SPOT", "FUTURE"):
            remaining = _known_gap_count(
                session_factory, provider, code, role, start, end
            )
            if remaining:
                issues.append(
                    f"{code}: {remaining} known {role.lower()} intraday gaps remain after repair"
                )
        if check_option_metadata:
            with session_factory() as session:
                pending_metadata = session.scalar(
                    select(Instrument.id).where(
                        Instrument.provider == provider,
                        Instrument.underlying_code == code,
                        Instrument.instrument_kind == "OPTION",
                        Instrument.metadata_verified.is_(False),
                    ).limit(1)
                )
            if pending_metadata:
                issues.append(
                    f"{code}: historical option lot/tick metadata still requires verification"
                )
    return issues


async def sync_research_data(
    client: BrokerClient,
    session_factory,
    indices: list[str],
    start: date,
    end: date,
    *,
    include_options: bool,
    include_one_minute: bool = False,
    repair_only: bool = False,
    progress: Callable[[int, int, str], Awaitable[None]] | None = None,
) -> SyncResult:
    provider = provider_name(client.provider)
    underlying_count = 0
    futures_count = 0
    option_count = 0
    futures_contract_count = 0
    option_contract_count = 0
    warnings: list[str] = []
    incomplete_reasons: list[str] = []
    completed_steps = 0
    total_steps = len(indices) * (3 if include_options else 2)

    for code in indices:
        underlying_count += await _store_candles(
            client,
            session_factory,
            client.index_key(code),
            code,
            start,
            end,
            provider=provider,
            data_role="SPOT",
            repair_only=repair_only,
            warnings=warnings,
        )
        if include_one_minute:
            underlying_count += await _store_candles(
                client,
                session_factory,
                client.index_key(code),
                code,
                start,
                end,
                provider=provider,
                data_role="SPOT",
                minutes=1,
                repair_only=repair_only,
                warnings=warnings,
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: spot history stored")

    expiry_end = min(date.today(), end + timedelta(days=45))
    for code in indices:
        futures_repair_dates = (
            _known_gap_days(
                session_factory, provider, code, "FUTURE", start, end
            )
            if repair_only
            else None
        )
        try:
            expiries = await client.historical_expiries(code, start, expiry_end)
        except Exception as exc:
            warnings.append(
                f"{code}: historical futures discovery unavailable from {provider}: {exc}"
            )
            continue
        futures_found = False
        for expiry in _monthly_expiries(expiries):
            try:
                contracts = await client.historical_future_contracts(code, expiry)
            except Exception as exc:
                warnings.append(f"{code}: futures metadata unavailable for {expiry}: {exc}")
                continue
            for contract in contracts:
                candle_end = min(end, contract.expiry)
                if candle_end < start:
                    continue
                futures_count += await _store_candles(
                    client,
                    session_factory,
                    contract.instrument_key,
                    code,
                    start,
                    candle_end,
                    provider=provider,
                    contract=contract,
                    data_role="FUTURE",
                    repair_only=repair_only,
                    repair_dates=futures_repair_dates,
                    warnings=warnings,
                )
                with session_factory() as session:
                    first_candle = session.scalar(
                        select(Candle.timestamp)
                        .where(
                            Candle.provider == provider,
                            Candle.instrument_key == contract.instrument_key,
                            Candle.timeframe == "5minute",
                            Candle.source_verified.is_(True),
                        )
                        .order_by(Candle.timestamp)
                        .limit(1)
                    )
                if first_candle is not None:
                    futures_found = True
                    futures_contract_count += int(
                        _store_contract_version(session_factory, contract, code, first_candle)
                    )
        if not futures_found:
            warnings.append(
                f"{code}: no verified {provider} futures candles were stored; VWAP/volume training is unavailable"
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: futures history stored")

    if not include_options:
        incomplete_reasons.extend(
            f"{code}: no verified {provider} futures candles are available for the requested period"
            for code in indices
            if not _has_candles(session_factory, provider, code, "FUTURE", start, end)
        )
        if repair_only:
            incomplete_reasons.extend(
                _repair_issues(
                    session_factory,
                    provider,
                    indices,
                    start,
                    end,
                    check_option_metadata=False,
                )
            )
        return SyncResult(
            underlying_count,
            futures_count,
            0,
            futures_contract_count,
            0,
            warnings,
            incomplete_reasons,
        )

    for code in indices:
        with session_factory() as session:
            underlying_rows = session.scalars(
                select(Candle)
                .where(
                    Candle.provider == provider,
                    Candle.underlying_code == code,
                    Candle.data_role == "SPOT",
                    Candle.timeframe == "5minute",
                    Candle.timestamp >= datetime.combine(start, datetime.min.time()),
                    Candle.timestamp <= datetime.combine(end, datetime.max.time()),
                    Candle.source_verified.is_(True),
                )
                .order_by(Candle.timestamp)
            ).all()
        if not underlying_rows:
            warnings.append(f"{code}: no {provider} underlying candles available")
            continue

        try:
            expiries = await client.historical_expiries(code, start, expiry_end)
        except Exception as exc:
            warnings.append(f"{code}: historical option discovery unavailable from {provider}: {exc}")
            continue
        if not expiries:
            warnings.append(f"{code}: {provider} returned no historical expiries for this period")
            continue
        contracts_by_expiry: dict[date, list[Contract]] = {}
        selected: dict[str, Contract] = {}
        first_selected_day: dict[str, date] = {}
        daily_spot: dict[date, Decimal] = {}
        for row in underlying_rows:
            stamp = row.timestamp.replace(tzinfo=UTC) if row.timestamp.tzinfo is None else row.timestamp.astimezone(UTC)
            daily_spot[stamp.astimezone(ZoneInfo("Asia/Kolkata")).date()] = row.close

        for trading_day, spot in daily_spot.items():
            eligible = [expiry for expiry in expiries if expiry > trading_day]
            if not eligible:
                continue
            expiry = min(eligible)
            if expiry not in contracts_by_expiry:
                if progress:
                    await progress(
                        completed_steps, total_steps,
                        f"{code}: discovering option contracts for {expiry}",
                    )
                contracts_by_expiry[expiry] = await client.historical_contracts(code, expiry)
            for contract in _contract_choices(contracts_by_expiry[expiry], spot):
                selected[contract.instrument_key] = contract
                first_selected_day.setdefault(contract.instrument_key, trading_day)

        if not selected:
            warning = f"{code}: no historical ATM/ITM1 contracts were resolved from {provider}"
            if provider == "GROWW":
                warning += "; Groww contract metadata must include historical lot and tick sizes"
            warnings.append(warning)
            continue

        contract_total = len(selected)
        for contract_number, contract in enumerate(selected.values(), start=1):
            if progress:
                await progress(
                    completed_steps * 100 + 100 * (contract_number - 1) // contract_total,
                    total_steps * 100,
                    f"{code}: downloading option contracts {contract_number}/{contract_total}",
                )
            candle_end = min(end, contract.expiry)
            contract_start = max(start, first_selected_day[contract.instrument_key])
            if candle_end < contract_start:
                continue
            count = await _store_candles(
                client,
                session_factory,
                contract.instrument_key,
                code,
                contract_start,
                candle_end,
                provider=provider,
                contract=contract,
                data_role="OPTION",
                repair_only=repair_only,
                warnings=warnings,
            )
            option_count += count
            with session_factory() as session:
                first_candle = session.scalar(
                    select(Candle.timestamp)
                    .where(
                        Candle.provider == provider,
                        Candle.instrument_key == contract.instrument_key,
                        Candle.timeframe == "5minute",
                        Candle.source_verified.is_(True),
                    )
                    .order_by(Candle.timestamp)
                    .limit(1)
                )
                if first_candle is None:
                    continue
            option_contract_count += int(
                _store_contract_version(session_factory, contract, code, first_candle)
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: option history stored")

    if option_count == 0:
        warnings.append(
            f"No verified {provider} historical option candles were stored. The backtest will fail closed instead of manufacturing premiums."
        )
    for code in indices:
        if not _has_candles(session_factory, provider, code, "FUTURE", start, end):
            incomplete_reasons.append(
                f"{code}: no verified {provider} futures candles are available for the requested period"
            )
        if not _has_candles(session_factory, provider, code, "OPTION", start, end):
            incomplete_reasons.append(
                f"{code}: no verified {provider} option candles are available for the requested period"
            )
    if repair_only:
        incomplete_reasons.extend(
            _repair_issues(
                session_factory,
                provider,
                indices,
                start,
                end,
                check_option_metadata=True,
            )
        )
    return SyncResult(
        underlying_count,
        futures_count,
        option_count,
        futures_contract_count,
        option_contract_count,
        warnings,
        incomplete_reasons,
    )
