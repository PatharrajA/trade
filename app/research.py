from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from sqlalchemy import select

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


def _parse(raw: list) -> tuple[datetime, Decimal, Decimal, Decimal, Decimal, int, int]:
    stamp = datetime.fromisoformat(str(raw[0]).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    return (
        stamp.astimezone(UTC),
        *(Decimal(str(raw[index])) for index in range(1, 5)),
        int(raw[5] or 0),
        int(raw[6] or 0) if len(raw) > 6 else 0,
    )


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
) -> int:
    inserted = 0
    for chunk_start, chunk_end in _chunks(start, end):
        rows = (
            await client.historical_option_candles(contract, chunk_start, chunk_end, minutes)
            if contract
            else await client.historical_candles(instrument_key, chunk_start, chunk_end, minutes)
        )
        parsed = [_parse(row) for row in rows]
        if not parsed:
            continue
        with session_factory() as session:
            existing = {
                _naive_utc(value)
                for value in session.scalars(
                    select(Candle.timestamp).where(
                        Candle.provider == provider,
                        Candle.instrument_key == instrument_key,
                        Candle.timeframe == f"{minutes}minute",
                        Candle.timestamp >= datetime.combine(chunk_start, datetime.min.time()),
                        Candle.timestamp <= datetime.combine(chunk_end, datetime.max.time()),
                    )
                )
            }
            for stamp, open_, high, low, close, volume, oi in parsed:
                if _naive_utc(stamp) in existing:
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
                inserted += 1
            session.commit()
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


async def sync_research_data(
    client: BrokerClient,
    session_factory,
    indices: list[str],
    start: date,
    end: date,
    *,
    include_options: bool,
    include_one_minute: bool = False,
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
            )
        completed_steps += 1
        if progress:
            await progress(completed_steps, total_steps, f"{code}: spot history stored")

    expiry_end = min(date.today(), end + timedelta(days=45))
    for code in indices:
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
        return SyncResult(
            underlying_count,
            futures_count,
            0,
            futures_contract_count,
            0,
            warnings,
            [
                f"{code}: no verified {provider} futures candles are available for the requested period"
                for code in indices
                if not _has_candles(session_factory, provider, code, "FUTURE", start, end)
            ],
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
                contracts_by_expiry[expiry] = await client.historical_contracts(code, expiry)
            for contract in _contract_choices(contracts_by_expiry[expiry], spot):
                selected[contract.instrument_key] = contract

        if not selected:
            warning = f"{code}: no historical ATM/ITM1 contracts were resolved from {provider}"
            if provider == "GROWW":
                warning += "; Groww contract metadata must include historical lot and tick sizes"
            warnings.append(warning)
            continue

        for contract in selected.values():
            candle_end = min(end, contract.expiry)
            if candle_end < start:
                continue
            count = await _store_candles(
                client,
                session_factory,
                contract.instrument_key,
                code,
                start,
                candle_end,
                provider=provider,
                contract=contract,
                data_role="OPTION",
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
    return SyncResult(
        underlying_count,
        futures_count,
        option_count,
        futures_contract_count,
        option_contract_count,
        warnings,
        incomplete_reasons,
    )
