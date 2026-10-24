from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from .models import Candle, Instrument


def option_replay_issue(session: Session, provider: str, code: str, start: date, end: date) -> str | None:
    """Check that option candles have matching, dated contract metadata."""
    period = (
        Candle.provider == provider,
        Candle.underlying_code == code,
        Candle.data_role == "OPTION",
        Candle.timeframe == "5minute",
        Candle.source_verified.is_(True),
        Candle.timestamp >= datetime.combine(start, time.min, tzinfo=UTC),
        Candle.timestamp <= datetime.combine(end, time.max, tzinfo=UTC),
    )
    if session.scalar(select(Candle.id).where(*period).limit(1)) is None:
        return f"{code}: no verified option candles in {start} to {end}"
    matching = session.scalar(
        select(Candle.id)
        .join(Instrument, and_(
            Instrument.provider == Candle.provider,
            Instrument.instrument_key == Candle.instrument_key,
        ))
        .where(
            *period,
            Instrument.instrument_kind == "OPTION",
            Instrument.metadata_verified.is_(True),
            Instrument.lot_size > 0,
            Instrument.valid_from <= func.date(Candle.timestamp),
            (Instrument.valid_to.is_(None) | (Instrument.valid_to >= func.date(Candle.timestamp))),
            Instrument.expiry > func.date(Candle.timestamp),
        )
        .limit(1)
    )
    if matching is None:
        return f"{code}: option candles have no matching verified historical lot-size metadata"
    return None


@dataclass(frozen=True)
class DataReadiness:
    provider: str
    underlying_code: str
    spot_count: int
    futures_count: int
    option_count: int
    aligned_count: int
    alignment_percent: float
    zero_volume_percent: float
    missing_intervals: int
    spot_missing_intervals: int
    futures_missing_intervals: int
    first_missing_timestamp: datetime | None
    last_missing_timestamp: datetime | None
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    verified_contracts: int
    pending_contracts: int
    metadata_gap_contracts: int
    training_ready: bool
    backtest_ready: bool
    reason: str


def _utc_naive(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).replace(tzinfo=None)


INDIA_TIME = ZoneInfo("Asia/Kolkata")
REGULAR_SESSION = ((time(9, 15), time(15, 30)),)
# NSE/MSD/60677 and NSE/MSD/61893 scheduled a midday DR-site switchover.
SPECIAL_SESSIONS = {
    date(2024, 3, 2): ((time(9, 15), time(10, 0)), (time(11, 30), time(12, 30))),
    date(2024, 5, 18): ((time(9, 15), time(10, 0)), (time(11, 30), time(12, 30))),
}


def _missing_timestamps(values: list[datetime], minutes: int = 5) -> list[datetime]:
    if len(values) < 2:
        return []
    normalized = sorted(_utc_naive(value) for value in values)
    missing: list[datetime] = []
    for previous, current in zip(normalized, normalized[1:]):
        local_day = previous.replace(tzinfo=UTC).astimezone(INDIA_TIME).date()
        if local_day != current.replace(tzinfo=UTC).astimezone(INDIA_TIME).date():
            continue
        candidate = previous + timedelta(minutes=minutes)
        sessions = SPECIAL_SESSIONS.get(local_day, REGULAR_SESSION)
        while candidate < current:
            local_time = candidate.replace(tzinfo=UTC).astimezone(INDIA_TIME).time()
            if any(open_time <= local_time <= close_time for open_time, close_time in sessions):
                missing.append(candidate.replace(tzinfo=UTC))
            candidate += timedelta(minutes=minutes)
    return missing


def _missing_intervals(values: list[datetime]) -> int:
    """Return genuine intraday gaps inside known exchange sessions."""
    return len(_missing_timestamps(values))


def research_readiness(session: Session) -> list[DataReadiness]:
    rows: list[DataReadiness] = []
    for provider in ("UPSTOX", "GROWW"):
        for code in ("NIFTY", "BANK_NIFTY", "SENSEX"):
            base = (Candle.provider == provider, Candle.underlying_code == code, Candle.source_verified.is_(True))
            spot_values = session.scalars(
                select(Candle.timestamp)
                .where(*base, Candle.data_role == "SPOT", Candle.timeframe == "5minute")
                .order_by(Candle.timestamp)
            ).all()
            future_values = session.scalars(
                select(Candle.timestamp)
                .where(*base, Candle.data_role == "FUTURE", Candle.timeframe == "5minute")
            ).all()
            option_count = session.scalar(
                select(func.count()).select_from(Candle).where(*base, Candle.data_role == "OPTION")
            ) or 0
            zero_volume = session.scalar(
                select(func.count()).select_from(Candle).where(
                    *base,
                    Candle.data_role == "SPOT",
                    Candle.timeframe == "5minute",
                    Candle.volume == 0,
                )
            ) or 0
            verified_contracts = session.scalar(
                select(func.count()).select_from(Instrument).where(
                    Instrument.provider == provider,
                    Instrument.underlying_code == code,
                    Instrument.instrument_kind == "OPTION",
                    Instrument.metadata_verified.is_(True),
                    Instrument.lot_size > 0,
                )
            ) or 0
            pending_contracts = session.scalar(
                select(func.count()).select_from(Instrument).where(
                    Instrument.provider == provider,
                    Instrument.underlying_code == code,
                    Instrument.instrument_kind == "OPTION",
                    Instrument.metadata_verified.is_(False),
                )
            ) or 0
            option_instrument_keys = set(session.scalars(
                select(Candle.instrument_key).where(
                    *base,
                    Candle.data_role == "OPTION",
                    Candle.timeframe == "5minute",
                ).distinct()
            ).all())
            verified_option_keys = set(session.scalars(
                select(Instrument.instrument_key).where(
                    Instrument.provider == provider,
                    Instrument.underlying_code == code,
                    Instrument.instrument_kind == "OPTION",
                    Instrument.metadata_verified.is_(True),
                    Instrument.lot_size > 0,
                ).distinct()
            ).all())
            metadata_gap_contracts = len(option_instrument_keys - verified_option_keys)
            spot_stamps = {_utc_naive(value) for value in spot_values}
            future_stamps = {_utc_naive(value) for value in future_values}
            spot_missing = _missing_timestamps(list(spot_values))
            futures_missing = _missing_timestamps(list(future_values))
            all_missing = sorted(spot_missing + futures_missing)
            aligned = len(spot_stamps & future_stamps)
            training_ready = aligned >= 500
            replay_issue = (
                option_replay_issue(
                    session, provider, code,
                    _utc_naive(min(spot_values)).date(),
                    _utc_naive(max(spot_values)).date(),
                )
                if spot_values and option_count and verified_contracts else None
            )
            backtest_ready = training_ready and option_count > 0 and verified_contracts > 0 and metadata_gap_contracts == 0 and not spot_missing and not futures_missing and replay_issue is None
            reasons = []
            if len(spot_values) < 500:
                reasons.append("spot history below 500 candles")
            if aligned < 500:
                reasons.append("aligned futures history below 500 candles")
            if spot_missing:
                reasons.append(f"{len(spot_missing)} spot intraday gaps")
            if futures_missing:
                reasons.append(f"{len(futures_missing)} futures intraday gaps")
            if option_count == 0:
                reasons.append("no option candles")
            if verified_contracts == 0:
                reasons.append("no verified option lot metadata")
            elif replay_issue:
                reasons.append(replay_issue.split(": ", 1)[-1])
            if metadata_gap_contracts:
                reasons.append(f"{metadata_gap_contracts} option instruments lack verified metadata")
            rows.append(
                DataReadiness(
                    provider=provider,
                    underlying_code=code,
                    spot_count=len(spot_values),
                    futures_count=len(future_values),
                    option_count=int(option_count),
                    aligned_count=aligned,
                    alignment_percent=round(aligned / len(spot_stamps) * 100, 1) if spot_stamps else 0.0,
                    zero_volume_percent=round(int(zero_volume) / len(spot_values) * 100, 1) if spot_values else 0.0,
                    missing_intervals=len(spot_missing),
                    spot_missing_intervals=len(spot_missing),
                    futures_missing_intervals=len(futures_missing),
                    first_missing_timestamp=all_missing[0] if all_missing else None,
                    last_missing_timestamp=all_missing[-1] if all_missing else None,
                    first_timestamp=min(spot_values) if spot_values else None,
                    last_timestamp=max(spot_values) if spot_values else None,
                    verified_contracts=int(verified_contracts),
                    pending_contracts=int(pending_contracts),
                    metadata_gap_contracts=metadata_gap_contracts,
                    training_ready=training_ready,
                    backtest_ready=backtest_ready,
                    reason="Ready" if backtest_ready else "; ".join(reasons),
                )
            )
    return rows
