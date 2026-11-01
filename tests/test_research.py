from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Candle
from app.domain import Contract
from app.research import _parse, _store_candles, sync_research_data


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
async def test_option_download_starts_when_contract_first_selected(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'option-window.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    expiry = date(2025, 1, 30)

    class ArchiveClient:
        provider = "GROWW"

        def __init__(self):
            self.option_calls = []

        @staticmethod
        def index_key(_code):
            return "NSE-NIFTY"

        async def historical_candles(self, _key, _start, _end, _minutes):
            return [
                ["2025-01-01T09:15:00", 25000, 25001, 24999, 25000, 0],
                ["2025-01-10T09:15:00", 25050, 25051, 25049, 25050, 0],
                ["2025-01-10T15:20:00", 25200, 25201, 25199, 25200, 0],
            ]

        async def historical_expiries(self, _code, _start, _end):
            return [expiry]

        async def historical_future_contracts(self, _code, _expiry):
            return []

        async def historical_contracts(self, _code, _expiry):
            return [
                Contract(
                    instrument_key=f"CE-{strike}", trading_symbol=f"CE-{strike}",
                    exchange="NSE", underlying_code="NIFTY", expiry=expiry,
                    strike=Decimal(strike), option_type="CE", lot_size=50,
                    tick_size=Decimal("0.05"), provider="GROWW",
                )
                for strike in (24950, 25000, 25050, 25100, 25150, 25200)
            ]

        async def historical_option_candles(self, contract, start, end, minutes):
            self.option_calls.append((contract.instrument_key, start, end, minutes))
            return []

    client = ArchiveClient()
    updates = []

    async def progress(completed, total, message):
        updates.append((completed, total, message))

    await sync_research_data(
        client, factory, ["NIFTY"], date(2025, 1, 1), date(2025, 1, 10),
        include_options=True, progress=progress,
    )

    assert ("CE-25050", date(2025, 1, 10), date(2025, 1, 10), 5) in client.option_calls
    assert ("CE-25200", date(2025, 1, 10), date(2025, 1, 10), 5) in client.option_calls
    assert any("downloading option contracts" in message for _, _, message in updates)
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


@pytest.mark.asyncio
async def test_repair_downloads_selected_day_with_no_stored_option_bars(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'missing-option-day.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        session.add(Candle(
            provider="GROWW", instrument_key="NIFTY-OPTION",
            underlying_code="NIFTY", timeframe="5minute",
            timestamp=datetime(2025, 11, 21, 3, 45, tzinfo=UTC),
            open=100, high=101, low=99, close=100,
            source_verified=True, data_role="OPTION",
        ))
        session.commit()

    class MissingDayClient:
        def __init__(self):
            self.calls = []

        async def historical_option_candles(self, _contract, start, end, _minutes):
            self.calls.append((start, end))
            return [["2025-11-24T09:15:00", 100, 101, 99, 100, 10, 10]]

    contract = Contract(
        instrument_key="NIFTY-OPTION", trading_symbol="NIFTY OPTION",
        exchange="NSE", underlying_code="NIFTY", expiry=date(2025, 11, 25),
        strike=Decimal("25000"), option_type="CE", lot_size=65,
        tick_size=Decimal("0.05"), provider="GROWW",
    )
    client = MissingDayClient()
    inserted = await _store_candles(
        client, factory, contract.instrument_key, "NIFTY",
        date(2025, 11, 21), date(2025, 11, 24), provider="GROWW",
        contract=contract, data_role="OPTION", repair_only=True,
        required_dates={date(2025, 11, 21), date(2025, 11, 24)},
    )

    assert client.calls == [(date(2025, 11, 24), date(2025, 11, 24))]
    assert inserted == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_repair_only_can_force_an_aggregate_futures_gap_day(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'futures-repair.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    trading_day = date(2025, 11, 21)
    with factory() as session:
        session.add_all([
            Candle(provider="GROWW", instrument_key="NSE-NIFTY-FUT", underlying_code="NIFTY", timeframe="5minute", timestamp=datetime(2025, 11, 21, 3, 45, tzinfo=UTC), open=100, high=102, low=99, close=101, volume=100, open_interest=10, source_verified=True, data_role="FUTURE"),
            Candle(provider="GROWW", instrument_key="NSE-NIFTY-FUT", underlying_code="NIFTY", timeframe="5minute", timestamp=datetime(2025, 11, 21, 3, 50, tzinfo=UTC), open=101, high=103, low=100, close=102, volume=100, open_interest=10, source_verified=True, data_role="FUTURE"),
        ])
        session.commit()

    class RepairClient:
        def __init__(self):
            self.calls = []

        async def historical_candles(self, instrument_key, start, end, minutes):
            self.calls.append((start, end, minutes))
            return [["2025-11-21T09:25:00", 102, 104, 101, 103, 100, 10]]

    client = RepairClient()
    inserted = await _store_candles(
        client=client,
        session_factory=factory,
        instrument_key="NSE-NIFTY-FUT",
        underlying_code="NIFTY",
        start=trading_day,
        end=trading_day,
        provider="GROWW",
        data_role="FUTURE",
        repair_only=True,
        repair_dates={trading_day},
    )

    assert inserted == 1
    assert client.calls == [(trading_day, trading_day, 5)]
    engine.dispose()


@pytest.mark.asyncio
async def test_repair_without_options_reports_remaining_futures_gaps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'remaining-gap.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    trading_day = date(2025, 11, 21)
    with factory() as session:
        for minute in (45, 50, 55):
            session.add(Candle(provider="GROWW", instrument_key="NSE-NIFTY", underlying_code="NIFTY", timeframe="5minute", timestamp=datetime(2025, 11, 21, 3, minute, tzinfo=UTC), open=100, high=102, low=99, close=101, volume=0, open_interest=0, source_verified=True, data_role="SPOT"))
        for minute in (45, 55):
            session.add(Candle(provider="GROWW", instrument_key="NSE-NIFTY-FUT", underlying_code="NIFTY", timeframe="5minute", timestamp=datetime(2025, 11, 21, 3, minute, tzinfo=UTC), open=100, high=102, low=99, close=101, volume=100, open_interest=10, source_verified=True, data_role="FUTURE"))
        session.commit()

    class UnavailableArchiveClient:
        provider = "GROWW"

        @staticmethod
        def index_key(_code):
            return "NSE-NIFTY"

        async def historical_candles(self, instrument_key, start, end, minutes):
            return []

        async def historical_expiries(self, code, start, end):
            return []

    result = await sync_research_data(
        UnavailableArchiveClient(),
        factory,
        ["NIFTY"],
        trading_day,
        trading_day,
        include_options=False,
        repair_only=True,
    )

    assert result.incomplete_reasons == [
        "NIFTY: 1 known future intraday gaps remain after repair"
    ]
    engine.dispose()
