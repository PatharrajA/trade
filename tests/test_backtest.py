from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.backtest import _intrabar_underlying_outcome, metrics, option_replay_issue
from app.models import Candle, Instrument


def test_metrics_are_chronological_and_after_costs():
    rows = [
        {"exit_time": datetime(2025, 2, 1, tzinfo=UTC), "underlying_code": "NIFTY", "option_type": "PE", "moneyness": "ITM1", "net_pnl": Decimal("-500")},
        {"exit_time": datetime(2025, 1, 1, tzinfo=UTC), "underlying_code": "NIFTY", "option_type": "CE", "moneyness": "ATM", "net_pnl": Decimal("1000")},
    ]
    result = metrics(rows, Decimal("100000"))
    assert result["net_pnl"] == 500.0
    assert result["trades"] == 2
    assert result["equity_curve"] == [101000.0, 100500.0]


def test_one_minute_data_resolves_stop_before_later_target(session):
    stamp = datetime(2026, 1, 2, 4, 0, tzinfo=UTC)
    five_minute = Candle(
        provider="UPSTOX", instrument_key="spot", underlying_code="NIFTY", timeframe="5minute",
        timestamp=stamp, open=100, high=106, low=98, close=103, volume=0, open_interest=0,
        source_verified=True, data_role="SPOT",
    )
    session.add_all([
        Candle(provider="UPSTOX", instrument_key="spot", underlying_code="NIFTY", timeframe="1minute", timestamp=stamp, open=100, high=102, low=98, close=99, volume=0, open_interest=0, source_verified=True, data_role="SPOT"),
        Candle(provider="UPSTOX", instrument_key="spot", underlying_code="NIFTY", timeframe="1minute", timestamp=stamp + timedelta(minutes=1), open=99, high=106, low=99, close=105, volume=0, open_interest=0, source_verified=True, data_role="SPOT"),
    ])
    session.commit()
    assert _intrabar_underlying_outcome(session, "UPSTOX", "NIFTY", five_minute, "CE", Decimal("99"), Decimal("105")) == "UNDERLYING_STOP"


def test_option_replay_requires_matching_historical_metadata(session):
    day = date(2026, 1, 2)
    candle = Candle(
        provider="GROWW", instrument_key="old-option", underlying_code="NIFTY",
        timeframe="5minute", timestamp=datetime(2026, 1, 2, 4, tzinfo=UTC),
        open=100, high=102, low=99, close=101, volume=100, open_interest=1,
        source_verified=True, data_role="OPTION",
    )
    contract = Instrument(
        provider="GROWW", instrument_key="old-option", underlying_code="NIFTY",
        exchange="NSE", trading_symbol="NIFTY TEST CE", option_type="CE",
        strike=25000, expiry=date(2026, 1, 29), lot_size=0,
        tick_size=Decimal("0.05"), valid_from=day,
        instrument_kind="OPTION", metadata_verified=False,
        metadata_source="GROWW_HISTORICAL_SYMBOL",
    )
    session.add_all([candle, contract])
    session.commit()
    assert "lot-size metadata" in option_replay_issue(session, "GROWW", "NIFTY", day, day)
    contract.lot_size = 65
    contract.metadata_verified = True
    session.commit()
    assert option_replay_issue(session, "GROWW", "NIFTY", day, day) is None
