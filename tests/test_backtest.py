from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtest import _intrabar_underlying_outcome, metrics
from app.models import Candle


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
