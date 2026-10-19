from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from app.domain import Contract, Quote
from app.execution import ExecutionService
from app.models import RuntimeSetting, Signal, Trade


class BrokerMustNotBeCalled:
    async def place_market_order(self, **_kwargs):
        raise AssertionError("Paper mode called an Upstox order endpoint")


@pytest.mark.asyncio
async def test_paper_entry_and_exit_never_place_broker_order(session):
    runtime = session.get(RuntimeSetting, 1)
    runtime.execution_mode, runtime.entries_paused = "PAPER", False
    session.commit()
    service = ExecutionService(BrokerMustNotBeCalled())
    contract = Contract("NSE_FO|1", "NIFTY TEST CE", "NSE", "NIFTY", date.today() + timedelta(days=7), Decimal("25000"), "CE", 50, Decimal("0.05"))
    quote = Quote(contract.instrument_key, datetime.now(UTC), Decimal("89"), Decimal("90"), Decimal("89.5"), 1000, 5000, .52, 14)
    trade = await service.enter(session, underlying_code="NIFTY", candle_time=datetime.now(UTC), regime="UPTREND", confidence=.8, direction="CE", underlying_price=Decimal("25000"), underlying_stop=Decimal("24950"), contract=contract, quote=quote, moneyness="ATM", decision_context={"checks": {"breakout": True}})
    assert trade.status == "OPEN" and trade.mode == "PAPER"
    assert trade.entry_bid == Decimal("89")
    assert trade.entry_ask == Decimal("90")
    assert trade.risk_budget == Decimal("1250")
    assert "breakout" in session.get(Signal, trade.signal_id).decision_json
    exit_quote = Quote(contract.instrument_key, datetime.now(UTC), Decimal("110"), Decimal("111"), Decimal("110.5"), 1000, 5000, .55, 14)
    await service.exit(session, trade, exit_quote, "TEST_EXIT")
    assert trade.status == "CLOSED" and trade.gross_pnl > 0
    assert session.query(Trade).count() == 1
