from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.domain import Contract, Quote
from app.option_selector import select_option


def contract(strike: str, option_type: str = "CE") -> Contract:
    return Contract(f"key-{strike}", f"NIFTY-{strike}-{option_type}", "NSE", "NIFTY", date.today() + timedelta(days=7), Decimal(strike), option_type, 50, Decimal("0.05"))


def quote(item: Contract, spread: str = "1", delta: float = 0.52) -> Quote:
    bid = Decimal("100")
    return Quote(item.instrument_key, datetime.now(UTC), bid, bid + Decimal(spread), Decimal("100.5"), 1000, 5000, delta, 14.2)


def test_atm_selected_before_itm():
    items = [contract("24950"), contract("25000"), contract("25050")]
    quotes = {item.instrument_key: quote(item) for item in items}
    selected = select_option(items, quotes, spot=Decimal("25010"), direction="CE", trading_day=date.today())
    assert selected.moneyness == "ATM"
    assert selected.contract.strike == Decimal("25000")


def test_itm1_fallback_when_atm_spread_is_too_wide():
    items = [contract("24950"), contract("25000"), contract("25050")]
    quotes = {item.instrument_key: quote(item) for item in items}
    quotes["key-25000"] = quote(items[1], spread="5")
    selected = select_option(items, quotes, spot=Decimal("25000"), direction="CE", trading_day=date.today())
    assert selected.moneyness == "ITM1"
    assert selected.contract.strike == Decimal("24950")


def test_expiry_day_contract_is_not_eligible():
    item = Contract("today", "NIFTY", "NSE", "NIFTY", date.today(), Decimal("25000"), "CE", 50, Decimal("0.05"))
    selected = select_option([item], {"today": quote(item)}, spot=Decimal("25000"), direction="CE", trading_day=date.today())
    assert selected.contract is None
