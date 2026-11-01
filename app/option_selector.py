from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, ROUND_HALF_UP

from .domain import Contract, Quote


STRIKE_STEPS = {
    "NIFTY": Decimal("50"),
    "BANK_NIFTY": Decimal("100"),
    "SENSEX": Decimal("100"),
}


def expected_atm_strike(underlying_code: str, spot: Decimal) -> Decimal | None:
    step = STRIKE_STEPS.get(underlying_code)
    if step is None or not spot.is_finite() or spot <= 0:
        return None
    return (spot / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step


def atm_itm1_strikes(
    underlying_code: str, spot: Decimal, direction: str, strikes: list[Decimal]
) -> tuple[Decimal, Decimal | None] | None:
    """Return true ATM and adjacent ITM1 strikes, not the nearest sparse strike."""
    step = STRIKE_STEPS.get(underlying_code)
    atm = expected_atm_strike(underlying_code, spot)
    if step is None or atm is None or direction not in {"CE", "PE"}:
        return None
    available = set(strikes)
    if atm not in available:
        return None
    itm = atm - step if direction == "CE" else atm + step
    return atm, itm if itm in available else None


@dataclass(frozen=True)
class Selection:
    contract: Contract | None
    quote: Quote | None
    moneyness: str | None
    reason: str


def _spread_pct(quote: Quote) -> Decimal:
    midpoint = (quote.bid + quote.ask) / 2
    return Decimal("999") if midpoint <= 0 else (quote.ask - quote.bid) / midpoint * 100


def select_option(
    contracts: list[Contract],
    quotes: dict[str, Quote],
    *,
    spot: Decimal,
    direction: str,
    trading_day: date,
    now: datetime | None = None,
    max_spread_pct: Decimal = Decimal("1.5"),
    max_quote_age_seconds: int = 3,
    delta_min: float = 0.45,
    delta_max: float = 0.60,
) -> Selection:
    now = now or datetime.now(UTC)
    expiries = sorted({item.expiry for item in contracts if item.expiry > trading_day})
    if not expiries:
        return Selection(None, None, None, "No eligible non-expiry-day contract")
    chain = [item for item in contracts if item.expiry == expiries[0] and item.option_type == direction]
    if not chain:
        return Selection(None, None, None, "No contract for signal direction")
    selected_strikes = atm_itm1_strikes(chain[0].underlying_code, spot, direction, [item.strike for item in chain])
    if selected_strikes is None:
        return Selection(None, None, None, "Near-ATM strike missing from option chain")
    atm, itm1 = selected_strikes
    reasons: list[str] = []
    choices = [(atm, "ATM")]
    if itm1 is not None:
        choices.append((itm1, "ITM1"))
    for strike, moneyness in choices:
        contract = next(item for item in chain if item.strike == strike)
        quote = quotes.get(contract.instrument_key)
        if not quote:
            reasons.append(f"{moneyness}: quote unavailable")
            continue
        stamp = quote.timestamp if quote.timestamp.tzinfo else quote.timestamp.replace(tzinfo=UTC)
        quote_age = (now - stamp.astimezone(UTC)).total_seconds()
        checks = (
            (quote.bid > 0 and quote.ask >= quote.bid, "invalid bid/ask"),
            (_spread_pct(quote) <= max_spread_pct, "spread too wide"),
            (0 <= quote_age <= max_quote_age_seconds, "stale or future-dated quote"),
            (quote.volume > 0 and quote.open_interest > 0, "volume or OI unavailable"),
            (quote.delta is not None and delta_min <= abs(quote.delta) <= delta_max, "delta outside range"),
        )
        failed = next((reason for passed, reason in checks if not passed), None)
        if failed is None:
            return Selection(contract, quote, moneyness, f"{moneyness} passed all filters")
        reasons.append(f"{moneyness}: {failed}")
    return Selection(None, None, None, "; ".join(reasons))

