from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from .domain import Contract, Quote


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
    strikes = sorted({item.strike for item in chain})
    atm = min(strikes, key=lambda strike: abs(strike - spot))
    index = strikes.index(atm)
    itm1 = strikes[max(0, index - 1)] if direction == "CE" else strikes[min(len(strikes) - 1, index + 1)]
    reasons: list[str] = []
    for strike, moneyness in dict.fromkeys(((atm, "ATM"), (itm1, "ITM1"))):
        contract = next(item for item in chain if item.strike == strike)
        quote = quotes.get(contract.instrument_key)
        if not quote:
            reasons.append(f"{moneyness}: quote unavailable")
            continue
        stamp = quote.timestamp if quote.timestamp.tzinfo else quote.timestamp.replace(tzinfo=UTC)
        checks = (
            (quote.bid > 0 and quote.ask >= quote.bid, "invalid bid/ask"),
            (_spread_pct(quote) <= max_spread_pct, "spread too wide"),
            ((now - stamp.astimezone(UTC)).total_seconds() <= max_quote_age_seconds, "stale quote"),
            (quote.volume > 0 and quote.open_interest > 0, "volume or OI unavailable"),
            (quote.delta is not None and delta_min <= abs(quote.delta) <= delta_max, "delta outside range"),
        )
        failed = next((reason for passed, reason in checks if not passed), None)
        if failed is None:
            return Selection(contract, quote, moneyness, f"{moneyness} passed all filters")
        reasons.append(f"{moneyness}: {failed}")
    return Selection(None, None, None, "; ".join(reasons))

