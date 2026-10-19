from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


@dataclass(frozen=True)
class RiskLimits:
    capital: Decimal
    risk_per_trade_pct: Decimal
    daily_loss_limit_pct: Decimal
    max_trades_per_day: int
    max_consecutive_losses: int
    max_open_positions: int
    live_max_lots: int
    premium_emergency_stop_pct: Decimal
    reward_risk: Decimal


@dataclass(frozen=True)
class PositionPlan:
    allowed: bool
    quantity: int
    lots: int
    risk_budget: Decimal
    premium_stop: Decimal
    reason: str


def limits_from_settings(settings: dict[str, str]) -> RiskLimits:
    return RiskLimits(
        capital=Decimal(settings["paper_capital"]),
        risk_per_trade_pct=Decimal(settings["risk_per_trade_pct"]),
        daily_loss_limit_pct=Decimal(settings["daily_loss_limit_pct"]),
        max_trades_per_day=int(settings["max_trades_per_day"]),
        max_consecutive_losses=int(settings["max_consecutive_losses"]),
        max_open_positions=int(settings["max_open_positions"]),
        live_max_lots=int(settings["live_max_lots"]),
        premium_emergency_stop_pct=Decimal(settings["premium_emergency_stop_pct"]),
        reward_risk=Decimal(settings["reward_risk"]),
    )


def round_to_tick(value: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    return (value / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_size


def paper_fill(side: str, bid: Decimal, ask: Decimal, slippage_pct: Decimal, tick_size: Decimal) -> Decimal:
    if bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError("Invalid executable quote")
    multiplier = Decimal("1") + slippage_pct / 100 if side == "BUY" else Decimal("1") - slippage_pct / 100
    base = ask if side == "BUY" else bid
    return round_to_tick(base * multiplier, tick_size)


def position_size(entry_price: Decimal, lot_size: int, limits: RiskLimits, mode: str) -> PositionPlan:
    if entry_price <= 0 or lot_size <= 0:
        return PositionPlan(False, 0, 0, Decimal("0"), Decimal("0"), "Invalid price or lot size")
    premium_stop = entry_price * (Decimal("1") - limits.premium_emergency_stop_pct / 100)
    risk_per_lot = (entry_price - premium_stop) * lot_size
    budget = limits.capital * limits.risk_per_trade_pct / 100
    if risk_per_lot > budget:
        return PositionPlan(False, 0, 0, budget, premium_stop, "One lot exceeds permitted risk")
    lots = int((budget / risk_per_lot).to_integral_value(rounding=ROUND_DOWN))
    if mode == "LIVE":
        lots = min(lots, limits.live_max_lots)
    if lots < 1:
        return PositionPlan(False, 0, 0, budget, premium_stop, "No whole lot fits permitted risk")
    return PositionPlan(True, lots * lot_size, lots, budget, premium_stop, "Risk checks passed")


def can_enter(
    *,
    limits: RiskLimits,
    open_positions: int,
    trades_today: int,
    consecutive_losses: int,
    realized_pnl_today: Decimal,
    quote_fresh: bool,
    broker_connected: bool,
    reconciled: bool,
) -> tuple[bool, str]:
    checks = (
        (open_positions < limits.max_open_positions, "Maximum open positions reached"),
        (trades_today < limits.max_trades_per_day, "Maximum trades per day reached"),
        (consecutive_losses < limits.max_consecutive_losses, "Consecutive-loss limit reached"),
        (realized_pnl_today > -(limits.capital * limits.daily_loss_limit_pct / 100), "Daily-loss limit reached"),
        (quote_fresh, "Market quote is stale"),
        (broker_connected, "Selected broker is disconnected"),
        (reconciled, "Broker and database positions do not match"),
    )
    return next(((False, reason) for passed, reason in checks if not passed), (True, "Risk checks passed"))


def option_charges(
    entry: Decimal,
    exit_: Decimal,
    quantity: int,
    broker: str = "UPSTOX",
    trade_date: date | None = None,
    exchange_name: str = "NSE",
) -> Decimal:
    """Estimate Indian index-option charges using the rates effective on the trade date."""
    effective_date = trade_date or date.today()
    buy, sell = entry * quantity, exit_ * quantity
    # Upstox Basic and Groww currently charge ₹20 per executed options order.
    # Keep Plus explicit because it charges ₹30 per executed order.
    brokerage = Decimal("60") if broker.upper() == "UPSTOX_PLUS" else Decimal("40")
    # The options sell-side STT increased from 0.10% to 0.15% on 1 April 2026.
    stt_rate = Decimal("0.0015") if effective_date >= date(2026, 4, 1) else Decimal("0.001")
    stt = sell * stt_rate
    turnover = buy + sell
    if exchange_name.upper() == "BSE":
        exchange = turnover * Decimal("0.000325")
    elif effective_date >= date(2026, 3, 1):
        exchange = turnover * Decimal("0.0003553")
    elif effective_date >= date(2024, 10, 1):
        exchange = turnover * Decimal("0.0003503")
    else:
        exchange = turnover * Decimal("0.000495")
    sebi = (buy + sell) * Decimal("0.000001")
    stamp = buy * Decimal("0.00003")
    ipft = turnover * Decimal("0.000005") if exchange_name.upper() == "NSE" else Decimal("0")
    gst = (brokerage + exchange + sebi + ipft) * Decimal("0.18")
    return (brokerage + stt + exchange + sebi + ipft + stamp + gst).quantize(Decimal("0.01"))

