from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal


@dataclass(frozen=True)
class Quote:
    instrument_key: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: int
    open_interest: int
    delta: float | None = None
    iv: float | None = None


@dataclass(frozen=True)
class Contract:
    instrument_key: str
    trading_symbol: str
    exchange: str
    underlying_code: str
    expiry: date
    strike: Decimal
    option_type: str
    lot_size: int
    tick_size: Decimal
    provider: str = "UPSTOX"
    segment: str = "FNO"
    exchange_token: str | None = None
    metadata_verified: bool = True
    metadata_source: str = "BROKER"


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    status: str
    quantity: int
    filled_quantity: int
    average_price: Decimal
    raw: dict

