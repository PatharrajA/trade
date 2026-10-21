from __future__ import annotations

import asyncio
import csv
import io
import os
import re
import time as monotonic_time
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from .domain import BrokerOrder, Contract, Quote


class GrowwError(RuntimeError):
    pass


GROWW_INDEX_KEYS = {
    "NIFTY": "NSE-NIFTY",
    "BANK_NIFTY": "NSE-BANKNIFTY",
    "SENSEX": "BSE-SENSEX",
}
GROWW_UNDERLYINGS = {
    "NIFTY": ("NSE", "NIFTY"),
    "BANK_NIFTY": ("NSE", "BANKNIFTY"),
    "SENSEX": ("BSE", "SENSEX"),
}


class GrowwClient:
    provider = "GROWW"
    api_url = "https://api.groww.in"
    instruments_url = "https://growwapi-assets.groww.in/instruments/instrument.csv"

    def __init__(self, token_file: Path, client: httpx.AsyncClient | None = None) -> None:
        self.token_file = token_file
        self.client = client or httpx.AsyncClient(timeout=20)
        self._instruments: dict[str, dict[str, str]] = {}
        self._loaded_on: date | None = None
        # Groww applies shared limits to historical endpoints. Keep research
        # traffic conservative and serialize it so a multi-year discovery loop
        # cannot burst dozens of monthly requests at once.
        self._historical_request_lock = asyncio.Lock()
        self._last_historical_request_at = 0.0
        self._historical_min_interval_seconds = 0.5
        self._maximum_rate_limit_retries = 5

    def access_token(self) -> str:
        try:
            return self.token_file.read_text().strip()
        except FileNotFoundError:
            return ""

    def save_token(self, token: str) -> None:
        token = token.strip()
        if len(token) < 20:
            raise ValueError("Groww token appears invalid")
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(token)
        os.chmod(self.token_file, 0o600)

    def index_key(self, underlying_code: str) -> str:
        try:
            return GROWW_INDEX_KEYS[underlying_code]
        except KeyError as exc:
            raise GrowwError(f"Unsupported Groww index: {underlying_code}") from exc

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        if not token:
            raise GrowwError("Groww access token is missing")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-VERSION": "1.0",
        }

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        async def send() -> httpx.Response:
            return await self.client.request(
                method,
                f"{self.api_url}{path}",
                headers=self._headers(self.access_token()),
                **kwargs,
            )

        if not path.startswith("/v1/historical/"):
            return await send()

        async with self._historical_request_lock:
            elapsed = monotonic_time.monotonic() - self._last_historical_request_at
            delay = self._historical_min_interval_seconds - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await send()
            finally:
                self._last_historical_request_at = monotonic_time.monotonic()

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            return str(body.get("message") or body.get("error") or body)
        except Exception:
            return response.text[:160]

    async def _json(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response: httpx.Response | None = None
        maximum_retries = (
            self._maximum_rate_limit_retries if path.startswith("/v1/historical/") else 0
        )
        for attempt in range(maximum_retries + 1):
            response = await self._request(method, path, **kwargs)
            if response.status_code != 429:
                break
            if attempt >= maximum_retries:
                break
            retry_after = response.headers.get("Retry-After", "")
            try:
                wait_seconds = max(float(retry_after), 0.0)
            except ValueError:
                wait_seconds = float(2**attempt)
            if not retry_after:
                wait_seconds = float(2**attempt)
            await asyncio.sleep(min(wait_seconds, 30.0))

        assert response is not None
        if response.is_error:
            detail = self._response_detail(response)
            raise GrowwError(f"Groww request failed ({response.status_code}){': ' + detail if detail else ''}")
        payload = response.json()
        if str(payload.get("status", "SUCCESS")).upper() != "SUCCESS":
            raise GrowwError(str(payload.get("message") or payload.get("error") or "Groww request was unsuccessful"))
        return payload

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        try:
            return Decimal(str(value)) if value not in (None, "") else Decimal("0")
        except (InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    async def validate_token(self, token: str, *, market_data: bool) -> tuple[bool, str]:
        token = token.strip()
        if len(token) < 20:
            return False, "Token appears invalid"
        try:
            path = "/v1/live-data/ltp"
            params = {"segment": "CASH", "exchange_symbols": "NSE_NIFTY"}
            if not market_data:
                path, params = "/v1/user/detail", None
            response = await self.client.get(
                f"{self.api_url}{path}", headers=self._headers(token), params=params
            )
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("status", "SUCCESS")).upper() != "SUCCESS":
                return False, str(payload.get("message") or "Groww rejected the token")
            return True, "Groww API access verified"
        except httpx.HTTPStatusError as exc:
            return False, f"Groww rejected the token ({exc.response.status_code})"
        except Exception as exc:
            return False, str(exc)

    async def health(self) -> tuple[bool, str]:
        try:
            payload = await self._json("GET", "/v1/user/detail")
            user = payload.get("payload") or {}
            return True, str(user.get("name") or user.get("user_name") or "Connected")
        except Exception as exc:
            return False, str(exc)

    async def market_data_health(self) -> tuple[bool, str]:
        try:
            quotes = await self.quotes([self.index_key("NIFTY")])
            quote = quotes.get(self.index_key("NIFTY"))
            if not quote or quote.last <= 0:
                return False, "NIFTY quote was not returned"
            return True, f"NIFTY ₹{quote.last}"
        except Exception as exc:
            return False, str(exc)

    async def _load_instruments(self, *, force: bool = False) -> None:
        if self._instruments and self._loaded_on == date.today() and not force:
            return
        response = await self.client.get(self.instruments_url)
        response.raise_for_status()
        rows = csv.DictReader(io.StringIO(response.text))
        self._instruments = {
            str(row.get("groww_symbol", "")).strip(): {str(k): str(v or "").strip() for k, v in row.items()}
            for row in rows
            if str(row.get("groww_symbol", "")).strip()
        }
        self._loaded_on = date.today()

    @staticmethod
    def _tick(value: str) -> Decimal:
        tick = Decimal(value or "0.05")
        return tick if tick > 0 else Decimal("0.05")

    @classmethod
    def _contract_from_row(cls, row: dict[str, str], underlying_code: str) -> Contract | None:
        try:
            option_type = row.get("instrument_type", "").upper()
            if option_type not in {"CE", "PE"}:
                return None
            return Contract(
                instrument_key=row["groww_symbol"],
                trading_symbol=row["trading_symbol"],
                exchange=row["exchange"],
                underlying_code=underlying_code,
                expiry=date.fromisoformat(row["expiry_date"]),
                strike=Decimal(row["strike_price"]),
                option_type=option_type,
                lot_size=int(Decimal(row["lot_size"])),
                tick_size=cls._tick(row.get("tick_size", "0.05")),
                provider="GROWW",
                segment=row.get("segment", "FNO") or "FNO",
                exchange_token=row.get("exchange_token") or None,
                metadata_verified=True,
                metadata_source="GROWW_INSTRUMENT_MASTER",
            )
        except (KeyError, ValueError, ArithmeticError):
            return None

    @classmethod
    def _future_from_row(cls, row: dict[str, str], underlying_code: str) -> Contract | None:
        try:
            instrument_type = row.get("instrument_type", "").upper()
            if instrument_type not in {"FUT", "FUTIDX", "FUTURE"}:
                return None
            return Contract(
                instrument_key=row["groww_symbol"],
                trading_symbol=row["trading_symbol"],
                exchange=row["exchange"],
                underlying_code=underlying_code,
                expiry=date.fromisoformat(row["expiry_date"]),
                strike=Decimal("0"),
                option_type="FUT",
                lot_size=int(Decimal(row["lot_size"])),
                tick_size=cls._tick(row.get("tick_size", "0.05")),
                provider="GROWW",
                segment=row.get("segment", "FNO") or "FNO",
                exchange_token=row.get("exchange_token") or None,
                metadata_verified=True,
                metadata_source="GROWW_INSTRUMENT_MASTER",
            )
        except (KeyError, ValueError, ArithmeticError):
            return None

    @classmethod
    def _historical_contract_from_symbol(
        cls, symbol: str, underlying_code: str, row: dict[str, str] | None = None
    ) -> Contract | None:
        if row:
            return cls._future_from_row(row, underlying_code) or cls._contract_from_row(row, underlying_code)
        match = re.fullmatch(
            r"(NSE|BSE)-(.+)-(\d{2}[A-Za-z]{3}\d{2})-(?:(\d+(?:\.\d+)?)-(CE|PE)|(FUT))",
            symbol,
        )
        if not match:
            return None
        exchange, _underlying, expiry_token, strike_token, option_token, future_token = match.groups()
        try:
            expiry = datetime.strptime(expiry_token, "%d%b%y").date()
            option_type = "FUT" if future_token else str(option_token)
            strike = Decimal(strike_token) if strike_token else Decimal("0")
        except (ValueError, ArithmeticError):
            return None
        # Groww's historical contracts endpoint documents symbols, not historical
        # lot/tick metadata. Preserve the candles but fail closed for option P&L
        # until a verified instrument-master version supplies the contract size.
        return Contract(
            instrument_key=symbol,
            trading_symbol=symbol,
            exchange=exchange,
            underlying_code=underlying_code,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            lot_size=0,
            tick_size=Decimal("0.05"),
            provider="GROWW",
            segment="FNO",
            metadata_verified=False,
            metadata_source="GROWW_HISTORICAL_SYMBOL",
        )

    async def contracts(self, underlying_key: str, underlying_code: str) -> list[Contract]:
        del underlying_key
        await self._load_instruments()
        _, symbol = GROWW_UNDERLYINGS[underlying_code]
        result: list[Contract] = []
        for row in self._instruments.values():
            if (
                row.get("segment") != "FNO"
                or row.get("underlying_symbol", "").upper() != symbol
                or row.get("buy_allowed", "1") not in {"", "1", "true", "True"}
            ):
                continue
            contract = self._contract_from_row(row, underlying_code)
            if contract and contract.expiry >= date.today():
                result.append(contract)
        return result

    async def future_contracts(self, underlying_code: str) -> list[Contract]:
        await self._load_instruments()
        _, symbol = GROWW_UNDERLYINGS[underlying_code]
        result: list[Contract] = []
        for row in self._instruments.values():
            if row.get("segment") != "FNO" or row.get("underlying_symbol", "").upper() != symbol:
                continue
            contract = self._future_from_row(row, underlying_code)
            if contract and contract.expiry >= date.today():
                result.append(contract)
        return sorted(result, key=lambda item: item.expiry)

    def _instrument_details(self, key: str) -> tuple[str, str, str, dict[str, str] | None]:
        if key in GROWW_INDEX_KEYS.values():
            exchange, symbol = key.split("-", 1)
            return exchange, "CASH", symbol, None
        row = self._instruments.get(key)
        if not row:
            # Historical symbols remain self-describing even when absent from today's CSV.
            parts = key.split("-")
            if len(parts) >= 3:
                return parts[0], "FNO", "-".join(parts[1:]), None
            raise GrowwError(f"Groww instrument metadata is unavailable for {key}")
        return row["exchange"], row.get("segment", "FNO"), row["trading_symbol"], row

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        if isinstance(value, (int, float)):
            scale = 1000 if value > 10_000_000_000 else 1
            return datetime.fromtimestamp(value / scale, tz=UTC)
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        return datetime.now(UTC)

    async def quotes(self, instrument_keys: list[str]) -> dict[str, Quote]:
        if any(key not in GROWW_INDEX_KEYS.values() for key in instrument_keys):
            await self._load_instruments()
        result: dict[str, Quote] = {}
        for key in instrument_keys:
            exchange, segment, symbol, row = self._instrument_details(key)
            payload = await self._json(
                "GET",
                "/v1/live-data/quote",
                params={"exchange": exchange, "segment": segment, "trading_symbol": symbol},
            )
            data = payload.get("payload") or {}
            depth = data.get("depth") or {}
            buys = depth.get("buy") or [{}]
            sells = depth.get("sell") or [{}]
            bid = self._decimal(data.get("bid_price") or buys[0].get("price"))
            ask = self._decimal(data.get("offer_price") or data.get("ask_price") or sells[0].get("price"))
            last = self._decimal(data.get("last_price") or data.get("ltp") or data.get("close"))
            if bid <= 0:
                bid = last
            if ask <= 0:
                ask = last
            delta = iv = None
            if segment == "FNO" and row:
                underlying = row.get("underlying_symbol")
                expiry = row.get("expiry_date")
                if underlying and expiry:
                    try:
                        greeks_payload = await self._json(
                            "GET",
                            f"/v1/live-data/greeks/exchange/{exchange}/underlying/{underlying}/trading_symbol/{symbol}/expiry/{expiry}",
                        )
                        greeks = (greeks_payload.get("payload") or {}).get("greeks") or {}
                        delta = float(greeks["delta"]) if greeks.get("delta") is not None else None
                        iv = float(greeks["iv"]) if greeks.get("iv") is not None else None
                    except GrowwError:
                        pass
            result[key] = Quote(
                instrument_key=key,
                timestamp=self._timestamp(data.get("last_trade_time") or data.get("timestamp")),
                bid=bid,
                ask=ask,
                last=last,
                volume=int(data.get("volume") or data.get("total_traded_quantity") or 0),
                open_interest=int(data.get("open_interest") or data.get("oi") or 0),
                delta=delta,
                iv=iv,
            )
        return result

    async def historical_candles(
        self, instrument_key: str, start: date, end: date, minutes: int = 5
    ) -> list[list[Any]]:
        if instrument_key not in GROWW_INDEX_KEYS.values():
            await self._load_instruments()
        exchange, segment, symbol, _ = self._instrument_details(instrument_key)
        groww_symbol = instrument_key if instrument_key.startswith(("NSE-", "BSE-")) else f"{exchange}-{symbol}"
        payload = await self._json(
            "GET",
            "/v1/historical/candles",
            params={
                "exchange": exchange,
                "segment": segment,
                "groww_symbol": groww_symbol,
                "start_time": datetime.combine(start, time(9, 15)).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.combine(end, time(15, 30)).strftime("%Y-%m-%d %H:%M:%S"),
                "candle_interval": f"{minutes}minute",
            },
        )
        return list((payload.get("payload") or {}).get("candles") or [])

    async def historical_expiries(self, underlying_code: str, start: date, end: date) -> list[date]:
        exchange, symbol = GROWW_UNDERLYINGS[underlying_code]
        cursor = date(start.year, start.month, 1)
        values: set[date] = set()
        while cursor <= end:
            payload = await self._json(
                "GET",
                "/v1/historical/expiries",
                params={
                    "exchange": exchange,
                    "underlying_symbol": symbol,
                    "year": cursor.year,
                    "month": cursor.month,
                },
            )
            for value in (payload.get("payload") or {}).get("expiries") or []:
                parsed = date.fromisoformat(str(value))
                if start <= parsed <= end:
                    values.add(parsed)
            cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
        return sorted(values)

    async def historical_contracts(self, underlying_code: str, expiry: date) -> list[Contract]:
        await self._load_instruments()
        exchange, symbol = GROWW_UNDERLYINGS[underlying_code]
        payload = await self._json(
            "GET",
            "/v1/historical/contracts",
            params={"exchange": exchange, "underlying_symbol": symbol, "expiry_date": expiry.isoformat()},
        )
        contract_symbols = set((payload.get("payload") or {}).get("contracts") or [])
        result: list[Contract] = []
        for groww_symbol in contract_symbols:
            row = self._instruments.get(str(groww_symbol))
            contract = self._historical_contract_from_symbol(str(groww_symbol), underlying_code, row)
            if contract and contract.option_type in {"CE", "PE"}:
                result.append(contract)
        return result

    async def historical_future_contracts(self, underlying_code: str, expiry: date) -> list[Contract]:
        await self._load_instruments()
        exchange, symbol = GROWW_UNDERLYINGS[underlying_code]
        payload = await self._json(
            "GET",
            "/v1/historical/contracts",
            params={"exchange": exchange, "underlying_symbol": symbol, "expiry_date": expiry.isoformat()},
        )
        contract_symbols = set((payload.get("payload") or {}).get("contracts") or [])
        result: list[Contract] = []
        for groww_symbol in contract_symbols:
            key = str(groww_symbol)
            contract = self._historical_contract_from_symbol(key, underlying_code, self._instruments.get(key))
            if contract and contract.option_type == "FUT":
                result.append(contract)
        return result

    async def historical_option_candles(
        self, contract: Contract, start: date, end: date, minutes: int = 5
    ) -> list[list[Any]]:
        return await self.historical_candles(contract.instrument_key, start, end, minutes)

    async def market_context(self, underlying_key: str, spot: Decimal) -> dict[str, float | None]:
        del underlying_key, spot
        # The current model is trained only on reproducible price/volume features.
        return {"india_vix": None, "futures_basis_pct": None, "pcr": None}

    async def place_market_order(self, *, instrument_key: str, side: str, quantity: int, tag: str) -> str:
        await self._load_instruments()
        exchange, segment, symbol, _ = self._instrument_details(instrument_key)
        reference = tag.replace("_", "-")[:20]
        if len(reference) < 8:
            reference = f"np-{reference}-order"[:20]
        payload = await self._json(
            "POST",
            "/v1/order/create",
            json={
                "trading_symbol": symbol,
                "quantity": quantity,
                "price": 0,
                "trigger_price": 0,
                "validity": "DAY",
                "exchange": exchange,
                "segment": segment,
                "product": "MIS",
                "order_type": "MARKET",
                "transaction_type": side,
                "order_reference_id": reference,
            },
        )
        order_id = (payload.get("payload") or {}).get("groww_order_id")
        if not order_id:
            raise GrowwError("Groww did not return an order ID")
        return str(order_id)

    @staticmethod
    def _order(row: dict[str, Any]) -> BrokerOrder:
        return BrokerOrder(
            order_id=str(row.get("groww_order_id", "")),
            status=str(row.get("order_status", "UNKNOWN")).upper(),
            quantity=int(row.get("quantity") or row.get("filled_quantity") or 0),
            filled_quantity=int(row.get("filled_quantity") or 0),
            average_price=Decimal(str(row.get("average_fill_price") or row.get("average_price") or 0)),
            raw=row,
        )

    async def order_history(self, order_id: str) -> list[BrokerOrder]:
        payload = await self._json("GET", f"/v1/order/detail/{order_id}", params={"segment": "FNO"})
        row = payload.get("payload") or {}
        return [self._order(row)] if row else []

    async def orders(self) -> list[BrokerOrder]:
        payload = await self._json(
            "GET", "/v1/order/list", params={"segment": "FNO", "page": 0, "page_size": 100}
        )
        return [self._order(row) for row in (payload.get("payload") or {}).get("order_list") or []]

    async def positions(self) -> dict[str, int]:
        await self._load_instruments()
        payload = await self._json("GET", "/v1/positions/user", params={"segment": "FNO"})
        rows = (payload.get("payload") or {}).get("positions") or payload.get("payload") or []
        if not isinstance(rows, list):
            return {}
        by_symbol = {row.get("trading_symbol"): key for key, row in self._instruments.items()}
        result: dict[str, int] = {}
        for row in rows:
            quantity = int(row.get("quantity") or row.get("net_quantity") or 0)
            key = row.get("groww_symbol") or by_symbol.get(row.get("trading_symbol"))
            if key and quantity:
                result[str(key)] = quantity
        return result
