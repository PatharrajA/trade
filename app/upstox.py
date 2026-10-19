from __future__ import annotations

import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote as urlquote

import httpx

from .domain import BrokerOrder, Contract, Quote


class UpstoxError(RuntimeError):
    pass


class UpstoxClient:
    provider = "UPSTOX"
    api_url = "https://api.upstox.com"
    order_url = "https://api-hft.upstox.com"

    def __init__(self, token_file: Path, client: httpx.AsyncClient | None = None, *, credential_name: str = "Upstox") -> None:
        self.token_file = token_file
        self.client = client or httpx.AsyncClient(timeout=15)
        self.credential_name = credential_name

    def access_token(self) -> str:
        try:
            return self.token_file.read_text().strip()
        except FileNotFoundError:
            return ""

    def index_key(self, underlying_code: str) -> str:
        try:
            return INDEX_KEYS[underlying_code]
        except KeyError as exc:
            raise UpstoxError(f"Unsupported Upstox index: {underlying_code}") from exc

    def save_token(self, token: str) -> None:
        token = token.strip()
        if len(token) < 20:
            raise ValueError("Upstox token appears invalid")
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(token)
        os.chmod(self.token_file, 0o600)

    async def validate_token(self, token: str, *, market_data: bool) -> tuple[bool, str]:
        token = token.strip()
        if len(token) < 20:
            return False, "Token appears invalid"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"}
        try:
            if market_data:
                response = await self.client.get(
                    f"{self.api_url}/v3/market-quote/quotes",
                    headers=headers,
                    params={"instrument_key": "NSE_INDEX|Nifty 50"},
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") not in {None, "success"} or not payload.get("data"):
                    return False, "Analytics token did not return market data"
                return True, "Market-data access verified"
            response = await self.client.get(f"{self.api_url}/v2/user/profile", headers=headers)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") not in {None, "success"}:
                return False, "Trading token was rejected"
            return True, str(payload.get("data", {}).get("user_name", "Trading access verified"))
        except httpx.HTTPStatusError as exc:
            return False, f"Upstox rejected the token ({exc.response.status_code})"
        except Exception as exc:
            return False, str(exc)

    def _headers(self) -> dict[str, str]:
        token = self.access_token()
        if not token:
            raise UpstoxError(f"{self.credential_name} token is missing")
        return {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"}

    async def _json(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        response = await self.client.request(method, url, headers=self._headers(), **kwargs)
        if response.is_error:
            raise UpstoxError(f"Upstox request failed ({response.status_code})")
        payload = response.json()
        if payload.get("status") not in {None, "success"}:
            raise UpstoxError("Upstox returned an unsuccessful response")
        return payload

    async def health(self) -> tuple[bool, str]:
        try:
            payload = await self._json("GET", f"{self.api_url}/v2/user/profile")
            user = payload.get("data", {}).get("user_name", "Connected")
            return True, str(user)
        except Exception as exc:
            return False, str(exc)

    async def market_data_health(self) -> tuple[bool, str]:
        try:
            quotes = await self.quotes(["NSE_INDEX|Nifty 50"])
            quote = quotes.get("NSE_INDEX|Nifty 50")
            if not quote or quote.last <= 0:
                return False, "NIFTY quote was not returned"
            return True, f"NIFTY ₹{quote.last}"
        except Exception as exc:
            return False, str(exc)

    async def contracts(self, underlying_key: str, underlying_code: str) -> list[Contract]:
        payload = await self._json("GET", f"{self.api_url}/v2/option/contract", params={"instrument_key": underlying_key})
        return [
            Contract(
                instrument_key=row["instrument_key"],
                trading_symbol=row["trading_symbol"],
                exchange=row.get("exchange", "NSE"),
                underlying_code=underlying_code,
                expiry=date.fromisoformat(row["expiry"]),
                strike=Decimal(str(row["strike_price"])),
                option_type=row["instrument_type"],
                lot_size=int(row["lot_size"]),
                tick_size=Decimal(str(row.get("tick_size", 5))) / Decimal("100"),
                provider=self.provider,
                segment=str(row.get("segment", "NSE_FO")),
                exchange_token=str(row.get("exchange_token", "")) or None,
            )
            for row in payload.get("data", [])
        ]

    @staticmethod
    def _future_from_row(row: dict[str, Any], underlying_code: str) -> Contract | None:
        try:
            expiry = date.fromisoformat(str(row["expiry"]))
            return Contract(
                instrument_key=str(row["instrument_key"]),
                trading_symbol=str(row["trading_symbol"]),
                exchange=str(row.get("exchange", "BSE" if underlying_code == "SENSEX" else "NSE")),
                underlying_code=underlying_code,
                expiry=expiry,
                strike=Decimal("0"),
                option_type="FUT",
                lot_size=int(row.get("lot_size") or row.get("minimum_lot") or 0),
                tick_size=Decimal(str(row.get("tick_size", 5))) / Decimal("100"),
                provider="UPSTOX",
                segment=str(row.get("segment", "BSE_FO" if underlying_code == "SENSEX" else "NSE_FO")),
                exchange_token=str(row.get("exchange_token", "")) or None,
                metadata_verified=bool(row.get("lot_size") or row.get("minimum_lot")),
                metadata_source="UPSTOX_CONTRACT",
            )
        except (KeyError, ValueError, ArithmeticError):
            return None

    async def future_contracts(self, underlying_code: str) -> list[Contract]:
        symbol = {"NIFTY": "NIFTY", "BANK_NIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}[underlying_code]
        exchange = "BSE" if underlying_code == "SENSEX" else "NSE"
        payload = await self._json(
            "GET",
            f"{self.api_url}/v2/instruments/search",
            params={
                "query": symbol,
                "exchanges": exchange,
                "segments": "FO",
                "instrument_types": "FUT",
                "expiry": "current_month,next_month",
                "page_number": 1,
                "records": 30,
            },
        )
        result = [self._future_from_row(row, underlying_code) for row in payload.get("data", [])]
        return sorted(
            [item for item in result if item and item.expiry >= date.today()],
            key=lambda item: item.expiry,
        )

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        if isinstance(value, (int, float)):
            scale = 1000 if value > 10_000_000_000 else 1
            return datetime.fromtimestamp(value / scale, tz=UTC)
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        raise UpstoxError("Quote timestamp is unavailable")

    async def quotes(self, instrument_keys: list[str]) -> dict[str, Quote]:
        result: dict[str, Quote] = {}
        for start in range(0, len(instrument_keys), 500):
            keys = instrument_keys[start : start + 500]
            payload = await self._json("GET", f"{self.api_url}/v3/market-quote/quotes", params={"instrument_key": ",".join(keys)})
            for response_key, row in payload.get("data", {}).items():
                depth = row.get("market_depth") or row.get("depth") or {}
                buys = depth.get("bid") or depth.get("buy") or [{}]
                sells = depth.get("ask") or depth.get("sell") or [{}]
                ltpc = row.get("ltpc") or {}
                key = str(row.get("instrument_token") or row.get("instrument_key") or response_key)
                timestamp = self._timestamp(row.get("timestamp") or row.get("last_trade_time") or ltpc.get("ltt"))
                result[key] = Quote(
                    instrument_key=key,
                    timestamp=timestamp,
                    bid=Decimal(str(buys[0].get("price", 0))),
                    ask=Decimal(str(sells[0].get("price", 0))),
                    last=Decimal(str(ltpc.get("ltp", row.get("last_price", 0)))),
                    volume=int(row.get("volume", 0) or 0),
                    open_interest=int(row.get("oi", 0) or 0),
                )
        option_keys = [key for key in instrument_keys if "_FO|" in key][:50]
        if option_keys:
            greek_payload = await self._json("GET", f"{self.api_url}/v3/market-quote/option-greek", params={"instrument_key": ",".join(option_keys)})
            rows = greek_payload.get("data", {})
            iterable = rows.values() if isinstance(rows, dict) else rows
            for row in iterable:
                key = str(row.get("instrument_token") or row.get("instrument_key", ""))
                if key in result:
                    previous = result[key]
                    result[key] = Quote(**{**previous.__dict__, "delta": float(row["delta"]) if row.get("delta") is not None else None, "iv": float(row["iv"]) if row.get("iv") is not None else None})
        return result

    async def historical_candles(self, instrument_key: str, start: date, end: date, minutes: int = 5) -> list[list[Any]]:
        encoded = urlquote(instrument_key, safe="")
        payload = await self._json("GET", f"{self.api_url}/v3/historical-candle/{encoded}/minutes/{minutes}/{end}/{start}")
        return payload.get("data", {}).get("candles", [])

    async def historical_expiries(self, underlying_code: str, start: date, end: date) -> list[date]:
        payload = await self._json(
            "GET",
            f"{self.api_url}/v2/expired-instruments/expiries",
            params={"instrument_key": self.index_key(underlying_code)},
        )
        return sorted(
            parsed
            for value in payload.get("data", [])
            if start <= (parsed := date.fromisoformat(str(value))) <= end
        )

    async def historical_contracts(self, underlying_code: str, expiry: date) -> list[Contract]:
        payload = await self._json(
            "GET",
            f"{self.api_url}/v2/expired-instruments/option/contract",
            params={"instrument_key": self.index_key(underlying_code), "expiry_date": expiry.isoformat()},
        )
        contracts: list[Contract] = []
        for row in payload.get("data", []):
            contracts.append(
                Contract(
                    instrument_key=str(row["instrument_key"]),
                    trading_symbol=str(row["trading_symbol"]),
                    exchange=str(row.get("exchange", "NSE")),
                    underlying_code=underlying_code,
                    expiry=date.fromisoformat(str(row["expiry"])),
                    strike=Decimal(str(row["strike_price"])),
                    option_type=str(row["instrument_type"]),
                    lot_size=int(row["lot_size"]),
                    tick_size=Decimal(str(row.get("tick_size", 5))) / Decimal("100"),
                    provider=self.provider,
                    segment=str(row.get("segment", "NSE_FO")),
                    exchange_token=str(row.get("exchange_token", "")) or None,
                )
            )
        return contracts

    async def historical_future_contracts(self, underlying_code: str, expiry: date) -> list[Contract]:
        payload = await self._json(
            "GET",
            f"{self.api_url}/v2/expired-instruments/future/contract",
            params={"instrument_key": self.index_key(underlying_code), "expiry_date": expiry.isoformat()},
        )
        rows = payload.get("data", [])
        if isinstance(rows, dict):
            rows = [rows]
        result = [self._future_from_row(row, underlying_code) for row in rows]
        return [item for item in result if item]

    async def historical_option_candles(
        self, contract: Contract, start: date, end: date, minutes: int = 5
    ) -> list[list[Any]]:
        encoded = urlquote(contract.instrument_key, safe="")
        if contract.instrument_key.count("|") >= 2:
            payload = await self._json(
                "GET",
                f"{self.api_url}/v2/expired-instruments/historical-candle/{encoded}/{minutes}minute/{end}/{start}",
            )
            return payload.get("data", {}).get("candles", [])
        return await self.historical_candles(contract.instrument_key, start, end, minutes)

    async def market_context(self, underlying_key: str, spot: Decimal) -> dict[str, float | None]:
        contracts = await self.contracts(underlying_key, "CONTEXT")
        expiries = sorted({item.expiry for item in contracts if item.expiry > date.today()})
        pcr = None
        if expiries:
            chain = await self._json("GET", f"{self.api_url}/v2/option/chain", params={"instrument_key": underlying_key, "expiry_date": expiries[0].isoformat()})
            ce_oi = sum(int(((row.get("call_options") or {}).get("market_data") or {}).get("oi", 0) or 0) for row in chain.get("data", []))
            pe_oi = sum(int(((row.get("put_options") or {}).get("market_data") or {}).get("oi", 0) or 0) for row in chain.get("data", []))
            pcr = pe_oi / ce_oi if ce_oi else None
        vix_quote = (await self.quotes(["NSE_INDEX|India VIX"])).get("NSE_INDEX|India VIX")
        basis = None
        try:
            code = {
                "NSE_INDEX|Nifty 50": "NIFTY",
                "NSE_INDEX|Nifty Bank": "BANK_NIFTY",
                "BSE_INDEX|SENSEX": "SENSEX",
            }[underlying_key]
            futures = await self.future_contracts(code)
            future = min(futures, key=lambda item: item.expiry) if futures else None
            if future and spot:
                future_quote = (await self.quotes([future.instrument_key])).get(future.instrument_key)
                if future_quote: basis = float((future_quote.last - spot) / spot * 100)
        except Exception:
            basis = None
        return {"india_vix": float(vix_quote.last) if vix_quote else None, "futures_basis_pct": basis, "pcr": float(pcr) if pcr is not None else None}

    async def place_market_order(self, *, instrument_key: str, side: str, quantity: int, tag: str) -> str:
        body = {"quantity": quantity, "product": "I", "validity": "DAY", "price": 0, "tag": tag[:40], "instrument_token": instrument_key, "order_type": "MARKET", "transaction_type": side, "disclosed_quantity": 0, "trigger_price": 0, "is_amo": False, "slice": False, "market_protection": -1}
        payload = await self._json("POST", f"{self.order_url}/v3/order/place", json=body)
        data = payload.get("data", {})
        order_ids = data.get("order_ids") or []
        order_id = order_ids[0] if order_ids else data.get("order_id")
        if not order_id:
            raise UpstoxError("Upstox did not return an order ID")
        return str(order_id)

    @staticmethod
    def _order(row: dict[str, Any]) -> BrokerOrder:
        return BrokerOrder(str(row.get("order_id", "")), str(row.get("status", "UNKNOWN")).upper(), int(row.get("quantity", 0)), int(row.get("filled_quantity", 0)), Decimal(str(row.get("average_price", 0))), row)

    async def order_history(self, order_id: str) -> list[BrokerOrder]:
        payload = await self._json("GET", f"{self.api_url}/v2/order/history", params={"order_id": order_id})
        return [self._order(row) for row in payload.get("data", [])]

    async def orders(self) -> list[BrokerOrder]:
        payload = await self._json("GET", f"{self.api_url}/v2/order/retrieve-all")
        return [self._order(row) for row in payload.get("data", [])]

    async def positions(self) -> dict[str, int]:
        payload = await self._json("GET", f"{self.api_url}/v2/portfolio/short-term-positions")
        return {str(row.get("instrument_token")): int(row.get("quantity", 0)) for row in payload.get("data", []) if int(row.get("quantity", 0))}


INDEX_KEYS = {"NIFTY": "NSE_INDEX|Nifty 50", "BANK_NIFTY": "NSE_INDEX|Nifty Bank", "SENSEX": "BSE_INDEX|SENSEX"}

