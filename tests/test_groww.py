from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest

from app.groww import GrowwClient


def _client(tmp_path, handler):
    token_file = tmp_path / "groww-token"
    token_file.write_text("test-access-token-with-enough-characters")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GrowwClient(token_file, client=http)


@pytest.mark.asyncio
async def test_groww_index_quote_uses_official_live_quote_shape(tmp_path):
    requested = []

    def handler(request: httpx.Request):
        requested.append(request.url.path)
        assert request.url.path == "/v1/live-data/quote"
        assert request.url.params["exchange"] == "NSE"
        assert request.url.params["segment"] == "CASH"
        assert request.url.params["trading_symbol"] == "NIFTY"
        return httpx.Response(
            200,
            json={
                "status": "SUCCESS",
                "payload": {
                    "bid_price": 25000,
                    "offer_price": 25000.5,
                    "last_price": 25000.25,
                    "last_trade_time": 1_797_000_000_000,
                    "volume": 100,
                    "open_interest": 0,
                },
            },
        )

    client = _client(tmp_path, handler)
    try:
        quote = (await client.quotes([client.index_key("NIFTY")]))["NSE-NIFTY"]
        assert quote.bid == Decimal("25000")
        assert quote.ask == Decimal("25000.5")
        assert quote.timestamp.tzinfo == UTC
        assert requested == ["/v1/live-data/quote"]
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
async def test_groww_contract_master_and_greeks_are_normalized(tmp_path):
    expiry = date(2026, 10, 29)
    symbol = "NSE-NIFTY-29Oct26-25000-CE"
    csv_body = (
        "exchange,exchange_token,trading_symbol,groww_symbol,instrument_type,segment,"
        "underlying_symbol,expiry_date,strike_price,lot_size,tick_size,buy_allowed\n"
        f"NSE,123,NIFTY26O2925000CE,{symbol},CE,FNO,NIFTY,{expiry},25000,75,0.05,1\n"
    )

    def handler(request: httpx.Request):
        if request.url.host == "growwapi-assets.groww.in":
            return httpx.Response(200, text=csv_body)
        if request.url.path == "/v1/live-data/quote":
            return httpx.Response(
                200,
                json={
                    "status": "SUCCESS",
                    "payload": {
                        "bid_price": 120,
                        "offer_price": 121,
                        "last_price": 120.5,
                        "last_trade_time": 1_797_000_000_000,
                        "volume": 1000,
                        "open_interest": 5000,
                    },
                },
            )
        if request.url.path.startswith("/v1/live-data/greeks/"):
            return httpx.Response(
                200,
                json={"status": "SUCCESS", "payload": {"greeks": {"delta": 0.52, "iv": 14.3}}},
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    client = _client(tmp_path, handler)
    try:
        contracts = await client.contracts(client.index_key("NIFTY"), "NIFTY")
        assert len(contracts) == 1
        assert contracts[0].lot_size == 75
        assert contracts[0].provider == "GROWW"
        quote = (await client.quotes([symbol]))[symbol]
        assert quote.delta == pytest.approx(0.52)
        assert quote.iv == pytest.approx(14.3)
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
async def test_groww_historical_and_order_responses_are_normalized(tmp_path):
    def handler(request: httpx.Request):
        if request.url.path == "/v1/historical/candles":
            return httpx.Response(
                200,
                json={
                    "status": "SUCCESS",
                    "payload": {
                        "candles": [["2026-09-01T09:15:00", 100, 102, 99, 101, 1000, None]]
                    },
                },
            )
        if request.url.path == "/v1/order/detail/order-1":
            return httpx.Response(
                200,
                json={
                    "status": "SUCCESS",
                    "payload": {
                        "groww_order_id": "order-1",
                        "order_status": "EXECUTED",
                        "quantity": 75,
                        "filled_quantity": 75,
                        "average_fill_price": 120.25,
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    client = _client(tmp_path, handler)
    try:
        rows = await client.historical_candles(
            client.index_key("NIFTY"), date(2026, 9, 1), date(2026, 9, 1)
        )
        assert rows[0][4] == 101
        order = (await client.order_history("order-1"))[0]
        assert order.status == "EXECUTED"
        assert order.filled_quantity == 75
        assert order.average_price == Decimal("120.25")
    finally:
        await client.client.aclose()


def test_groww_naive_timestamps_are_interpreted_as_india_time():
    stamp = GrowwClient._timestamp("2026-10-12T09:15:00")
    assert stamp.utcoffset().total_seconds() == 19_800
    assert stamp.astimezone(UTC) == datetime(2026, 10, 12, 3, 45, tzinfo=UTC)


def test_groww_expired_option_symbol_is_preserved_but_unverified():
    contract = GrowwClient._historical_contract_from_symbol(
        "NSE-NIFTY-16Jul26-22500-CE", "NIFTY"
    )

    assert contract is not None
    assert contract.expiry == date(2026, 7, 16)
    assert contract.strike == Decimal("22500")
    assert contract.option_type == "CE"
    assert contract.lot_size == 0
    assert contract.metadata_verified is False


def test_groww_expired_future_symbol_is_normalized():
    contract = GrowwClient._historical_contract_from_symbol(
        "NSE-BANKNIFTY-30Jul26-FUT", "BANK_NIFTY"
    )

    assert contract is not None
    assert contract.expiry == date(2026, 7, 30)
    assert contract.option_type == "FUT"
    assert contract.metadata_source == "GROWW_HISTORICAL_SYMBOL"
