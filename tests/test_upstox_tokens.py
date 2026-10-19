from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest

from app.upstox import UpstoxClient


def test_analytics_and_trading_credentials_are_isolated(tmp_path: Path):
    analytics = UpstoxClient(tmp_path / "analytics", credential_name="Analytics")
    trading = UpstoxClient(tmp_path / "trading", credential_name="Trading")
    analytics.save_token("a" * 40)
    trading.save_token("t" * 40)

    assert analytics._headers()["Authorization"] == f"Bearer {'a' * 40}"
    assert trading._headers()["Authorization"] == f"Bearer {'t' * 40}"
    assert (tmp_path / "analytics").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "trading").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_active_future_search_uses_documented_upstox_filters(tmp_path: Path):
    token_file = tmp_path / "analytics"
    token_file.write_text("a" * 40)
    expiry = date.today() + timedelta(days=20)

    def handler(request: httpx.Request):
        assert request.url.path == "/v2/instruments/search"
        assert request.url.params["query"] == "NIFTY"
        assert request.url.params["exchanges"] == "NSE"
        assert request.url.params["segments"] == "FO"
        assert request.url.params["instrument_types"] == "FUT"
        assert request.url.params["expiry"] == "current_month,next_month"
        assert request.url.params["records"] == "30"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": [
                    {
                        "instrument_key": "NSE_FO|12345",
                        "trading_symbol": "NIFTY FUT 29 OCT 26",
                        "exchange": "NSE",
                        "segment": "NSE_FO",
                        "expiry": expiry.isoformat(),
                        "lot_size": 65,
                        "minimum_lot": 65,
                        "tick_size": 5,
                        "exchange_token": "12345",
                    }
                ],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = UpstoxClient(token_file, client=http)
    try:
        contracts = await client.future_contracts("NIFTY")
        assert len(contracts) == 1
        assert contracts[0].expiry == expiry
        assert contracts[0].lot_size == 65
    finally:
        await http.aclose()
