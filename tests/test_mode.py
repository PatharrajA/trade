import pytest

from app.mode_control import change_mode
from app.models import RuntimeSetting


class FakeUpstox:
    def access_token(self):
        return "token"

    async def health(self):
        return True, "Connected"

    async def market_data_health(self):
        return True, "Market data connected"

    async def positions(self):
        return {}


@pytest.mark.asyncio
async def test_mode_persists_without_restart(session):
    ok, _ = await change_mode(session, FakeUpstox(), FakeUpstox(), "PAPER")
    assert ok
    assert session.get(RuntimeSetting, 1).execution_mode == "PAPER"


@pytest.mark.asyncio
async def test_live_requires_exact_confirmation(session):
    ok, message = await change_mode(session, FakeUpstox(), FakeUpstox(), "LIVE", "yes")
    assert not ok and "LIVE TRADING" in message
    ok, _ = await change_mode(session, FakeUpstox(), FakeUpstox(), "LIVE", "LIVE TRADING")
    assert ok


@pytest.mark.asyncio
async def test_live_requires_both_token_preflights(session):
    class MissingTrading(FakeUpstox):
        async def health(self):
            return False, "Trading token is missing"

    ok, message = await change_mode(session, FakeUpstox(), MissingTrading(), "LIVE", "LIVE TRADING")
    assert not ok
    assert "Trading-token preflight" in message


@pytest.mark.asyncio
async def test_paper_requires_analytics_but_not_trading_token(session):
    class MissingTrading(FakeUpstox):
        async def health(self):
            raise AssertionError("Paper mode checked the Trading Token")

    ok, _ = await change_mode(session, FakeUpstox(), MissingTrading(), "PAPER")
    assert ok


@pytest.mark.asyncio
async def test_live_safety_gate_blocks_unready_model_and_paper_soak(session):
    ok, message = await change_mode(
        session,
        FakeUpstox(),
        FakeUpstox(),
        "LIVE",
        "LIVE TRADING",
        model_available=False,
        paper_sessions=0,
        required_paper_sessions=20,
        static_ip_verified=True,
        feed_fresh=True,
    )
    assert not ok and "trained model" in message

    ok, message = await change_mode(
        session,
        FakeUpstox(),
        FakeUpstox(),
        "LIVE",
        "LIVE TRADING",
        model_available=True,
        paper_sessions=4,
        required_paper_sessions=20,
        static_ip_verified=True,
        feed_fresh=True,
    )
    assert not ok and "4 completed" in message


@pytest.mark.asyncio
async def test_live_can_skip_only_the_paper_session_gate(session):
    ok, message = await change_mode(
        session,
        FakeUpstox(),
        FakeUpstox(),
        "LIVE",
        "LIVE TRADING",
        model_available=True,
        paper_sessions=0,
        required_paper_sessions=20,
        enforce_paper_sessions=False,
        static_ip_verified=True,
        feed_fresh=True,
    )
    assert ok, message
    assert session.get(RuntimeSetting, 1).execution_mode == "LIVE"


@pytest.mark.asyncio
async def test_disabled_paper_gate_keeps_other_live_checks(session):
    ok, message = await change_mode(
        session,
        FakeUpstox(),
        FakeUpstox(),
        "LIVE",
        "LIVE TRADING",
        model_available=False,
        paper_sessions=0,
        required_paper_sessions=20,
        enforce_paper_sessions=False,
        static_ip_verified=True,
        feed_fresh=True,
    )
    assert not ok
    assert "trained model" in message
