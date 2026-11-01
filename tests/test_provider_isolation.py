from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.backtest import _historical_contract
from app.models import Candle, Instrument


def test_historical_contract_selection_is_provider_scoped(session):
    trading_day = date(2026, 9, 1)
    expiry = trading_day + timedelta(days=7)
    for provider, key, lot_size in (
        ("UPSTOX", "NSE_FO|upstox-option", 75),
        ("GROWW", "NSE-NIFTY-08Sep26-25000-CE", 50),
    ):
        session.add(
            Instrument(
                provider=provider,
                instrument_key=key,
                underlying_code="NIFTY",
                exchange="NSE",
                trading_symbol=key,
                option_type="CE",
                strike=Decimal("25000"),
                expiry=expiry,
                lot_size=lot_size,
                tick_size=Decimal("0.05"),
                valid_from=trading_day,
                valid_to=expiry,
                active=False,
            )
        )
    session.commit()

    choices = _historical_contract(
        session, "GROWW", "NIFTY", "CE", trading_day, Decimal("25010")
    )
    assert len(choices) == 1
    assert choices[0][0].provider == "GROWW"
    assert choices[0][0].lot_size == 50


def test_historical_contract_rejects_distant_sensex_strike(session):
    trading_day = date(2026, 2, 20)
    session.add(Instrument(
        provider="GROWW", instrument_key="BSE-SENSEX-26Feb26-91100-CE",
        underlying_code="SENSEX", exchange="BSE", trading_symbol="SENSEX CE",
        option_type="CE", strike=Decimal("91100"), expiry=date(2026, 2, 26),
        lot_size=20, tick_size=Decimal("0.05"), valid_from=trading_day,
        valid_to=date(2026, 2, 26), active=False,
    ))
    session.commit()

    reasons = []
    assert _historical_contract(
        session, "GROWW", "SENSEX", "CE", trading_day, Decimal("83000"),
        rejection_reasons=reasons,
    ) == []
    assert reasons == ["NO_NEAR_ATM_CONTRACT"]


def test_same_external_key_can_be_stored_for_two_providers(session):
    stamp = datetime(2026, 9, 1, 3, 45, tzinfo=UTC)
    for provider in ("UPSTOX", "GROWW"):
        session.add(
            Candle(
                provider=provider,
                instrument_key="shared-test-key",
                underlying_code="NIFTY",
                timeframe="5minute",
                timestamp=stamp,
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100.5"),
                volume=100,
                open_interest=0,
                source_verified=True,
            )
        )
    session.commit()
    assert session.query(Candle).filter(Candle.instrument_key == "shared-test-key").count() == 2
