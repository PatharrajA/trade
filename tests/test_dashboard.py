import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.dashboard import backtests_context, dashboard_context
from app.models import ModelArtifact, Setting, ShadowObservation, Signal, Trade


def _closed_trade(session, *, suffix: str, code: str, option_type: str, moneyness: str, net: str, gross: str, charges: str, minutes_ago: int):
    exit_time = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    signal_id = f"dashboard-{suffix}"
    session.add(
        Signal(
            id=signal_id,
            mode="PAPER",
            underlying_code=code,
            candle_timestamp=exit_time - timedelta(minutes=30),
            direction=option_type,
            regime="UPTREND" if option_type == "CE" else "DOWNTREND",
            confidence=0.8,
            status="APPROVED",
        )
    )
    session.add(
        Trade(
            signal_id=signal_id,
            mode="PAPER",
            underlying_code=code,
            instrument_key=f"TEST-{suffix}",
            trading_symbol=f"TEST {suffix}",
            option_type=option_type,
            moneyness=moneyness,
            expiry=date.today() + timedelta(days=7),
            quantity=50,
            lot_size=50,
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            underlying_entry=Decimal("25000"),
            underlying_stop=Decimal("24950"),
            underlying_target=Decimal("25075"),
            premium_emergency_stop=Decimal("75"),
            gross_pnl=Decimal(gross),
            charges=Decimal(charges),
            net_pnl=Decimal(net),
            status="CLOSED",
            exit_reason="TARGET" if Decimal(net) > 0 else "STOP",
            entry_time=exit_time - timedelta(minutes=30),
            exit_time=exit_time,
        )
    )


def test_dashboard_profit_risk_and_breakdowns(session):
    _closed_trade(
        session,
        suffix="win",
        code="NIFTY",
        option_type="CE",
        moneyness="ATM",
        net="1000",
        gross="1100",
        charges="100",
        minutes_ago=60,
    )
    _closed_trade(
        session,
        suffix="loss",
        code="BANK_NIFTY",
        option_type="PE",
        moneyness="ITM1",
        net="-400",
        gross="-350",
        charges="50",
        minutes_ago=30,
    )
    session.commit()

    context = dashboard_context(session)

    assert context["today_pnl"] == Decimal("600")
    assert context["month_gross_pnl"] == Decimal("750")
    assert context["month_charges"] == Decimal("150")
    assert context["profit_factor"] == 2.5
    assert context["expectancy"] == Decimal("300")
    assert context["maximum_drawdown"] == Decimal("400")
    assert context["option_breakdown"]["CE"]["pnl"] == Decimal("1000")
    assert context["moneyness_breakdown"]["ITM1"]["pnl"] == Decimal("-400")
    assert context["per_index"]["NIFTY"]["win_rate"] == 100


def test_backtest_context_compares_shadow_take_and_skip(session):
    session.add(ModelArtifact(
        version="setup-test", status="SHADOW", model_path="/tmp/setup-test.json",
        metadata_path="/tmp/setup-test.metadata.json",
        metadata_json=json.dumps({"version": "setup-test", "model_type": "SETUP_QUALITY", "indices": ["NIFTY"]}),
    ))
    session.get(Setting, "setup_shadow_model_path").value = "/tmp/setup-test.json"
    session.get(Setting, "active_broker").value = "GROWW"
    now = datetime.now(UTC)
    for index, (decision, pnl) in enumerate((("TAKE", "500"), ("TAKE", "-100"), ("SKIP", "-300"))):
        session.add(
            ShadowObservation(
                provider="GROWW",
                model_version="setup-test",
                underlying_code="NIFTY",
                signal_time=now - timedelta(minutes=index * 5),
                direction="CE",
                setup_type="OPENING_RANGE_BREAKOUT",
                probability=0.7,
                decision=decision,
                threshold=0.6,
                lot_size=65,
                underlying_entry=Decimal("25000"),
                underlying_stop=Decimal("24950"),
                underlying_target=Decimal("25075"),
                option_entry=Decimal("100"),
                option_exit=Decimal("110"),
                gross_pnl=Decimal(pnl),
                charges=Decimal("0"),
                net_pnl=Decimal(pnl),
                status="CLOSED",
                exit_reason="UNDERLYING_TARGET" if Decimal(pnl) > 0 else "UNDERLYING_STOP",
                timeout_at=now + timedelta(minutes=45),
                closed_at=now,
            )
        )
    session.commit()

    context = backtests_context(session)

    assert context["shadow_take"]["count"] == 2
    assert context["shadow_take"]["profit_factor"] == 5.0
    assert context["shadow_skip"]["count"] == 1
    assert context["shadow_skip"]["net_pnl"] == -300.0
