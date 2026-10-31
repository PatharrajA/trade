from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from app.shadow_quality import shadow_report


def test_shadow_gate_requires_each_selected_index_to_pass():
    start = datetime(2026, 1, 1, 4, tzinfo=UTC)
    rows = []
    for i in range(100):
        rows.append(SimpleNamespace(
            underlying_code="NIFTY", provider="GROWW", model_version="one",
            decision="TAKE", status="CLOSED",
            net_pnl=Decimal("120") if i % 5 else Decimal("-100"),
            direction="CE", moneyness="ATM", exit_reason="TARGET", probability=0.7,
            signal_time=start + timedelta(days=i // 5, minutes=i % 5 * 5),
        ))

    report = shadow_report(rows, ["NIFTY", "SENSEX"])

    assert report["by_index"]["NIFTY"]["issues"] == []
    assert report["by_index"]["NIFTY"]["take"]["sessions"] == 20
    assert any(issue.startswith("SENSEX:") for issue in report["issues"])


def test_shadow_report_excludes_rejected_observations_from_performance():
    row = SimpleNamespace(
        underlying_code="BANK_NIFTY", decision="TAKE", status="REJECTED",
        net_pnl=None, signal_time=datetime.now(UTC), direction="PE",
        moneyness=None, exit_reason=None, probability=0.5,
    )
    report = shadow_report([row], ["BANK_NIFTY"])

    assert report["take"]["count"] == 0
    assert report["by_index"]["BANK_NIFTY"]["rejected"] == 1
