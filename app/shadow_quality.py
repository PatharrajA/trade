from __future__ import annotations

from collections import defaultdict
from datetime import UTC
from decimal import Decimal
from zoneinfo import ZoneInfo

from .models import ShadowObservation


def _ist(value):
    utc_value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return utc_value.astimezone(ZoneInfo("Asia/Kolkata"))


def _summary(rows: list[ShadowObservation]) -> dict:
    closed = [row for row in rows if row.status == "CLOSED" and row.net_pnl is not None]
    wins = sum((row.net_pnl for row in closed if row.net_pnl > 0), Decimal("0"))
    losses = sum((row.net_pnl for row in closed if row.net_pnl < 0), Decimal("0"))
    net = wins + losses
    return {
        "count": len(closed),
        "win_rate": round(100 * sum(row.net_pnl > 0 for row in closed) / len(closed), 2) if closed else 0.0,
        "profit_factor": round(float(wins / abs(losses)), 3) if losses else None,
        "net_pnl": float(net),
        "expectancy": round(float(net / len(closed)), 2) if closed else 0.0,
        "predicted_pct": round(100 * sum(getattr(row, "probability", 0.0) for row in closed) / len(closed), 1) if closed else 0.0,
        "sessions": len({_ist(row.signal_time).date() for row in closed}),
    }


def shadow_report(rows: list[ShadowObservation], indices: list[str]) -> dict:
    """Summarize one model version and provider, including rejected setups."""
    by_index = {}
    for code in indices:
        scoped = [row for row in rows if row.underlying_code == code]
        take = _summary([row for row in scoped if row.decision == "TAKE"])
        skip = _summary([row for row in scoped if row.decision == "SKIP"])
        rejected = sum(row.status == "REJECTED" for row in scoped)
        issues = []
        if take["count"] < 100:
            issues.append("fewer than 100 closed TAKE observations")
        if take["sessions"] < 20:
            issues.append("fewer than 20 trading sessions")
        if take["win_rate"] < 55:
            issues.append("TAKE win rate below 55%")
        if take["profit_factor"] is None or take["profit_factor"] < 1.20:
            issues.append("TAKE profit factor below 1.20")
        if take["expectancy"] <= 0:
            issues.append("TAKE expectancy not positive")
        if skip["profit_factor"] is not None and take["profit_factor"] is not None and take["profit_factor"] <= skip["profit_factor"]:
            issues.append("TAKE profit factor does not exceed SKIP")
        by_index[code] = {"take": take, "skip": skip, "rejected": rejected, "issues": issues}
    dimensions: dict[str, dict[str, dict]] = {}
    closed = [row for row in rows if row.status == "CLOSED" and row.net_pnl is not None]
    for name, key in (
        ("decision", lambda row: row.decision),
        ("direction", lambda row: row.direction),
        ("moneyness", lambda row: row.moneyness or "Unknown"),
        ("exit_reason", lambda row: row.exit_reason or "Unknown"),
        ("month", lambda row: _ist(row.signal_time).strftime("%Y-%m")),
        ("time", lambda row: (
            "Open" if _ist(row.signal_time).hour < 11
            else "Midday" if _ist(row.signal_time).hour < 13
            else "Afternoon"
        )),
        ("confidence", lambda row: f"{min(int(row.probability * 10) * 10, 90)}–{min(int(row.probability * 10) * 10 + 10, 100)}%"),
    ):
        groups: dict[str, list[ShadowObservation]] = defaultdict(list)
        for row in closed:
            groups[str(key(row))].append(row)
        dimensions[name] = {label: _summary(group) for label, group in sorted(groups.items())}
    return {
        "by_index": by_index,
        "dimensions": dimensions,
        "take": _summary([row for row in rows if row.decision == "TAKE"]),
        "skip": _summary([row for row in rows if row.decision == "SKIP"]),
        "issues": [f"{code}: {issue}" for code, result in by_index.items() for issue in result["issues"]],
    }
