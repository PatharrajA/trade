from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path


def file_permission_status(paths: list[Path]) -> tuple[bool, str]:
    configured = [path for path in paths if path.exists()]
    if not configured:
        return True, "No credential files stored"
    insecure = [path.name for path in configured if path.stat().st_mode & 0o077]
    return (not insecure, "0600 permissions verified" if not insecure else f"Permissions too broad: {', '.join(insecure)}")


def backup_status(root: Path) -> dict[str, object]:
    directory = root / "data" / "backups"
    backups = sorted(directory.glob("niftypilot-*.db"), key=lambda path: path.stat().st_mtime, reverse=True) if directory.exists() else []
    if not backups:
        return {"ok": False, "detail": "No verified backup found", "latest": None, "count": 0}
    latest = backups[0]
    stamp = datetime.fromtimestamp(latest.stat().st_mtime, tz=UTC)
    age_hours = (datetime.now(UTC) - stamp).total_seconds() / 3600
    return {
        "ok": age_hours <= 36,
        "detail": f"{latest.name} · {age_hours:.1f}h old",
        "latest": stamp,
        "count": len(backups),
    }
