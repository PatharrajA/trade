from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'niftypilot.db'}")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    upstox_analytics_token_file: Path = ROOT / os.getenv("UPSTOX_ANALYTICS_TOKEN_FILE", "data/secrets/upstox_analytics_token")
    upstox_trading_token_file: Path = ROOT / os.getenv("UPSTOX_TRADING_TOKEN_FILE", "data/secrets/upstox_trading_token")
    groww_token_file: Path = ROOT / os.getenv("GROWW_TOKEN_FILE", "data/secrets/groww_access_token")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_group_chat_id: str = os.getenv("TELEGRAM_GROUP_CHAT_ID", "")
    telegram_owner_chat_id: str = os.getenv("TELEGRAM_OWNER_CHAT_ID", "")
    telegram_enabled: bool = _bool("TELEGRAM_ENABLED", False)
    model_path: Path = ROOT / os.getenv("MODEL_PATH", "models/regime.json")
    trading_loop_enabled: bool = _bool("TRADING_LOOP_ENABLED", True)


config = Config()

