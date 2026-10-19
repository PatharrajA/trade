from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select

from .config import config
from .database import settings_dict
from .models import NotificationOutbox


def enqueue(session, destination: str, event_type: str, message: str, dedupe_key: str) -> bool:
    if session.scalar(select(NotificationOutbox.id).where(NotificationOutbox.dedupe_key == dedupe_key)):
        return False
    session.add(NotificationOutbox(destination=destination, event_type=event_type, message=message, dedupe_key=dedupe_key))
    return True


class TelegramWorker:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.client = httpx.AsyncClient(timeout=10)
        self.stop_event = asyncio.Event()

    async def _send(self, item: NotificationOutbox, settings: dict[str, str]) -> None:
        group_id = settings.get("telegram_group_chat_id") or config.telegram_group_chat_id
        owner_id = settings.get("telegram_owner_chat_id") or config.telegram_owner_chat_id
        enabled = settings.get("telegram_enabled", str(config.telegram_enabled)).lower() == "true"
        chat_id = group_id if item.destination == "GROUP" else owner_id
        if not enabled or not config.telegram_bot_token or not chat_id:
            raise RuntimeError("Telegram destination is not configured")
        response = await self.client.post(f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage", json={"chat_id": chat_id, "text": item.message, "disable_web_page_preview": True})
        response.raise_for_status()
        if not response.json().get("ok"):
            raise RuntimeError("Telegram rejected the message")

    async def run(self) -> None:
        while not self.stop_event.is_set():
            now = datetime.now(UTC)
            with self.session_factory() as session:
                items = session.scalars(select(NotificationOutbox).where(NotificationOutbox.status == "PENDING", NotificationOutbox.next_attempt_at <= now).order_by(NotificationOutbox.id).limit(10)).all()
                for item in items:
                    item.status = "SENDING"
                    session.commit()
                    try:
                        await self._send(item, settings_dict(session))
                        item.status, item.sent_at, item.last_error = "SENT", datetime.now(UTC), None
                    except Exception as exc:
                        item.attempts += 1
                        item.last_error = str(exc)[:500]
                        item.status = "FAILED" if item.attempts >= 8 else "PENDING"
                        item.next_attempt_at = datetime.now(UTC) + timedelta(seconds=min(2**item.attempts, 300))
                    session.commit()
            await asyncio.sleep(2)

    def stop(self) -> None:
        self.stop_event.set()

