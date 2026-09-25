"""Send ONE clearly marked test notification to the owner's Telegram chat.

Real network call with the TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env (never printed):

    .venv/bin/python scripts/send_test_notification.py

It runs the real path end to end: a temporary SQLite database, NotifyingSink, background
delivery, TelegramNotifier, and the "notified" mark. The sample booking contains characters
that would break HTML/Markdown formatting, to check they arrive verbatim in plain text.
The temporary database is deleted afterwards; nothing touches data/agent.db.
"""

import asyncio
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.notifier import NotifyingSink, TelegramNotifier  # noqa: E402
from agent.records import Booking  # noqa: E402
from agent.settings import get_settings  # noqa: E402
from agent.storage import SqliteSink  # noqa: E402

TIMEOUT_SECONDS = 20


async def main() -> int:
    settings = get_settings()
    notifier = TelegramNotifier(
        settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id
    )
    now = datetime.now(settings.tz)
    booking = Booking(
        name="ТЕСТ Тестович <b>&</b>",
        phone="+79990000000",
        car="Тестовая машина *_[x](y)_*",
        service_id="other",
        service_name="Другое / консультация",
        preferred_date=date.today(),
        preferred_time=None,
        preferred_period="день",
        notes="ЭТО ПРОВЕРКА УВЕДОМЛЕНИЙ: заявка не настоящая, отвечать не нужно.",
        caller_phone="+79991112233",
        created_at=now,
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = await SqliteSink.open(Path(tmp) / "test.db")
        sink = NotifyingSink(store, notifier, sweep_interval=3600)
        await sink.start()
        started = time.monotonic()
        try:
            record_id = await sink.add_booking(booking)
            saved_in = time.monotonic() - started
            print(
                f"saved as test record #{record_id}; add_booking returned in "
                f"{saved_in * 1000:.0f} ms"
            )

            while await store.list_unnotified():
                if time.monotonic() - started > TIMEOUT_SECONDS:
                    print(f"FAILED: not delivered within {TIMEOUT_SECONDS}s (see log lines above)")
                    return 1
                await asyncio.sleep(0.1)
            stored = await store.get_booking(record_id)
            delivered_in = time.monotonic() - started
            print(
                f"delivered and marked notified after {delivered_in:.2f}s "
                f"(notified_at set: {stored.notified_at is not None})"
            )
            return 0
        finally:
            await sink.aclose()
            await store.close()
            await notifier.aclose()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(name)s: %(message)s")
    raise SystemExit(asyncio.run(main()))
