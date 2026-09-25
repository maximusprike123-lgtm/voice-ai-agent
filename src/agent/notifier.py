"""Telegram notifications to the owner, delivered from a background task.

The conversation never waits for Telegram: `NotifyingSink` saves to SQLite (the source of truth),
returns the row id at once, and a single background worker sends the message and then marks the
row notified. Rows that are still unnotified (Telegram down, process killed) are re-sent on
startup and by a periodic sweep.

Delivery is AT-LEAST-ONCE: if the process dies between a successful send and the mark, the
message is sent again on the next start. A duplicate is acceptable; a lost booking is not.

Messages are plain text (no parse_mode), so whatever the caller said can never break them.
The bot token is part of Telegram's request URL, so exceptions raised here are built by us and
never chain to httpx's (which contain the URL), and httpx's own INFO log line for each request
is redacted.
"""

import asyncio
import contextlib
import logging
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import httpx

from agent.prompt import format_date_ru
from agent.records import Booking, CallbackMessage, RecordSink
from agent.storage import StoredBooking, StoredMessage

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org"
MAX_TEXT_LENGTH = 4096  # Telegram's limit for one message
TELEGRAM_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # everything but \n and \t
PERIOD_LABELS = {"утро": "утром", "день": "днём", "вечер": "вечером", "любое": "любое время"}


class NotifyError(Exception):
    """A message could not be delivered. `permanent` errors are not worth retrying."""

    def __init__(
        self, message: str, *, permanent: bool = False, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.permanent = permanent
        self.retry_after = retry_after  # seconds Telegram asked us to wait (HTTP 429)


class Notifier(Protocol):
    async def send(self, text: str) -> None:
        """Deliver one text message. Raises NotifyError."""
        ...


# --- Formatting -----------------------------------------------------------------------------------


def sanitize_text(text: str) -> str:
    """Drop control characters (keeping newlines and tabs) that Telegram may reject."""
    return _CONTROL_CHARS.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def _field(value: str) -> str:
    return sanitize_text(value).strip()


def format_booking(record_id: int, booking: Booking) -> str:
    lines = [
        f"Новая заявка №{record_id} (ждёт подтверждения администратором)",
        "",
        f"Имя: {_field(booking.name)}",
        f"Телефон: {booking.phone}",
    ]
    if booking.caller_phone and booking.caller_phone != booking.phone:
        lines.append(f"Звонил с номера: {booking.caller_phone}")
    lines.append(f"Автомобиль: {_field(booking.car)}")
    lines.append(f"Услуга: {_field(booking.service_name)}")
    lines.append(f"Дата: {format_date_ru(booking.preferred_date)}")
    if booking.preferred_time is not None:
        when = f"{booking.preferred_time:%H:%M}"
        if booking.preferred_period:
            when += f" (клиент сказал: {PERIOD_LABELS.get(booking.preferred_period, '?')})"
    elif booking.preferred_period:
        when = f"точное время не названо, {PERIOD_LABELS.get(booking.preferred_period, '?')}"
    else:
        when = "не указано"
    lines.append(f"Время: {when}")
    if booking.notes:
        lines.append(f"Комментарий: {_field(booking.notes)}")
    lines.append(f"Принято: {_format_created(booking.created_at)}")
    return "\n".join(lines)


def format_message(record_id: int, message: CallbackMessage) -> str:
    lines = [f"Сообщение для администратора №{record_id}", ""]
    if message.name:
        lines.append(f"Имя: {_field(message.name)}")
    if message.phone:
        lines.append(f"Телефон: {message.phone}")
    if message.caller_phone and message.caller_phone != message.phone:
        lines.append(f"Звонил с номера: {message.caller_phone}")
    lines.append(f"Сообщение: {_field(message.message)}")
    lines.append(f"Принято: {_format_created(message.created_at)}")
    return "\n".join(lines)


def format_record(record: StoredBooking | StoredMessage) -> str:
    if isinstance(record, StoredBooking):
        return format_booking(record.id, record.booking)
    return format_message(record.id, record.message)


def _format_created(created_at: datetime) -> str:
    return f"{created_at:%d.%m.%Y %H:%M}"


# --- Telegram -------------------------------------------------------------------------------------


class _RedactToken(logging.Filter):
    """httpx logs every request URL at INFO ("HTTP Request: POST https://api.telegram.org/
    bot<TOKEN>/sendMessage ..."), which would put the bot token into the logs."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._token in message:
            record.msg = message.replace(self._token, "<telegram-token>")
            record.args = None
        return True


def _redact_token_in_httpx_logs(token: str) -> None:
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(f, _RedactToken) and f._token == token for f in httpx_logger.filters):
        httpx_logger.addFilter(_RedactToken(token))


class TelegramNotifier:
    """Sends plain-text messages through the Telegram Bot API (`sendMessage`)."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        client: httpx.AsyncClient | None = None,
        api_url: str = TELEGRAM_API_URL,
    ) -> None:
        _redact_token_in_httpx_logs(token)
        self._url = f"{api_url}/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=TELEGRAM_TIMEOUT)

    def __repr__(self) -> str:  # never show the URL: it contains the token
        return f"TelegramNotifier(chat_id={self._chat_id!r})"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def send(self, text: str) -> None:
        text = sanitize_text(text)
        if len(text) > MAX_TEXT_LENGTH:
            text = text[: MAX_TEXT_LENGTH - 1] + "…"

        failure = None
        try:
            response = await self._client.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text},  # no parse_mode: plain text
                timeout=TELEGRAM_TIMEOUT,
            )
        except httpx.TimeoutException:
            failure = "Telegram request timed out"
        except httpx.HTTPError as exc:
            failure = f"could not reach Telegram ({type(exc).__name__})"
        if failure is not None:
            # Raised outside the except block on purpose: httpx's exception text contains the
            # request URL (and so the bot token), so it must not be chained as __context__.
            raise NotifyError(failure)

        status = response.status_code
        if status == 200:
            if _json(response).get("ok") is True:
                return
            raise NotifyError("Telegram answered 200 but not ok")
        description = str(_json(response).get("description", ""))[:200]
        if status == 429:
            raise NotifyError(
                "Telegram rate limit (429)", retry_after=_retry_after(response) or 1.0
            )
        if 400 <= status < 500:
            raise NotifyError(
                f"Telegram rejected the message ({status}): {description}", permanent=True
            )
        raise NotifyError(f"Telegram server error ({status}): {description}")


def _json(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _retry_after(response: httpx.Response) -> float | None:
    parameters = _json(response).get("parameters")
    value = parameters.get("retry_after") if isinstance(parameters, dict) else None
    if value is None:
        value = response.headers.get("Retry-After")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


# --- Background delivery --------------------------------------------------------------------------


class OutboxStore(RecordSink, Protocol):
    """A RecordSink that also remembers what has been notified (SqliteSink)."""

    async def list_unnotified(self) -> list[StoredBooking | StoredMessage]: ...

    async def mark_booking_notified(self, booking_id: int, at: datetime | None = None) -> None: ...

    async def mark_message_notified(self, message_id: int, at: datetime | None = None) -> None: ...


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5  # sends per record per round; the sweep tries again later
    base_delay: float = 1.0  # after attempt n: base * 2**(n-1), capped at max_delay
    max_delay: float = 30.0
    jitter: float = 0.25  # +-25% so retries don't march in lockstep
    max_retry_after: float = 120.0  # a longer 429 wait is left to the sweep
    retry_after_margin: float = 0.5

    def backoff(self, attempt: int) -> float:
        delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return delay * (1 + random.uniform(-self.jitter, self.jitter))

    def delay_after(self, error: NotifyError, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None to stop retrying for now."""
        if error.retry_after is not None:
            if error.retry_after > self.max_retry_after:
                return None
            return error.retry_after + self.retry_after_margin
        return self.backoff(attempt)


class NotifyingSink:
    """A RecordSink that saves first, replies at once, and notifies from a background task.

    `add_*` only waits for the database write. Call `start()` once (it resends everything
    still unnotified) and `aclose()` on shutdown. It does not own the store or the notifier.
    """

    def __init__(
        self,
        store: OutboxStore,
        notifier: Notifier,
        *,
        retry: RetryPolicy | None = None,
        sweep_interval: float = 300.0,
        shutdown_grace: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._store = store
        self._notifier = notifier
        self._retry = retry or RetryPolicy()
        self._sweep_interval = sweep_interval
        self._shutdown_grace = shutdown_grace
        self._sleep = sleep  # used for retry backoff only (injectable for tests)
        self._queue: asyncio.Queue[StoredBooking | StoredMessage] = asyncio.Queue()
        self._queued: set[tuple[str, int]] = set()
        self._worker: asyncio.Task | None = None
        self._sweeper: asyncio.Task | None = None
        self._closing = False

    # --- RecordSink ----------------------------------------------------------------------------

    async def add_booking(self, booking: Booking) -> int:
        record_id = await self._store.add_booking(booking)  # a save failure propagates
        self._submit(StoredBooking(record_id, booking, None))
        return record_id

    async def add_message(self, message: CallbackMessage) -> int:
        record_id = await self._store.add_message(message)
        self._submit(StoredMessage(record_id, message, None))
        return record_id

    # --- Lifecycle -------------------------------------------------------------------------------

    async def start(self) -> int:
        """Resend everything unnotified, then start the worker and the periodic sweep.

        Returns how many older records were queued for (re)sending.
        """
        if self._worker is not None:
            return 0
        self._closing = False
        queued = await self._enqueue_pending()
        self._worker = asyncio.create_task(self._work(), name="notifier-worker")
        self._sweeper = asyncio.create_task(self._sweep(), name="notifier-sweep")
        return queued

    async def aclose(self) -> None:
        """Give queued messages `shutdown_grace` seconds, then stop. Unsent rows stay in the
        database and go out on the next start."""
        self._closing = True
        if self._sweeper is not None:
            await _cancel(self._sweeper)
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._queue.join(), self._shutdown_grace)
            except TimeoutError:
                logger.warning(
                    "shutting down with undelivered notifications; they stay queued in the database"
                )
            await _cancel(self._worker)
        self._worker = self._sweeper = None

    # --- Internals --------------------------------------------------------------------------------

    def _submit(self, record: StoredBooking | StoredMessage) -> None:
        if self._closing:
            return  # it is in the database; the next start sends it
        key = (type(record).__name__, record.id)
        if key in self._queued:
            return
        self._queued.add(key)
        self._queue.put_nowait(record)

    async def _enqueue_pending(self) -> int:
        pending = await self._store.list_unnotified()
        for record in pending:
            self._submit(record)
        return len(pending)

    async def _sweep(self) -> None:
        while True:
            await asyncio.sleep(self._sweep_interval)
            try:
                await self._enqueue_pending()
            except Exception:
                logger.exception("could not list unnotified records")

    async def _work(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                await self._deliver(record)
            except Exception:
                logger.exception("unexpected error while notifying about %s", _label(record))
            finally:
                self._queued.discard((type(record).__name__, record.id))
                self._queue.task_done()

    async def _deliver(self, record: StoredBooking | StoredMessage) -> None:
        text = format_record(record)
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                await self._notifier.send(text)
                break
            except NotifyError as exc:
                error = exc
            except Exception as exc:  # a notifier bug must not kill the worker
                error = NotifyError(f"unexpected error ({type(exc).__name__})")

            if error.permanent:
                logger.error("cannot notify about %s, not retrying: %s", _label(record), error)
                return
            delay = self._retry.delay_after(error, attempt)
            if delay is None or attempt == self._retry.max_attempts:
                logger.warning(
                    "notification about %s failed (%s); it stays unnotified and the sweep "
                    "will retry",
                    _label(record),
                    error,
                )
                return
            logger.info(
                "notification about %s failed (%s); retry in %.1fs", _label(record), error, delay
            )
            await self._sleep(delay)

        try:
            if isinstance(record, StoredBooking):
                await self._store.mark_booking_notified(record.id)
            else:
                await self._store.mark_message_notified(record.id)
        except Exception:
            # Sent but not marked: it will be sent again later (at-least-once).
            logger.exception("sent %s but could not mark it notified", _label(record))


def _label(record: StoredBooking | StoredMessage) -> str:
    kind = "booking" if isinstance(record, StoredBooking) else "message"
    return f"{kind} #{record.id}"


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
