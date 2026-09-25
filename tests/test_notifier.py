"""Offline tests for the Telegram notifier: formatting, HTTP handling, background delivery.

No network: httpx.MockTransport stands in for Telegram, a fake Notifier for the worker tests,
and an injected sleep keeps the backoff instant.
"""

import asyncio
import json
import logging
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from agent.business import load_business_config
from agent.llm import ToolCall
from agent.notifier import (
    MAX_TEXT_LENGTH,
    TELEGRAM_TIMEOUT,
    NotifyError,
    NotifyingSink,
    RetryPolicy,
    TelegramNotifier,
    format_booking,
    format_message,
    sanitize_text,
)
from agent.records import Booking, CallbackMessage
from agent.storage import SqliteSink, StoredBooking, StoredMessage
from agent.tools import ToolRegistry

MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 24, 17, 5, tzinfo=MOSCOW)
TOKEN = "123456:SECRET-TOKEN-abcdef"
CHAT_ID = "7077937946"
REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"


def make_booking(**overrides) -> Booking:
    fields = {
        "name": "Игорь",
        "phone": "+79161234567",
        "car": "Тойота Камри",
        "service_id": "polishing",
        "service_name": "Полировка кузова",
        "preferred_date": date(2026, 9, 26),
        "preferred_time": time(14, 30),
        "preferred_period": None,
        "notes": None,
        "caller_phone": "+79991234567",
        "created_at": NOW,
    }
    fields.update(overrides)
    return Booking(**fields)


def make_message(**overrides) -> CallbackMessage:
    fields = {
        "message": "Перезвоните мне",
        "name": "Анна",
        "phone": "+79161234567",
        "caller_phone": "+79991234567",
        "created_at": NOW,
    }
    fields.update(overrides)
    return CallbackMessage(**fields)


# --- Formatting -----------------------------------------------------------------------------------


def test_booking_message_has_every_field():
    text = format_booking(12, make_booking(notes="после обеда"))

    assert text.splitlines()[0] == "Новая заявка №12 (ждёт подтверждения администратором)"
    for line in (
        "Имя: Игорь",
        "Телефон: +79161234567",
        "Звонил с номера: +79991234567",
        "Автомобиль: Тойота Камри",
        "Услуга: Полировка кузова",
        "Дата: суббота, 26 сентября 2026",
        "Время: 14:30",
        "Комментарий: после обеда",
        "Принято: 24.09.2026 17:05",
    ):
        assert line in text.splitlines()


def test_caller_number_is_omitted_when_it_is_the_booking_phone_or_unknown():
    same = format_booking(1, make_booking(caller_phone="+79161234567"))
    unknown = format_booking(1, make_booking(caller_phone=None))
    assert "Звонил с" not in same and "Звонил с" not in unknown


def test_booking_with_only_a_period_says_the_exact_time_was_not_given():
    text = format_booking(1, make_booking(preferred_time=None, preferred_period="день"))
    assert "Время: точное время не названо, днём" in text.splitlines()


def test_booking_with_time_and_period_shows_both():
    text = format_booking(1, make_booking(preferred_period="вечер"))
    assert "Время: 14:30 (клиент сказал: вечером)" in text.splitlines()


def test_booking_without_notes_has_no_comment_line():
    assert "Комментарий" not in format_booking(1, make_booking())


def test_message_format_and_optional_lines():
    text = format_message(3, make_message())
    assert text.splitlines()[0] == "Сообщение для администратора №3"
    assert "Сообщение: Перезвоните мне" in text.splitlines()

    bare = format_message(4, make_message(name=None, phone=None, caller_phone=None))
    for absent in ("Имя", "Телефон", "Звонил"):
        assert absent not in bare


HOSTILE = "<b>жирный</b> & <a href='x'>ссылка</a> *звёзды* _подчёрк_ `код` [x](http://y) \\ ' \""


def test_hostile_caller_text_is_kept_verbatim_in_plain_text():
    text = format_booking(1, make_booking(name=HOSTILE, car=HOSTILE, notes=HOSTILE))
    assert text.count(HOSTILE) == 3  # nothing escaped, nothing stripped


def test_sanitize_removes_control_characters_but_keeps_newlines_and_tabs():
    assert sanitize_text("a\x00b\x07c\x1bd\x7fe\nf\tg") == "abcde\nf\tg"
    assert sanitize_text("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_control_characters_in_caller_text_do_not_reach_the_message():
    assert "\x00" not in format_booking(1, make_booking(name="Иг\x00орь", notes="a\x1bb"))


# --- TelegramNotifier: the HTTP call --------------------------------------------------------------


def make_notifier(handler) -> TelegramNotifier:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TelegramNotifier(TOKEN, CHAT_ID, client=client)


def ok_response() -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})


async def test_sends_a_plain_text_send_message_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return ok_response()

    await make_notifier(handler).send("Привет <b>мир</b> & co")

    assert captured["request"].method == "POST"
    assert captured["request"].url.path == f"/bot{TOKEN}/sendMessage"
    assert captured["request"].url.host == "api.telegram.org"
    assert captured["body"] == {"chat_id": CHAT_ID, "text": "Привет <b>мир</b> & co"}
    assert "parse_mode" not in captured["body"]


async def test_uses_short_timeouts_on_every_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions["timeout"]
        return ok_response()

    await make_notifier(handler).send("x")

    assert captured["timeout"] == {"connect": 3.0, "read": 5.0, "write": 5.0, "pool": 3.0}
    assert TELEGRAM_TIMEOUT.read == 5.0


async def test_long_text_is_truncated_to_the_telegram_limit():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["text"] = json.loads(request.content)["text"]
        return ok_response()

    await make_notifier(handler).send("я" * (MAX_TEXT_LENGTH + 500))

    assert len(captured["text"]) == MAX_TEXT_LENGTH and captured["text"].endswith("…")


async def test_control_characters_are_stripped_before_sending():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["text"] = json.loads(request.content)["text"]
        return ok_response()

    await make_notifier(handler).send("a\x00b")
    assert captured["text"] == "ab"


async def test_rate_limit_carries_telegrams_retry_after():
    def handler(request):
        return httpx.Response(
            429,
            json={
                "ok": False,
                "description": "Too Many Requests: retry after 7",
                "parameters": {"retry_after": 7},
            },
        )

    with pytest.raises(NotifyError) as info:
        await make_notifier(handler).send("x")

    assert info.value.retry_after == 7 and not info.value.permanent


async def test_rate_limit_falls_back_to_the_retry_after_header_then_to_one_second():
    def with_header(request):
        return httpx.Response(429, headers={"Retry-After": "3"}, json={"ok": False})

    def without(request):
        return httpx.Response(429, json={"ok": False})

    with pytest.raises(NotifyError) as a:
        await make_notifier(with_header).send("x")
    with pytest.raises(NotifyError) as b:
        await make_notifier(without).send("x")

    assert a.value.retry_after == 3 and b.value.retry_after == 1.0


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_server_errors_are_transient(status):
    def handler(request):
        return httpx.Response(status, json={"ok": False, "description": "oops"})

    with pytest.raises(NotifyError) as info:
        await make_notifier(handler).send("x")

    assert not info.value.permanent and info.value.retry_after is None


@pytest.mark.parametrize(
    ("status", "description"),
    [
        (400, "Bad Request: chat not found"),
        (401, "Unauthorized"),
        (403, "Forbidden: bot was blocked by the user"),
        (404, "Not Found"),
    ],
)
async def test_client_errors_are_permanent_and_explain_why(status, description):
    def handler(request):
        return httpx.Response(status, json={"ok": False, "description": description})

    with pytest.raises(NotifyError) as info:
        await make_notifier(handler).send("x")

    assert info.value.permanent and description in str(info.value)


async def test_timeouts_and_network_errors_are_transient():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    for handler in (timeout, refused):
        with pytest.raises(NotifyError) as info:
            await make_notifier(handler).send("x")
        assert not info.value.permanent


async def test_a_200_that_is_not_ok_or_not_json_is_a_transient_error():
    for response in (
        httpx.Response(200, json={"ok": False}),
        httpx.Response(200, content=b"<html>proxy</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
    ):
        with pytest.raises(NotifyError) as info:
            await make_notifier(lambda request, r=response: r).send("x")
        assert not info.value.permanent


async def test_the_token_never_appears_in_errors_reprs_or_logs(caplog):
    def refused(request):
        raise httpx.ConnectError(f"cannot connect to {request.url}", request=request)

    def server_error(request):
        return httpx.Response(500, json={"description": "boom"})

    caplog.set_level(logging.DEBUG)
    notifier = make_notifier(refused)
    errors = []
    for handler in (refused, server_error):
        with pytest.raises(NotifyError) as info:
            await make_notifier(handler).send("x")
        errors.append(info.value)

    for error in errors:
        assert TOKEN not in str(error) and TOKEN not in repr(error)
        assert error.__cause__ is None and error.__context__ is None  # no chained httpx error
    assert TOKEN not in repr(notifier) and "SECRET" not in repr(notifier)
    assert TOKEN not in caplog.text


async def test_httpx_request_log_line_is_redacted(caplog):
    """httpx logs the full URL at INFO; without the filter the token lands in the logs."""
    caplog.set_level(logging.INFO)

    await make_notifier(lambda request: ok_response()).send("x")

    request_lines = [r.getMessage() for r in caplog.records if r.name == "httpx"]
    assert request_lines, "expected httpx to log the request"
    assert all(TOKEN not in line for line in request_lines)
    assert any("<telegram-token>" in line for line in request_lines)
    assert TOKEN not in caplog.text


def test_the_redaction_filter_is_installed_once_per_token():
    from agent.notifier import _RedactToken

    TelegramNotifier(TOKEN, CHAT_ID)
    TelegramNotifier(TOKEN, CHAT_ID)

    filters = [f for f in logging.getLogger("httpx").filters if isinstance(f, _RedactToken)]
    assert len([f for f in filters if f._token == TOKEN]) == 1


# --- NotifyingSink: background delivery -----------------------------------------------------------


class FakeNotifier:
    """Plays back one outcome per send(): None = success, an exception to raise, or an
    asyncio.Event to wait for before succeeding. After the script runs out it succeeds."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.attempts: list[str] = []
        self.delivered: list[str] = []

    async def send(self, text: str) -> None:
        self.attempts.append(text)
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, asyncio.Event):
            await outcome.wait()
        elif outcome is not None:
            raise outcome
        self.delivered.append(text)


class Sleeps:
    """An injected sleep that records the requested delays and returns at once."""

    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        await asyncio.sleep(0)


NO_JITTER = RetryPolicy(jitter=0.0)


async def wait_until(condition, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.005)


@pytest.fixture
async def store(tmp_path):
    store = await SqliteSink.open(tmp_path / "agent.db")
    yield store
    await store.close()


async def started(store, notifier, **kwargs) -> NotifyingSink:
    kwargs.setdefault("sweep_interval", 3600)
    kwargs.setdefault("retry", NO_JITTER)
    kwargs.setdefault("sleep", Sleeps())
    sink = NotifyingSink(store, notifier, **kwargs)
    await sink.start()
    return sink


async def unnotified(store) -> list[int]:
    return [r.id for r in await store.list_unnotified()]


async def test_the_record_is_saved_before_it_is_notified_and_then_marked(store):
    notifier = FakeNotifier()
    sink = await started(store, notifier)

    booking_id = await sink.add_booking(make_booking())

    assert (await store.get_booking(booking_id)).booking == make_booking()  # saved at once
    await wait_until(lambda: notifier.delivered)
    await wait_until_marked(store)
    assert notifier.delivered == [format_booking(booking_id, make_booking())]
    await sink.aclose()


async def wait_until_marked(store, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while await store.list_unnotified():
        assert asyncio.get_running_loop().time() < deadline, "row never marked notified"
        await asyncio.sleep(0.005)


async def test_add_returns_immediately_even_while_telegram_hangs(store):
    gate = asyncio.Event()
    notifier = FakeNotifier(gate)  # the first send blocks until released
    sink = await started(store, notifier)

    booking_id = await asyncio.wait_for(sink.add_booking(make_booking()), 1.0)

    assert booking_id == 1
    await wait_until(lambda: notifier.attempts)  # the worker is stuck inside send()
    assert notifier.delivered == [] and await unnotified(store) == [1]
    # ...and the conversation can keep saving meanwhile
    assert await asyncio.wait_for(sink.add_message(make_message()), 1.0) == 1
    gate.set()
    await wait_until_marked(store)
    await sink.aclose()


async def test_messages_are_delivered_and_marked_too(store):
    notifier = FakeNotifier()
    sink = await started(store, notifier)

    message_id = await sink.add_message(make_message())

    await wait_until_marked(store)
    assert notifier.delivered == [format_message(message_id, make_message())]
    await sink.aclose()


async def test_messages_are_sent_in_the_order_they_were_saved(store):
    notifier = FakeNotifier()
    sink = await started(store, notifier)

    for i in range(5):
        await sink.add_booking(make_booking(car=f"car {i}"))

    await wait_until_marked(store)
    assert [t.split("Автомобиль: ")[1].split("\n")[0] for t in notifier.delivered] == [
        f"car {i}" for i in range(5)
    ]
    await sink.aclose()


async def test_a_save_failure_propagates_and_nothing_is_sent(tmp_path):
    store = await SqliteSink.open(tmp_path / "agent.db")
    notifier = FakeNotifier()
    sink = await started(store, notifier)
    await store.close()

    from agent.storage import StorageError

    with pytest.raises(StorageError):
        await sink.add_booking(make_booking())

    await asyncio.sleep(0.05)
    assert notifier.attempts == []
    await sink.aclose()


# --- Retries --------------------------------------------------------------------------------------


async def test_transient_failures_are_retried_with_exponential_backoff(store):
    boom = NotifyError("Telegram server error (502)")
    notifier = FakeNotifier(boom, boom, boom)
    sleeps = Sleeps()
    sink = await started(store, notifier, sleep=sleeps)

    await sink.add_booking(make_booking())

    await wait_until_marked(store)
    assert len(notifier.attempts) == 4 and len(notifier.delivered) == 1
    assert sleeps.delays == [1.0, 2.0, 4.0]
    await sink.aclose()


async def test_backoff_is_capped_and_jittered_within_bounds():
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=0.25)
    assert policy.delay_after(NotifyError("x"), 1) == pytest.approx(1.0, rel=0.25)
    assert policy.delay_after(NotifyError("x"), 4) == pytest.approx(8.0, rel=0.25)
    assert policy.delay_after(NotifyError("x"), 10) == pytest.approx(30.0, rel=0.25)  # capped
    assert len({policy.delay_after(NotifyError("x"), 3) for _ in range(20)}) > 1  # jitter


async def test_a_429_waits_for_retry_after_instead_of_the_backoff(store):
    notifier = FakeNotifier(NotifyError("rate limit", retry_after=7))
    sleeps = Sleeps()
    sink = await started(store, notifier, sleep=sleeps)

    await sink.add_booking(make_booking())

    await wait_until_marked(store)
    assert sleeps.delays == [7.5]  # retry_after + the safety margin
    await sink.aclose()


async def test_a_very_long_429_is_left_to_the_sweep_and_does_not_block_the_queue(store):
    notifier = FakeNotifier(NotifyError("rate limit", retry_after=3600))
    sleeps = Sleeps()
    sink = await started(store, notifier, sleep=sleeps)

    first = await sink.add_booking(make_booking(car="first"))
    second = await sink.add_booking(make_booking(car="second"))

    await wait_until(lambda: len(notifier.delivered) == 1)
    assert sleeps.delays == []  # it did not sit out an hour
    assert await unnotified(store) == [first]  # `second` went out, `first` waits for the sweep
    assert second == 2
    await sink.aclose()


async def test_permanent_errors_are_not_retried_and_the_row_stays_unnotified(store, caplog):
    notifier = FakeNotifier(NotifyError("Telegram rejected the message (403)", permanent=True))
    sleeps = Sleeps()
    sink = await started(store, notifier, sleep=sleeps)

    with caplog.at_level(logging.ERROR):
        await sink.add_booking(make_booking())
        await wait_until(lambda: "not retrying" in caplog.text)

    assert len(notifier.attempts) == 1 and sleeps.delays == []
    assert await unnotified(store) == [1]
    await sink.aclose()


async def test_it_gives_up_after_the_retry_budget_but_keeps_the_row(store):
    boom = NotifyError("Telegram server error (500)")
    notifier = FakeNotifier(*[boom] * 20)
    sleeps = Sleeps()
    sink = await started(store, notifier, sleep=sleeps)

    await sink.add_booking(make_booking())

    await wait_until(lambda: len(notifier.attempts) == 5)
    await asyncio.sleep(0.05)
    assert len(notifier.attempts) == 5  # max_attempts, no more
    assert sleeps.delays == [1.0, 2.0, 4.0, 8.0]
    assert await unnotified(store) == [1]
    await sink.aclose()


async def test_a_notifier_exception_never_reaches_add_and_the_worker_survives(store):
    notifier = FakeNotifier(RuntimeError("bug"), RuntimeError("bug"), RuntimeError("bug"))
    sink = await started(store, notifier, retry=RetryPolicy(max_attempts=2, jitter=0.0))

    first = await sink.add_booking(make_booking(car="unlucky"))
    assert first == 1  # no exception

    await wait_until(lambda: len(notifier.attempts) == 2)
    await sink.add_booking(make_booking(car="lucky"))
    await wait_until(lambda: len(notifier.delivered) == 1)
    assert "lucky" in notifier.delivered[0]
    await sink.aclose()


# --- Startup, sweep, at-least-once ----------------------------------------------------------------


async def test_start_resends_everything_still_unnotified_oldest_first(store):
    old_booking = await store.add_booking(make_booking(car="old", created_at=NOW))
    message = await store.add_message(make_message(created_at=NOW + timedelta(minutes=1)))
    newer_booking = await store.add_booking(
        make_booking(car="newer", created_at=NOW + timedelta(minutes=2))
    )
    already_sent = await store.add_booking(make_booking(car="sent", created_at=NOW))
    await store.mark_booking_notified(already_sent)
    notifier = FakeNotifier()

    sink = await started(store, notifier)

    await wait_until_marked(store)
    assert [t.splitlines()[0] for t in notifier.delivered] == [
        f"Новая заявка №{old_booking} (ждёт подтверждения администратором)",
        f"Сообщение для администратора №{message}",
        f"Новая заявка №{newer_booking} (ждёт подтверждения администратором)",
    ]
    await sink.aclose()


async def test_crash_between_send_and_mark_causes_a_resend_on_restart(tmp_path):
    """At-least-once: a duplicate is possible, a lost booking is not."""
    path = tmp_path / "agent.db"

    class DiesBeforeMarking(SqliteSink):
        async def mark_booking_notified(self, booking_id, at=None):
            raise RuntimeError("process killed")

    first_run = await SqliteSink.open(path)
    first_run.__class__ = DiesBeforeMarking
    notifier_1 = FakeNotifier()
    sink = await started(first_run, notifier_1)
    await sink.add_booking(make_booking())
    await wait_until(lambda: notifier_1.delivered)  # the message WAS sent...
    await sink.aclose()
    await first_run.close()

    second_run = await SqliteSink.open(path)  # ...but never marked
    notifier_2 = FakeNotifier()
    sink = await started(second_run, notifier_2)
    await wait_until_marked(second_run)

    assert notifier_2.delivered == notifier_1.delivered  # sent again: an accepted duplicate
    await sink.aclose()
    await second_run.close()


async def test_a_record_that_failed_all_retries_is_picked_up_by_the_periodic_sweep(store):
    boom = NotifyError("Telegram server error (500)")
    notifier = FakeNotifier(*[boom] * 5)  # the whole first round fails, then Telegram recovers
    sink = await started(store, notifier, sweep_interval=0.05)

    await sink.add_booking(make_booking())

    await wait_until_marked(store)
    assert len(notifier.attempts) == 6 and len(notifier.delivered) == 1
    await sink.aclose()


async def test_the_sweep_does_not_queue_a_record_that_is_already_in_flight(store):
    gate = asyncio.Event()
    notifier = FakeNotifier(gate)
    sink = await started(store, notifier, sweep_interval=0.02)

    await sink.add_booking(make_booking())
    await wait_until(lambda: notifier.attempts)
    await asyncio.sleep(0.15)  # several sweeps run while the first send is stuck
    gate.set()
    await wait_until_marked(store)

    assert len(notifier.attempts) == 1  # never duplicated
    await sink.aclose()


async def test_marking_failure_is_logged_and_does_not_stop_the_worker(store, caplog):
    class BrokenMark(SqliteSink):
        async def mark_booking_notified(self, booking_id, at=None):
            raise RuntimeError("disk full")

    store.__class__ = BrokenMark
    notifier = FakeNotifier()
    sink = await started(store, notifier)

    with caplog.at_level(logging.ERROR):
        await sink.add_booking(make_booking(car="a"))
        await sink.add_message(make_message())
        await wait_until(lambda: len(notifier.delivered) == 2)  # the worker carried on

    assert "could not mark it notified" in caplog.text
    await sink.aclose()


# --- Shutdown -------------------------------------------------------------------------------------


async def test_aclose_drains_the_queue_before_stopping(store):
    notifier = FakeNotifier()
    sink = await started(store, notifier)
    for i in range(5):
        await sink.add_booking(make_booking(car=f"car {i}"))

    await sink.aclose()

    assert len(notifier.delivered) == 5 and await unnotified(store) == []


async def test_aclose_gives_up_on_a_stuck_send_and_leaves_the_row_for_the_next_start(store):
    notifier = FakeNotifier(asyncio.Event())  # never released
    sink = await started(store, notifier, shutdown_grace=0.1)
    await sink.add_booking(make_booking())
    await wait_until(lambda: notifier.attempts)

    await asyncio.wait_for(sink.aclose(), 2.0)  # returns instead of hanging

    assert await unnotified(store) == [1]


async def test_records_added_after_close_are_saved_but_not_sent(store):
    notifier = FakeNotifier()
    sink = await started(store, notifier)
    await sink.aclose()

    booking_id = await sink.add_booking(make_booking())

    await asyncio.sleep(0.05)
    assert notifier.attempts == [] and await unnotified(store) == [booking_id]


async def test_start_and_aclose_are_idempotent(store):
    sink = await started(store, FakeNotifier())
    await sink.start()  # second start does nothing
    await sink.aclose()
    await sink.aclose()


# --- With the tools -------------------------------------------------------------------------------


BOOKING_ARGS = json.dumps(
    {
        "name": "Игорь",
        "phone": "8 916 123 45 67",
        "car": "Тойота Камри",
        "service_id": "polishing",
        "preferred_date": "2026-09-26",
        "preferred_period": "день",
        "notes": "после обеда",
    },
    ensure_ascii=False,
)


async def test_confirm_booking_returns_at_once_while_telegram_is_down(store):
    gate = asyncio.Event()
    notifier = FakeNotifier(gate)
    sink = await started(store, notifier)
    tools = ToolRegistry(
        load_business_config(REPO_CONFIG), sink, clock=lambda: NOW, caller_phone="+79991234567"
    )
    await tools.execute(ToolCall("c1", "prepare_booking", BOOKING_ARGS))

    outcome = await asyncio.wait_for(tools.execute(ToolCall("c2", "confirm_booking", "{}")), 1.0)

    assert not outcome.result.startswith("ОШИБКА")
    assert len(await store.list_bookings()) == 1  # saved
    await wait_until(lambda: notifier.attempts)  # the worker is trying, stuck in send()
    assert notifier.delivered == []  # ...and the caller was already answered
    gate.set()
    await wait_until_marked(store)
    text = notifier.delivered[0]
    assert "Время: точное время не названо, днём" in text and "Комментарий: после обеда" in text
    await sink.aclose()


async def test_stored_record_types_round_trip_through_the_queue_keys(store):
    sink = await started(store, FakeNotifier())
    booking_id = await sink.add_booking(make_booking())
    message_id = await sink.add_message(make_message())
    assert booking_id == message_id == 1  # same id, different tables: both must be delivered
    await wait_until_marked(store)
    assert isinstance((await store.list_bookings())[0], StoredBooking)
    assert isinstance((await store.list_messages())[0], StoredMessage)
    await sink.aclose()


def test_utc_notified_timestamps_are_timezone_aware():
    assert datetime.now(UTC).tzinfo is not None


async def test_start_reports_how_many_older_records_it_queued(store):
    await store.add_booking(make_booking(car="a"))
    await store.add_message(make_message())
    sink = NotifyingSink(store, FakeNotifier(), sweep_interval=3600)

    assert await sink.start() == 2
    assert await sink.start() == 0  # already running
    await sink.aclose()
