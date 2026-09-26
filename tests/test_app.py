"""Offline tests for the composition root: wiring, ownership, per-call setup, --now clock."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.app import offset_clock, open_runtime
from agent.llm import OpenAICompatibleLLMClient, Role, StreamEnd, TextDelta, ToolCall, ToolCallEvent
from agent.notifier import NotifyingSink
from agent.settings import Settings
from agent.storage import SqliteSink, StorageError

MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 26, 15, 0, tzinfo=MOSCOW)  # Saturday
REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "llm_base_url": "http://llm.test/v1",
        "llm_api_key": "key",
        "llm_model": "model",
        "telegram_bot_token": "123:abc",
        "telegram_chat_id": "1",
        "business_config_path": REPO_CONFIG,
        "db_path": tmp_path / "agent.db",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class ScriptedLLM:
    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream(self, messages, tools=None):
        self.calls.append(list(messages))
        for item in self._scripts.pop(0):
            yield item


class FakeNotifier:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


async def wait_until(condition, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.005)


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


def tool_round(call):
    return [ToolCallEvent(call), StreamEnd("tool_calls")]


# --- Wiring and ownership -------------------------------------------------------------------------


async def test_without_notify_the_tools_write_straight_to_the_database(tmp_path):
    notifier = FakeNotifier()
    async with open_runtime(
        make_settings(tmp_path), llm=ScriptedLLM(), notifier=notifier, clock=lambda: NOW
    ) as runtime:
        assert runtime.sink is runtime.store
        assert not isinstance(runtime.sink, NotifyingSink) and runtime.resent_at_start == 0

    assert notifier.sent == []


async def test_the_database_is_closed_on_exit(tmp_path):
    async with open_runtime(make_settings(tmp_path), llm=ScriptedLLM()) as runtime:
        store = runtime.store

    with pytest.raises(StorageError, match="closed"):
        await store.list_bookings()


async def test_the_db_path_argument_overrides_the_setting(tmp_path):
    other = tmp_path / "elsewhere" / "other.db"
    async with open_runtime(make_settings(tmp_path), llm=ScriptedLLM(), db_path=other):
        pass

    assert other.exists() and not (tmp_path / "agent.db").exists()


async def test_an_llm_client_is_built_from_the_settings_when_none_is_passed(tmp_path):
    async with open_runtime(make_settings(tmp_path, llm_reasoning_effort="none")) as runtime:
        assert isinstance(runtime.llm, OpenAICompatibleLLMClient)
        assert runtime.llm.model == "model" and runtime.llm.reasoning_effort == "none"
        assert runtime.llm.base_url == "http://llm.test/v1"


async def test_the_llm_extra_body_setting_reaches_the_client(tmp_path):
    settings = make_settings(tmp_path, llm_extra_body={"provider": {"sort": "latency"}})
    async with open_runtime(settings) as runtime:
        assert runtime.llm.extra_body == {"provider": {"sort": "latency"}}


async def test_notify_wires_the_notifying_sink_and_delivers_in_the_background(tmp_path):
    notifier = FakeNotifier()
    llm = ScriptedLLM(
        tool_round(ToolCall("c1", "prepare_booking", BOOKING_ARGS)),
        tool_round(ToolCall("c2", "confirm_booking", "{}")),
        [TextDelta("Заявка передана. Нужна ли помощь ещё?"), StreamEnd("stop")],
    )
    async with open_runtime(
        make_settings(tmp_path), notify=True, notifier=notifier, llm=llm, clock=lambda: NOW
    ) as runtime:
        assert isinstance(runtime.sink, NotifyingSink)
        session = runtime.new_call("+79991234567")
        [e async for e in session.handle("Меня зовут Игорь, номер 8 916 123 45 67")]
        [e async for e in session.handle("Да, всё верно")]

        await wait_until(lambda: notifier.sent)
        assert notifier.sent[0].startswith("Новая заявка №1")
        assert "Комментарий: после обеда" in notifier.sent[0]


async def test_notify_reports_how_many_older_records_it_resent(tmp_path):
    settings = make_settings(tmp_path)
    store = await SqliteSink.open(settings.db_path)
    from agent.records import CallbackMessage

    await store.add_message(CallbackMessage("старое", None, None, None, NOW - timedelta(days=1)))
    await store.close()
    notifier = FakeNotifier()

    async with open_runtime(settings, notify=True, notifier=notifier, llm=ScriptedLLM()) as runtime:
        assert runtime.resent_at_start == 1
        await wait_until(lambda: notifier.sent)
    assert "старое" in notifier.sent[0]


async def test_injected_llm_and_notifier_are_not_closed_by_the_runtime(tmp_path):
    class Tracking(FakeNotifier):
        closed = False

        async def aclose(self):
            self.closed = True

    notifier = Tracking()
    async with open_runtime(
        make_settings(tmp_path), notify=True, notifier=notifier, llm=ScriptedLLM()
    ):
        pass

    assert notifier.closed is False  # the caller owns it


# --- One call -------------------------------------------------------------------------------------


async def test_new_call_builds_the_prompt_from_the_clock_and_the_caller(tmp_path):
    llm = ScriptedLLM([TextDelta("Слушаю вас."), StreamEnd("stop")])
    async with open_runtime(make_settings(tmp_path), llm=llm, clock=lambda: NOW) as runtime:
        session = runtime.new_call("+79991234567")
        session.greet()
        [e async for e in session.handle("Привет")]

    system = llm.calls[0][0]
    assert system.role is Role.SYSTEM
    assert "Сейчас суббота, 26 сентября 2026, 15:00" in system.content
    assert "Номер звонящего: +79991234567." in system.content


async def test_a_call_without_caller_id_says_so_in_the_prompt(tmp_path):
    llm = ScriptedLLM([TextDelta("Слушаю вас."), StreamEnd("stop")])
    async with open_runtime(make_settings(tmp_path), llm=llm, clock=lambda: NOW) as runtime:
        [e async for e in runtime.new_call().handle("Привет")]

    assert "Номер звонящего: не определён." in llm.calls[0][0].content


async def test_each_call_is_independent(tmp_path):
    llm = ScriptedLLM(
        [TextDelta("Первый ответ."), StreamEnd("stop")],
        [TextDelta("Второй ответ."), StreamEnd("stop")],
    )
    async with open_runtime(make_settings(tmp_path), llm=llm, clock=lambda: NOW) as runtime:
        [e async for e in runtime.new_call().handle("звонок один")]
        [e async for e in runtime.new_call().handle("звонок два")]

    assert [m.content for m in llm.calls[1] if m.role is Role.USER] == ["звонок два"]


async def test_the_llm_stall_timeouts_come_from_the_settings(tmp_path):
    settings = make_settings(
        tmp_path, llm_first_event_timeout_seconds=1.5, llm_event_timeout_seconds=6.0
    )
    async with open_runtime(settings, llm=ScriptedLLM()) as runtime:
        engine = runtime.new_call()._engine

    assert (engine._first_event_timeout, engine._event_timeout) == (1.5, 6.0)


async def test_the_caller_id_and_clock_reach_the_saved_booking(tmp_path):
    llm = ScriptedLLM(
        tool_round(ToolCall("c1", "prepare_booking", BOOKING_ARGS)),
        tool_round(ToolCall("c2", "confirm_booking", "{}")),
        [TextDelta("Заявка передана."), StreamEnd("stop")],
    )
    async with open_runtime(make_settings(tmp_path), llm=llm, clock=lambda: NOW) as runtime:
        session = runtime.new_call("+79991234567")
        [e async for e in session.handle("Меня зовут Игорь, номер 8 916 123 45 67")]
        [e async for e in session.handle("Да")]
        [stored] = await runtime.store.list_bookings()

    assert stored.booking.caller_phone == "+79991234567"
    assert stored.booking.created_at == NOW and stored.booking.phone == "+79161234567"


# --- offset_clock (--now) -------------------------------------------------------------------------


def test_offset_clock_starts_at_the_requested_time_and_keeps_ticking():
    real = {"now": datetime(2030, 1, 1, 12, 0, tzinfo=MOSCOW)}
    clock = offset_clock(NOW, lambda: real["now"])

    assert clock() == NOW

    real["now"] += timedelta(minutes=7)
    assert clock() == NOW + timedelta(minutes=7)


async def test_the_speech_guard_setting_reaches_the_engine(tmp_path):
    async with open_runtime(make_settings(tmp_path), llm=ScriptedLLM()) as runtime:
        assert runtime.new_call()._engine.guard is not None
    async with open_runtime(make_settings(tmp_path, speech_guard=False), llm=ScriptedLLM()) as rt:
        assert rt.new_call()._engine.guard is None


async def test_each_call_gets_its_own_guard_state(tmp_path):
    async with open_runtime(make_settings(tmp_path), llm=ScriptedLLM()) as runtime:
        first, second = runtime.new_call()._engine, runtime.new_call()._engine
        first.guard.note_commit()
        assert first.guard.committed and not second.guard.committed
