"""Offline tests for CallSession: greeting, pass-through, and the LLM-failure policy."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.dialogue import DialogueEngine, EndCall, Say, ToolOutcome
from agent.llm import LLMError, Message, Role, StreamEnd, TextDelta, ToolCall, ToolCallEvent
from agent.records import InMemorySink
from agent.session import (
    ASK_TO_REPEAT,
    FINAL_APOLOGY,
    MAX_UTTERANCE_CHARS,
    CallSession,
    TurnFailed,
)

NOW = datetime(2026, 9, 24, 17, 5, tzinfo=ZoneInfo("Europe/Moscow"))
GREETING = "Здравствуйте! Детейлинг-центр «Пример». Чем могу помочь?"


class ScriptedLLM:
    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream(self, messages, tools=None):
        self.calls.append(list(messages))
        script = self._scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        for item in script:
            yield item


class NoTools:
    specs = []

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}

    def begin_turn(self):
        pass

    async def execute(self, call):
        return self.outcomes.get(call.name, ToolOutcome("ok"))


def text(*chunks):
    return [*(TextDelta(c) for c in chunks), StreamEnd("stop")]


def failing_round():
    """One failed LLM round. The engine retries a round once, so a failed TURN needs two."""
    return LLMError("backend down")


def make_session(*scripts, sink=None, caller_phone="+79991234567", tools=None, **kwargs):
    llm = ScriptedLLM(*scripts)
    engine = DialogueEngine(llm, tools or NoTools(), "SYS", first_event_timeout=0.05)
    sink = sink if sink is not None else InMemorySink()
    session = CallSession(engine, GREETING, sink, lambda: NOW, caller_phone, **kwargs)
    return session, llm, sink, engine


async def turn(session, user_text):
    return [event async for event in session.handle(user_text)]


# --- Greeting -------------------------------------------------------------------------------------


def test_greeting_is_returned_as_sentences_and_recorded_in_the_history():
    session, _, _, engine = make_session()

    assert session.greet() == [
        Say("Здравствуйте!"),
        Say("Детейлинг-центр «Пример»."),
        Say("Чем могу помочь?"),
    ]
    assert engine.messages[-1] == Message(Role.ASSISTANT, GREETING)


async def test_the_model_sees_the_greeting_on_the_first_turn():
    session, llm, _, _ = make_session(text("Слушаю вас."))
    session.greet()

    await turn(session, "Хочу записаться")

    assert [m.content for m in llm.calls[0]] == ["SYS", GREETING, "Хочу записаться"]


# --- Normal turns ---------------------------------------------------------------------------------


async def test_normal_turns_pass_engine_events_straight_through():
    session, _, _, _ = make_session(text("Слушаю вас."))

    assert await turn(session, "Привет") == [Say("Слушаю вас.")]
    assert not session.ended


async def test_end_call_from_the_engine_ends_the_session():
    call = ToolCall("c1", "end_call", "{}")
    session, _, _, _ = make_session(
        [ToolCallEvent(call), StreamEnd("tool_calls")],
        tools=NoTools({"end_call": ToolOutcome("ok", ends_call=True)}),
    )

    events = await turn(session, "До свидания")

    assert isinstance(events[-1], EndCall) and session.ended


# --- First failure: ask to repeat -----------------------------------------------------------------


async def test_first_failed_turn_asks_the_caller_to_repeat_and_forgets_the_attempt():
    session, llm, sink, engine = make_session(
        failing_round(), failing_round(), text("Стоимость от трёх тысяч рублей.")
    )
    session.greet()

    events = await turn(session, "Сколько стоит мойка?")

    assert [type(e) for e in events] == [TurnFailed, Say]
    assert events[0] == TurnFailed("backend down", 1)
    assert events[1] == Say(ASK_TO_REPEAT)
    assert not session.ended and sink.messages == []
    assert [m.role for m in engine.messages] == [Role.SYSTEM, Role.ASSISTANT]  # rolled back

    # ...so the caller just says it again, and the conversation carries on normally:
    assert await turn(session, "Сколько стоит мойка?") == [Say("Стоимость от трёх тысяч рублей.")]
    assert [m.content for m in llm.calls[-1]].count("Сколько стоит мойка?") == 1


async def test_an_empty_reply_counts_as_a_failed_turn():
    session, _, _, _ = make_session([StreamEnd("stop")], [StreamEnd("stop")])

    events = await turn(session, "Привет")

    assert isinstance(events[0], TurnFailed) and "empty reply" in events[0].reason
    assert events[-1] == Say(ASK_TO_REPEAT)


async def test_the_engines_own_retry_hides_a_single_hiccup_from_the_caller():
    session, llm, _, _ = make_session(failing_round(), text("Слушаю вас."))

    assert await turn(session, "Привет") == [Say("Слушаю вас.")]
    assert len(llm.calls) == 2


async def test_speech_that_was_already_heard_is_not_repeated_before_the_fallback():
    class DropsAfterFirstSentence(ScriptedLLM):
        async def stream(self, messages, tools=None):
            self.calls.append(list(messages))
            yield TextDelta("Стоимость от трёх тысяч. Точную")
            raise LLMError("connection dropped")

    llm = DropsAfterFirstSentence()
    engine = DialogueEngine(llm, NoTools(), "SYS")
    session = CallSession(engine, GREETING, InMemorySink(), lambda: NOW)

    events = await turn(session, "Сколько стоит?")

    assert events[0] == Say("Стоимость от трёх тысяч.")
    assert isinstance(events[1], TurnFailed) and events[2] == Say(ASK_TO_REPEAT)
    assert len(llm.calls) == 1  # no retry after speech started


async def test_a_looping_model_is_a_failed_turn_too(monkeypatch):
    import agent.dialogue as dialogue

    monkeypatch.setattr(dialogue, "MAX_LLM_ROUNDS", 2)
    call = ToolCall("c1", "take_message", "{}")
    tool_round = [ToolCallEvent(call), StreamEnd("tool_calls")]
    session, _, _, _ = make_session(tool_round, tool_round)

    events = await turn(session, "x")

    assert any(isinstance(e, TurnFailed) and "tools" in e.reason for e in events)
    assert events[-1] == Say(ASK_TO_REPEAT)


# --- Second failure in a row: apologise, save a callback, hang up ---------------------------------


async def test_second_failed_turn_in_a_row_apologises_saves_a_callback_and_ends_the_call():
    session, _, sink, _ = make_session(*[failing_round()] * 4)

    first = await turn(session, "Хочу записаться на полировку")
    second = await turn(session, "Алло, вы меня слышите?")

    assert first[-1] == Say(ASK_TO_REPEAT)
    assert second[0] == TurnFailed("backend down", 2)
    assert [e for e in second if isinstance(e, Say)] == [
        Say("Извините, у нас возникли технические неполадки."),
        Say("Администратор перезвонит вам."),
        Say("До свидания."),
    ]
    assert second[-1] == EndCall() and session.ended
    assert " ".join(e.text for e in second if isinstance(e, Say)) == FINAL_APOLOGY

    [message] = sink.messages
    assert "«Хочу записаться на полировку»" in message.message
    assert "«Алло, вы меня слышите?»" in message.message
    assert "Нужно перезвонить" in message.message
    assert message.caller_phone == "+79991234567" and message.created_at == NOW
    assert message.name is None and message.phone is None


async def test_a_success_in_between_resets_the_counter():
    session, _, sink, _ = make_session(
        failing_round(),
        failing_round(),  # turn 1 fails
        text("Слушаю вас."),  # turn 2 succeeds
        failing_round(),
        failing_round(),  # turn 3 fails: the first failure again, not the second
    )

    await turn(session, "раз")
    await turn(session, "два")
    third = await turn(session, "три")

    assert third[-1] == Say(ASK_TO_REPEAT)
    assert not session.ended and sink.messages == []


async def test_the_callback_holds_only_the_last_three_lines_truncated():
    session, _, sink, _ = make_session(*[failing_round()] * 4, max_failed_turns=4)
    for line in ("первая", "вторая", "третья"):
        session._utterances.append(line)  # earlier turns of the call
    long_line = "я" * (MAX_UTTERANCE_CHARS + 200)

    await turn(session, long_line)
    await turn(session, "последняя")
    await session._save_callback()

    message = sink.messages[-1].message
    assert "первая" not in message and "вторая" not in message  # older than the last three
    assert "«третья»" in message
    assert "я" * MAX_UTTERANCE_CHARS in message and "я" * (MAX_UTTERANCE_CHARS + 1) not in message
    assert "«последняя»" in message


async def test_a_failing_sink_does_not_stop_the_call_from_ending(caplog):
    class BrokenSink(InMemorySink):
        async def add_message(self, message):
            raise RuntimeError("disk full")

    session, _, _, _ = make_session(*[failing_round()] * 4, sink=BrokenSink())

    await turn(session, "раз")
    with caplog.at_level("ERROR"):
        second = await turn(session, "два")

    assert second[-1] == EndCall() and session.ended
    assert "could not save the callback message" in caplog.text


async def test_without_caller_id_the_callback_still_saves():
    session, _, sink, _ = make_session(*[failing_round()] * 4, caller_phone=None)

    await turn(session, "раз")
    await turn(session, "два")

    assert sink.messages[0].caller_phone is None


# --- The phrases ----------------------------------------------------------------------------------


def test_fallback_phrases_use_masculine_forms_to_be_matched_with_the_tts_voice():
    """Documented on purpose: «не расслышал» is masculine. Step 2 must pick a matching voice
    or change this phrase."""
    assert "не расслышал" in ASK_TO_REPEAT


@pytest.mark.parametrize("phrase", [ASK_TO_REPEAT, FINAL_APOLOGY])
def test_fallback_phrases_have_no_digits_and_no_markup(phrase):
    assert not any(ch.isdigit() for ch in phrase)
    assert not any(ch in phrase for ch in "*_#<>")
