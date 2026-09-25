"""Offline tests for DialogueEngine and SentenceSplitter, using a scripted fake LLM."""

import asyncio
import re
from collections.abc import AsyncIterator

import pytest

from agent.dialogue import (
    MAX_LLM_ROUNDS,
    TOOL_FAILED_RESULT,
    DialogueEngine,
    DialogueError,
    EndCall,
    Say,
    SentenceSplitter,
    ToolOutcome,
    ToolResult,
)
from agent.llm import (
    LLMError,
    Message,
    Role,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)

# --- Fakes ------------------------------------------------------------------------------------


class ScriptedLLM:
    """Plays back one scripted list of events per stream() call and records what it was sent.

    A script entry may be an Exception instance (raised when the stream starts) or the marker
    HANG, which makes the stream block forever after its events (to test cancellation).
    """

    HANG = object()

    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec] | None] = []
        self.closed = 0

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        script = self._scripts.pop(0)
        try:
            if isinstance(script, Exception):
                raise script
            for item in script:
                if item is self.HANG:
                    await asyncio.Event().wait()
                yield item
        finally:
            self.closed += 1


class FakeTools:
    def __init__(self, outcomes=None, specs=None):
        self.specs = specs or []
        self.outcomes = outcomes or {}
        self.executed: list[ToolCall] = []
        self.turns_begun = 0

    def begin_turn(self) -> None:
        self.turns_begun += 1

    async def execute(self, call: ToolCall) -> ToolOutcome:
        self.executed.append(call)
        outcome = self.outcomes.get(call.name, ToolOutcome("ok"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def text(*chunks: str) -> list[StreamEvent]:
    return [*(TextDelta(c) for c in chunks), StreamEnd("stop")]


def tool_turn(*calls: ToolCall, before: str = "") -> list[StreamEvent]:
    events: list[StreamEvent] = [TextDelta(before)] if before else []
    events += [ToolCallEvent(c) for c in calls]
    return [*events, StreamEnd("tool_calls")]


async def collect(engine: DialogueEngine, user_text: str):
    return [event async for event in engine.respond(user_text)]


def make_engine(*scripts, tools=None, prompt="SYS"):
    llm = ScriptedLLM(*scripts)
    tools = tools or FakeTools()
    return DialogueEngine(llm, tools, prompt), llm, tools


# --- SentenceSplitter ---------------------------------------------------------------------------


def split_stream(*chunks: str) -> list[str]:
    splitter = SentenceSplitter()
    out = []
    for chunk in chunks:
        out += splitter.feed(chunk)
    if tail := splitter.flush():
        out.append(tail)
    return out


def test_splits_on_terminators_followed_by_space():
    assert split_stream("Здравствуйте! Чем могу помочь? Слушаю вас.") == [
        "Здравствуйте!",
        "Чем могу помочь?",
        "Слушаю вас.",
    ]


def test_terminator_waits_for_next_chunk_before_splitting():
    splitter = SentenceSplitter()
    assert splitter.feed("Добрый день.") == []
    assert splitter.feed(" Чем") == ["Добрый день."]
    assert splitter.flush() == "Чем"


def test_result_is_independent_of_chunk_boundaries():
    whole = "Мы работаем каждый день. Приезжайте к десяти! Ждём вас."
    expected = split_stream(whole)
    assert split_stream(*whole) == expected  # one character at a time
    assert split_stream(*re.findall(r"\S+\s*", whole)) == expected  # word-sized chunks


def test_does_not_split_decimals():
    assert split_stream("Это стоит 3.5 тысячи рублей. Хорошо?") == [
        "Это стоит 3.5 тысячи рублей.",
        "Хорошо?",
    ]


def test_flush_returns_unterminated_tail():
    assert split_stream("Одну минуту, сейчас проверю") == ["Одну минуту, сейчас проверю"]


def test_ellipsis_and_repeated_terminators_are_one_boundary():
    assert split_stream("Ну что вы... Конечно?! Да.") == ["Ну что вы...", "Конечно?!", "Да."]


def test_newline_is_a_boundary():
    assert split_stream("Первая строка\nВторая строка") == ["Первая строка", "Вторая строка"]


def test_short_fragment_is_merged_into_next_sentence():
    assert split_stream("Да. Записываю вас на завтра.") == ["Да. Записываю вас на завтра."]


def test_short_fragment_at_end_is_still_flushed():
    assert split_stream("Ок.") == ["Ок."]


def test_empty_input_yields_nothing():
    assert split_stream("", "  ", "\n") == []


# --- Plain turns ----------------------------------------------------------------------------------


async def test_plain_turn_emits_sentences_in_order():
    engine, llm, _ = make_engine(text("Здравствуйте! ", "Чем могу ", "помочь?"))

    events = await collect(engine, "Привет")

    assert events == [Say("Здравствуйте!"), Say("Чем могу помочь?")]


async def test_history_holds_system_user_and_assistant():
    engine, _, _ = make_engine(text("Здравствуйте! Чем помочь?"))

    await collect(engine, "Привет")

    assert engine.messages == [
        Message(Role.SYSTEM, "SYS"),
        Message(Role.USER, "Привет"),
        Message(Role.ASSISTANT, "Здравствуйте! Чем помочь?"),
    ]


async def test_second_turn_sends_full_history_to_llm():
    engine, llm, _ = make_engine(text("Первый ответ."), text("Второй ответ."))

    await collect(engine, "раз")
    await collect(engine, "два")

    assert [m.content for m in llm.calls[1]] == ["SYS", "раз", "Первый ответ.", "два"]


async def test_tool_specs_are_passed_to_llm_or_omitted_when_empty():
    spec = ToolSpec("end_call", "d", {"type": "object", "properties": {}})
    engine, llm, _ = make_engine(text("ок ок ок."), tools=FakeTools(specs=[spec]))
    await collect(engine, "x")
    assert llm.tools_seen == [[spec]]

    engine, llm, _ = make_engine(text("ок ок ок."))
    await collect(engine, "x")
    assert llm.tools_seen == [None]


async def test_messages_property_is_a_copy():
    engine, _, _ = make_engine(text("Привет."))
    engine.messages.clear()
    assert len(engine.messages) == 1


# --- Tool turns -----------------------------------------------------------------------------------


async def test_tool_call_then_second_llm_round_sees_the_result():
    call = ToolCall("c1", "take_message", '{"text": "вопрос"}')
    engine, llm, tools = make_engine(
        tool_turn(call, before="Передам администратору. "),
        text("Готово, передал."),
        tools=FakeTools(outcomes={"take_message": ToolOutcome("сохранено")}),
    )

    events = await collect(engine, "У меня вопрос")

    assert events == [
        Say("Передам администратору."),
        ToolResult(call, "сохранено"),
        Say("Готово, передал."),
    ]
    assert tools.executed == [call]
    assert engine.messages[2:] == [
        Message(Role.ASSISTANT, "Передам администратору.", tool_calls=(call,)),
        Message(Role.TOOL, "сохранено", tool_call_id="c1"),
        Message(Role.ASSISTANT, "Готово, передал."),
    ]
    assert llm.calls[1][-1] == Message(Role.TOOL, "сохранено", tool_call_id="c1")


async def test_tool_call_without_text_has_no_assistant_content():
    call = ToolCall("c1", "submit_booking", "{}")
    engine, _, _ = make_engine(tool_turn(call), text("Записал."))

    await collect(engine, "да, всё верно")

    assert engine.messages[2] == Message(Role.ASSISTANT, content=None, tool_calls=(call,))


async def test_parallel_tool_calls_all_run_and_all_get_results():
    a, b = ToolCall("c1", "a", "{}"), ToolCall("c2", "b", "{}")
    engine, llm, tools = make_engine(tool_turn(a, b), text("Готово."))

    events = await collect(engine, "x")

    assert tools.executed == [a, b]
    assert [e for e in events if isinstance(e, ToolResult)] == [
        ToolResult(a, "ok"),
        ToolResult(b, "ok"),
    ]
    assert [m.tool_call_id for m in engine.messages if m.role is Role.TOOL] == ["c1", "c2"]


async def test_end_call_outcome_emits_end_call_and_skips_further_llm_round():
    call = ToolCall("c1", "end_call", "{}")
    engine, llm, _ = make_engine(
        tool_turn(call, before="До свидания! "),
        tools=FakeTools(outcomes={"end_call": ToolOutcome("ok", ends_call=True)}),
    )

    events = await collect(engine, "спасибо, всё")

    assert events == [Say("До свидания!"), ToolResult(call, "ok"), EndCall()]
    assert len(llm.calls) == 1


async def test_tool_exception_becomes_error_result_and_dialogue_continues():
    call = ToolCall("c1", "submit_booking", "{}")
    engine, llm, _ = make_engine(
        tool_turn(call),
        text("Не получилось, передам администратору."),
        tools=FakeTools(outcomes={"submit_booking": RuntimeError("db down")}),
    )

    events = await collect(engine, "x")

    assert ToolResult(call, TOOL_FAILED_RESULT) in events
    assert llm.calls[1][-1] == Message(Role.TOOL, TOOL_FAILED_RESULT, tool_call_id="c1")


async def test_looping_tool_calls_are_capped():
    scripts = [tool_turn(ToolCall(f"c{i}", "t", "{}")) for i in range(MAX_LLM_ROUNDS + 3)]
    engine, llm, _ = make_engine(*scripts)

    with pytest.raises(DialogueError):
        await collect(engine, "x")

    assert len(llm.calls) == MAX_LLM_ROUNDS
    # History stays consistent: every tool call has its result.
    calls = [c.id for m in engine.messages for c in m.tool_calls or ()]
    results = [m.tool_call_id for m in engine.messages if m.role is Role.TOOL]
    assert calls == results


# --- begin_turn hook ------------------------------------------------------------------------------


async def test_begin_turn_is_called_once_per_caller_utterance_before_tools_run():
    order = []

    class RecordingTools(FakeTools):
        def begin_turn(self):
            super().begin_turn()
            order.append("begin_turn")

        async def execute(self, call):
            order.append(f"execute:{call.name}")
            return await super().execute(call)

    call = ToolCall("c1", "take_message", "{}")
    tools = RecordingTools()
    engine, _, _ = make_engine(tool_turn(call), text("Передал."), text("Пожалуйста."), tools=tools)

    await collect(engine, "первая реплика")
    await collect(engine, "вторая реплика")

    assert order == ["begin_turn", "execute:take_message", "begin_turn"]
    assert tools.turns_begun == 2  # a tool round inside one turn does not count as a new turn


# --- Scripted speech (ToolOutcome.say) ------------------------------------------------------------


READ_BACK = (
    "Проверьте, пожалуйста: Игорь, полировка. Номер заканчивается на четыре пять. Всё верно?"
)


async def test_say_is_spoken_verbatim_as_sentences_and_ends_the_turn():
    call = ToolCall("c1", "prepare_booking", "{}")
    engine, llm, _ = make_engine(
        tool_turn(call),
        tools=FakeTools(outcomes={"prepare_booking": ToolOutcome("черновик готов", say=READ_BACK)}),
    )

    events = await collect(engine, "запишите меня")

    assert events == [
        ToolResult(call, "черновик готов"),
        Say("Проверьте, пожалуйста: Игорь, полировка."),
        Say("Номер заканчивается на четыре пять."),
        Say("Всё верно?"),
    ]
    assert len(llm.calls) == 1  # no further LLM round: the engine waits for the caller


async def test_say_goes_into_the_history_as_an_assistant_message():
    call = ToolCall("c1", "prepare_booking", "{}")
    engine, _, _ = make_engine(
        tool_turn(call),
        tools=FakeTools(outcomes={"prepare_booking": ToolOutcome("черновик готов", say=READ_BACK)}),
    )

    await collect(engine, "запишите меня")

    assert engine.messages[2:] == [
        Message(Role.ASSISTANT, content=None, tool_calls=(call,)),
        Message(Role.TOOL, "черновик готов", tool_call_id="c1"),
        Message(Role.ASSISTANT, READ_BACK),
    ]


async def test_next_turn_after_a_say_sends_the_read_back_to_the_llm():
    call = ToolCall("c1", "prepare_booking", "{}")
    engine, llm, _ = make_engine(
        tool_turn(call),
        text("Хорошо, отправляю."),
        tools=FakeTools(outcomes={"prepare_booking": ToolOutcome("готово", say=READ_BACK)}),
    )

    await collect(engine, "запишите меня")
    await collect(engine, "да")

    assert llm.calls[1][-2:] == [Message(Role.ASSISTANT, READ_BACK), Message(Role.USER, "да")]


async def test_model_text_before_the_tool_call_is_spoken_before_the_say():
    call = ToolCall("c1", "prepare_booking", "{}")
    engine, _, _ = make_engine(
        tool_turn(call, before="Секунду. "),
        tools=FakeTools(outcomes={"prepare_booking": ToolOutcome("готово", say="Всё верно?")}),
    )

    events = await collect(engine, "x")

    assert [e for e in events if isinstance(e, Say)] == [Say("Секунду."), Say("Всё верно?")]


async def test_several_says_in_one_round_are_spoken_in_order():
    a, b = ToolCall("c1", "a", "{}"), ToolCall("c2", "b", "{}")
    engine, llm, _ = make_engine(
        tool_turn(a, b),
        tools=FakeTools(
            outcomes={
                "a": ToolOutcome("ok", say="Первое сообщение."),
                "b": ToolOutcome("ok", say="Второе сообщение."),
            }
        ),
    )

    events = await collect(engine, "x")

    assert [e.text for e in events if isinstance(e, Say)] == [
        "Первое сообщение.",
        "Второе сообщение.",
    ]
    assert engine.messages[-1] == Message(Role.ASSISTANT, "Первое сообщение. Второе сообщение.")
    assert len(llm.calls) == 1


async def test_say_together_with_ends_call_speaks_then_hangs_up():
    call = ToolCall("c1", "end_call", "{}")
    engine, llm, _ = make_engine(
        tool_turn(call),
        tools=FakeTools(
            outcomes={"end_call": ToolOutcome("ok", ends_call=True, say="Всего доброго!")}
        ),
    )

    events = await collect(engine, "x")

    assert events == [ToolResult(call, "ok"), Say("Всего доброго!"), EndCall()]
    assert len(llm.calls) == 1


async def test_tool_without_say_still_continues_with_another_llm_round():
    call = ToolCall("c1", "take_message", "{}")
    engine, llm, _ = make_engine(tool_turn(call), text("Передал."))

    events = await collect(engine, "x")

    assert events == [ToolResult(call, "ok"), Say("Передал.")]
    assert len(llm.calls) == 2


async def test_empty_say_is_ignored():
    call = ToolCall("c1", "t", "{}")
    engine, llm, _ = make_engine(
        tool_turn(call),
        text("Дальше."),
        tools=FakeTools(outcomes={"t": ToolOutcome("ok", say="")}),
    )

    assert await collect(engine, "x") == [ToolResult(call, "ok"), Say("Дальше.")]


async def test_barge_in_during_a_say_keeps_the_full_read_back_in_history():
    call = ToolCall("c1", "prepare_booking", "{}")
    engine, _, _ = make_engine(
        tool_turn(call),
        tools=FakeTools(outcomes={"prepare_booking": ToolOutcome("готово", say=READ_BACK)}),
    )

    gen = engine.respond("x")
    assert isinstance(await anext(gen), ToolResult)
    assert await anext(gen) == Say("Проверьте, пожалуйста: Игорь, полировка.")
    await gen.aclose()  # the caller interrupts after the first sentence

    # Trimming to what was actually heard is step 3 (mark_spoken); until then the model
    # still knows what the caller was being asked, and every tool call has its result.
    assert engine.messages[-1] == Message(Role.ASSISTANT, READ_BACK)
    assert engine.messages[-2].role is Role.TOOL


# --- Cancellation (barge-in) ----------------------------------------------------------------------


async def test_cancel_mid_stream_keeps_emitted_sentences_and_closes_stream():
    engine, llm, _ = make_engine(
        [TextDelta("Первая фраза. Вторая"), ScriptedLLM.HANG],
        text("Слушаю вас."),
    )
    first_sentence = asyncio.Event()

    async def consume():
        async for event in engine.respond("привет"):
            if isinstance(event, Say):
                first_sentence.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_sentence.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.closed == 1
    assert engine.messages[1:] == [
        Message(Role.USER, "привет"),
        Message(Role.ASSISTANT, "Первая фраза."),
    ]

    # The engine is still usable afterwards.
    assert await collect(engine, "а теперь?") == [Say("Слушаю вас.")]


async def test_aclose_mid_stream_closes_llm_stream():
    engine, llm, _ = make_engine(text("Первая фраза. Вторая фраза. Третья."))

    gen = engine.respond("привет")
    assert await anext(gen) == Say("Первая фраза.")
    await gen.aclose()

    assert llm.closed == 1
    assert engine.messages[-1] == Message(Role.ASSISTANT, "Первая фраза.")


async def test_cancel_before_any_sentence_leaves_no_assistant_message():
    engine, llm, _ = make_engine([ScriptedLLM.HANG])

    task = asyncio.create_task(collect(engine, "привет"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.messages[1:] == [Message(Role.USER, "привет")]


async def test_close_during_tool_results_fills_in_missing_results():
    a, b = ToolCall("c1", "a", "{}"), ToolCall("c2", "b", "{}")
    engine, _, _ = make_engine(tool_turn(a, b))

    gen = engine.respond("x")
    assert isinstance(await anext(gen), ToolResult)  # result of `a`; `b` has not run yet
    await gen.aclose()

    tool_messages = [m for m in engine.messages if m.role is Role.TOOL]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert tool_messages[0].content == "ok"
    assert "прерван" in tool_messages[1].content


# --- LLM errors -----------------------------------------------------------------------------------


async def test_llm_error_on_first_round_rolls_back_the_user_message():
    engine, _, _ = make_engine(
        text("Первый ответ."), LLMError("boom"), LLMError("boom"), text("Повторный ответ.")
    )
    await collect(engine, "раз")
    before = engine.messages

    with pytest.raises(LLMError):
        await collect(engine, "два")

    assert engine.messages == before
    assert await collect(engine, "два") == [Say("Повторный ответ.")]


async def test_llm_error_after_tool_round_keeps_consistent_history():
    call = ToolCall("c1", "submit_booking", "{}")
    engine, _, _ = make_engine(tool_turn(call), LLMError("boom"), LLMError("boom"))

    with pytest.raises(LLMError):
        await collect(engine, "да")

    # The booking really happened, so the tool call and its result stay in the history.
    assert [m.role for m in engine.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
    ]


# --- Timeouts, retry, empty replies ---------------------------------------------------------------


FAST = {"first_event_timeout": 0.05, "event_timeout": 0.1}


def make_fast_engine(*scripts, tools=None, **kwargs):
    llm = ScriptedLLM(*scripts)
    engine = DialogueEngine(llm, tools or FakeTools(), "SYS", **(FAST | kwargs))
    return engine, llm


def user_messages(engine):
    return [m for m in engine.messages if m.role is Role.USER]


def test_default_timeouts_are_4s_to_first_event_and_8s_between_events():
    engine = DialogueEngine(ScriptedLLM(), FakeTools(), "SYS")
    assert (engine._first_event_timeout, engine._event_timeout) == (4.0, 8.0)


async def test_a_stalled_first_token_is_retried_once_without_duplicating_the_user_message():
    engine, llm = make_fast_engine([ScriptedLLM.HANG], text("Здравствуйте, слушаю вас."))

    events = await collect(engine, "Привет")

    assert events == [Say("Здравствуйте, слушаю вас.")]
    assert len(llm.calls) == 2 and llm.closed == 2  # the stalled stream was closed
    assert [m.content for m in user_messages(engine)] == ["Привет"]  # only once
    assert llm.calls[1] == llm.calls[0]  # the retry sent exactly the same messages


async def test_two_stalls_raise_llm_error_within_twice_the_first_event_timeout():
    engine, llm = make_fast_engine([ScriptedLLM.HANG], [ScriptedLLM.HANG])

    started = asyncio.get_running_loop().time()
    with pytest.raises(LLMError, match="no first LLM output"):
        await collect(engine, "Привет")
    elapsed = asyncio.get_running_loop().time() - started

    assert 0.09 < elapsed < 0.4  # 2 x 0.05s plus scheduling, not 2 x event_timeout
    assert len(llm.calls) == 2 and llm.closed == 2
    assert engine.messages == [Message(Role.SYSTEM, "SYS")]  # rolled back: caller may repeat


async def test_a_stall_in_a_stream_that_already_started_is_not_retried():
    engine, llm = make_fast_engine([TextDelta("При"), ScriptedLLM.HANG], text("не должно быть"))

    with pytest.raises(LLMError, match="no more LLM output"):
        await collect(engine, "Привет")

    assert len(llm.calls) == 1  # no second attempt: that would double the wait
    assert engine.messages == [Message(Role.SYSTEM, "SYS")]


async def test_no_retry_once_the_caller_has_heard_part_of_the_answer():
    class DropsMidStream(ScriptedLLM):
        async def stream(self, messages, tools=None):
            self.calls.append(list(messages))
            yield TextDelta("Первая фраза. Вторая")
            raise LLMError("connection dropped")

    llm = DropsMidStream()
    engine = DialogueEngine(llm, FakeTools(), "SYS", **FAST)
    heard = []

    with pytest.raises(LLMError, match="connection dropped"):
        async for event in engine.respond("Привет"):
            heard.append(event)

    assert heard == [Say("Первая фраза.")]  # spoken before the failure, so no repeat
    assert len(llm.calls) == 1


async def test_a_steady_slow_stream_is_not_mistaken_for_a_stall():
    class Trickle(ScriptedLLM):
        async def stream(self, messages, tools=None):
            self.calls.append(list(messages))
            for chunk in ("Первая фраза. ", "Вторая фраза. ", "Третья фраза."):
                await asyncio.sleep(0.04)  # each gap is below event_timeout; the total is above
                yield TextDelta(chunk)
            yield StreamEnd("stop")

    llm = Trickle()
    engine = DialogueEngine(llm, FakeTools(), "SYS", **FAST)

    assert await collect(engine, "x") == [
        Say("Первая фраза."),
        Say("Вторая фраза."),
        Say("Третья фраза."),
    ]
    assert len(llm.calls) == 1


async def test_a_transport_error_before_any_output_is_retried_once():
    engine, llm = make_fast_engine(LLMError("HTTP 502"), text("Слушаю вас."))

    assert await collect(engine, "x") == [Say("Слушаю вас.")]
    assert len(llm.calls) == 2


async def test_an_empty_reply_is_retried_then_reported_as_an_error():
    engine, llm = make_fast_engine([StreamEnd("stop")], text("Слушаю вас."))
    assert await collect(engine, "x") == [Say("Слушаю вас.")]
    assert len(llm.calls) == 2

    engine, llm = make_fast_engine([StreamEnd("stop")], [StreamEnd("stop")])
    with pytest.raises(LLMError, match="empty reply"):
        await collect(engine, "x")
    assert len(llm.calls) == 2
    assert engine.messages == [Message(Role.SYSTEM, "SYS")]


async def test_an_empty_reply_after_a_tool_round_keeps_the_tool_results_in_the_history():
    call = ToolCall("c1", "take_message", "{}")
    engine, llm = make_fast_engine(tool_turn(call), [StreamEnd("stop")], [StreamEnd("stop")])

    with pytest.raises(LLMError, match="empty reply"):
        await collect(engine, "x")

    assert [m.role for m in engine.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
    ]
    assert len(llm.calls) == 3  # tool round, then empty twice


async def test_the_retry_applies_to_later_rounds_too():
    call = ToolCall("c1", "take_message", "{}")
    engine, llm = make_fast_engine(tool_turn(call), [ScriptedLLM.HANG], text("Передал."))

    events = await collect(engine, "x")

    assert events == [ToolResult(call, "ok"), Say("Передал.")]
    assert len(llm.calls) == 3
    assert [m.role for m in engine.messages].count(Role.TOOL) == 1  # tool ran once, not twice


async def test_a_tool_call_only_reply_is_not_empty():
    call = ToolCall("c1", "end_call", "{}")
    engine, llm = make_fast_engine(
        tool_turn(call), tools=FakeTools(outcomes={"end_call": ToolOutcome("ok", ends_call=True)})
    )

    assert await collect(engine, "x") == [ToolResult(call, "ok"), EndCall()]
    assert len(llm.calls) == 1


async def test_the_retry_warning_is_logged(caplog):
    engine, _ = make_fast_engine([ScriptedLLM.HANG], text("Слушаю вас."))

    with caplog.at_level("WARNING"):
        await collect(engine, "x")

    assert "retrying once" in caplog.text


def test_add_assistant_message_puts_the_greeting_into_the_history():
    engine, llm, _ = make_engine(text("Слушаю."))
    engine.add_assistant_message("Здравствуйте! Чем могу помочь?")

    assert engine.messages[-1] == Message(Role.ASSISTANT, "Здравствуйте! Чем могу помочь?")


async def test_the_model_sees_the_greeting_on_the_first_turn():
    engine, llm, _ = make_engine(text("Слушаю вас."))
    engine.add_assistant_message("Здравствуйте!")

    await collect(engine, "Хочу записаться")

    assert [m.content for m in llm.calls[0]] == ["SYS", "Здравствуйте!", "Хочу записаться"]
