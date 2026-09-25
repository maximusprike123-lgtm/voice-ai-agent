"""Offline tests for the request telemetry wrapper (provider + first-event latency)."""

import asyncio

import pytest

from agent.dialogue import DialogueEngine, Say, ToolOutcome
from agent.llm import LLMError, Message, Role, StreamEnd, TextDelta
from evals import telemetry
from evals.cost import current_run, last_provider
from evals.telemetry import (
    RecordingLLM,
    RequestLog,
    RequestRecord,
    format_providers,
    provider_stats,
)


class FakeInner:
    """Stands in for OpenAICompatibleLLMClient: delays, an optional provider (set the way the
    usage hook would), and scripted events or an error."""

    def __init__(self, *, delay=0.0, provider=None, events=None, error=None, hang=False):
        self.delay, self.provider, self.error, self.hang = delay, provider, error, hang
        self.events = events if events is not None else [TextDelta("Привет."), StreamEnd("stop")]
        self.calls = 0

    async def stream(self, messages, tools=None):
        self.calls += 1
        if self.hang:
            await asyncio.Event().wait()
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        for event in self.events:
            yield event
        if self.provider:
            last_provider.set(self.provider)  # what the usage hook does at the end of a stream


MESSAGES = [Message(Role.USER, "hi")]


async def collect(llm):
    return [event async for event in llm.stream(MESSAGES)]


async def test_events_pass_through_unchanged_and_the_request_is_recorded():
    log = RequestLog()
    llm = RecordingLLM(FakeInner(delay=0.02, provider="Together"), log)

    events = await collect(llm)

    assert events == [TextDelta("Привет."), StreamEnd("stop")]
    [record] = log.records
    assert record.outcome == "completed" and record.provider == "Together"
    assert 0.015 < record.first_event_s < 0.5 and record.total_s >= record.first_event_s
    assert not record.engine_gave_up


async def test_the_scenario_is_taken_from_the_running_call():
    log = RequestLog()
    token = current_run.set("price_only#3")
    try:
        await collect(RecordingLLM(FakeInner(), log))
    finally:
        current_run.reset(token)
    assert log.records[0].scenario == "price_only"


async def test_a_request_without_a_provider_is_recorded_as_such():
    log = RequestLog()
    await collect(RecordingLLM(FakeInner(provider=None), log))
    assert log.records[0].provider is None


async def test_errors_reach_the_consumer_and_are_recorded():
    log = RequestLog()
    llm = RecordingLLM(FakeInner(error=LLMError("HTTP 502")), log)

    with pytest.raises(LLMError, match="HTTP 502"):
        await collect(llm)

    assert log.records[0].outcome == "error" and "502" in log.records[0].error


async def test_concurrent_requests_keep_their_own_providers():
    log = RequestLog()
    slow = RecordingLLM(FakeInner(delay=0.05, provider="Fireworks"), log)
    fast = RecordingLLM(FakeInner(delay=0.0, provider="Together"), log)

    await asyncio.gather(collect(slow), collect(fast))

    assert sorted(r.provider for r in log.records) == ["Fireworks", "Together"]


async def test_a_request_the_consumer_gives_up_on_is_still_attributed_afterwards():
    log = RequestLog()
    llm = RecordingLLM(FakeInner(delay=0.25, provider="DeepInfra"), log)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):  # the engine's first-event timeout
            await collect(llm)
    [record] = log.records
    assert record.engine_gave_up and record.first_event_s is None  # nothing had arrived yet

    await log.drain()

    assert record.provider == "DeepInfra" and record.outcome == "completed"
    assert 0.2 < record.first_event_s < 1.0  # the real latency the engine never saw


async def test_an_abandoned_request_that_never_answers_is_stopped_after_the_cap(monkeypatch):
    monkeypatch.setattr(telemetry, "BACKGROUND_CAP_SECONDS", 0.05)
    log = RequestLog()
    llm = RecordingLLM(FakeInner(hang=True), log)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02):
            await collect(llm)
    await asyncio.wait_for(log.drain(), 2.0)

    [record] = log.records
    assert record.outcome == "never_answered" and record.first_event_s is None
    assert record.provider is None and record.engine_gave_up


async def test_dialogue_engine_sees_no_difference_and_its_retry_is_logged_as_two_requests():
    log = RequestLog()
    slow_then_fast = [FakeInner(delay=0.3, provider="DeepInfra"), FakeInner(provider="Together")]

    class Sequenced:
        def __init__(self):
            self.i = 0

        def stream(self, messages, tools=None):
            inner = slow_then_fast[self.i]
            self.i += 1
            return inner.stream(messages, tools)

    class NoTools:
        specs = []

        def begin_turn(self):
            pass

        async def execute(self, call):
            return ToolOutcome("ok")

    engine = DialogueEngine(
        RecordingLLM(Sequenced(), log), NoTools(), "SYS", first_event_timeout=0.05
    )

    events = [e async for e in engine.respond("Привет")]
    await log.drain()

    assert events == [Say("Привет.")]  # the engine retried and got the fast answer
    first, second = log.records
    assert first.engine_gave_up and first.provider == "DeepInfra" and first.first_event_s > 0.2
    assert second.provider == "Together" and not second.engine_gave_up


def record(provider, first, gave_up=False) -> RequestRecord:
    return RequestRecord(
        "s",
        first_event_s=first,
        total_s=(first or 9) + 0.5,
        provider=provider,
        outcome="completed" if first is not None else "never_answered",
        engine_gave_up=gave_up,
    )


def test_provider_stats_group_and_count_the_slow_requests():
    records = [record("Together", f) for f in (0.5, 0.7, 0.9, 1.1)]
    records += [
        record("DeepInfra", 1.0),
        record("DeepInfra", 6.5, gave_up=True),
        record(None, None, True),
    ]

    stats = {s.provider: s for s in provider_stats(records, limit=4.0)}

    assert stats["Together"].requests == 4 and stats["Together"].slow == 0
    assert stats["Together"].first_p50 == pytest.approx(0.8)
    assert stats["DeepInfra"].slow == 1 and stats["DeepInfra"].gave_up == 1
    assert stats["DeepInfra"].first_max == 6.5
    assert stats["(unknown)"].slow == 1 and stats["(unknown)"].first_p50 is None
    assert [s.provider for s in provider_stats(records, 4.0)][
        0
    ] == "Together"  # most requests first


def test_the_provider_report_lists_the_timed_out_rounds_with_their_provider():
    records = [record("Together", 0.6), record("DeepInfra", 7.2, gave_up=True)]

    text = format_providers(records, limit=4.0)

    assert "AGENT LLM REQUESTS BY PROVIDER" in text and "Together" in text
    assert "1 rounds the engine gave up on" in text
    assert "provider DeepInfra" in text and "first event after 7.2s" in text
    assert format_providers([], 4.0) == ""
