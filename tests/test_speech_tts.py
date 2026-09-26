"""Offline tests for FailoverTTS: the fallback that keeps the agent talking."""

import asyncio
import logging

import pytest

from agent.speech.tts import FailoverTTS, TtsError

FRAME = b"\x01\x00" * 160  # 20 ms of 8 kHz audio


class FakeVoice:
    """A scripted TTS voice. `reason` makes it unconfigured; `error` raises before any audio;
    `delay` waits before the first chunk; `die_after` / `stall_after` break it after that many
    chunks."""

    def __init__(
        self,
        name,
        chunks=(FRAME, FRAME, FRAME),
        *,
        reason=None,
        error=None,
        delay=0.0,
        die_after=None,
        stall_after=None,
        warm_error=None,
    ):
        self.name = name
        self._chunks = list(chunks)
        self._reason = reason
        self._error = error
        self._delay = delay
        self._die_after = die_after
        self._stall_after = stall_after
        self._warm_error = warm_error
        self.calls = []
        self.finished = 0  # generators that were closed or ran to the end
        self.warmed = 0
        self.closed = 0

    @property
    def unconfigured_reason(self):
        return self._reason

    async def warm_up(self):
        self.warmed += 1
        if self._warm_error:
            raise self._warm_error

    async def synthesize(self, text):
        self.calls.append(text)
        try:
            if self._error:
                raise self._error
            if self._delay:
                await asyncio.sleep(self._delay)
            for index, chunk in enumerate(self._chunks):
                if self._die_after is not None and index == self._die_after:
                    raise TtsError("connection lost")
                if self._stall_after is not None and index == self._stall_after:
                    await asyncio.sleep(10)
                yield chunk
        finally:
            self.finished += 1

    async def aclose(self):
        self.closed += 1


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


async def speak(tts, text="Здравствуйте."):
    return [chunk async for chunk in tts.synthesize(text)]


def make(primary, fallback, clock=None, **kwargs):
    kwargs.setdefault("first_chunk_timeout", 0.2)
    kwargs.setdefault("chunk_timeout", 0.2)
    kwargs.setdefault("pause_seconds", 60.0)
    return FailoverTTS(primary, fallback, clock=clock or Clock(), **kwargs)


# --- The primary works ---------------------------------------------------------------------------


async def test_a_working_primary_speaks_and_the_fallback_is_left_alone():
    primary, fallback = FakeVoice("cloud"), FakeVoice("local")
    tts = make(primary, fallback)

    assert await speak(tts) == [FRAME, FRAME, FRAME]

    assert (primary.calls, fallback.calls) == (["Здравствуйте."], [])
    assert tts.last_provider == "cloud" and tts.stats.primary_sentences == 1
    assert not tts.primary_paused and tts.stats.primary_failures == 0


# --- The primary is not configured ---------------------------------------------------------------


async def test_an_unconfigured_primary_is_never_tried_and_the_reason_is_logged_once(caplog):
    primary, fallback = FakeVoice("cloud", reason="key: not set"), FakeVoice("local")
    tts = make(primary, fallback)

    with caplog.at_level(logging.INFO):
        await speak(tts, "Раз.")
        await speak(tts, "Два.")

    assert primary.calls == [] and fallback.calls == ["Раз.", "Два."]
    assert tts.last_provider == "local" and tts.stats.fallback_sentences == 2
    assert caplog.text.count("key: not set") == 1
    assert tts.stats.primary_failures == 0 and not tts.primary_paused


# --- The primary fails ---------------------------------------------------------------------------


async def test_a_primary_error_hands_that_sentence_to_the_fallback_and_pauses_the_primary():
    primary = FakeVoice("cloud", error=TtsError("HTTP 500"))
    fallback = FakeVoice("local")
    tts = make(primary, fallback)

    assert await speak(tts, "Раз.") == [FRAME, FRAME, FRAME]  # the fallback's audio

    assert fallback.calls == ["Раз."] and tts.last_provider == "local"
    assert tts.primary_paused and tts.stats.primary_failures == 1
    assert "HTTP 500" in tts.stats.recent_errors[0]


async def test_while_paused_the_primary_is_not_asked_at_all():
    primary = FakeVoice("cloud", error=TtsError("HTTP 500"))
    tts = make(primary, FakeVoice("local"))
    await speak(tts, "Раз.")

    await speak(tts, "Два.")
    await speak(tts, "Три.")

    assert primary.calls == ["Раз."]  # only the sentence that failed


async def test_after_the_pause_the_next_sentence_probes_the_primary_again():
    clock = Clock()
    primary = FakeVoice("cloud", error=TtsError("HTTP 500"))
    tts = make(primary, FakeVoice("local"), clock)
    await speak(tts, "Раз.")

    clock.now += 59
    await speak(tts, "Два.")
    assert primary.calls == ["Раз."]

    clock.now += 2  # 61 s since the failure
    await speak(tts, "Три.")
    assert primary.calls == ["Раз.", "Три."] and tts.primary_paused  # failed again: a new pause


async def test_a_primary_that_recovers_takes_over_again():
    clock = Clock()
    primary = FakeVoice("cloud", error=TtsError("HTTP 500"))
    tts = make(primary, FakeVoice("local"), clock)
    await speak(tts, "Раз.")

    primary._error = None  # the outage is over
    clock.now += 61
    await speak(tts, "Два.")

    assert tts.last_provider == "cloud" and not tts.primary_paused
    await speak(tts, "Три.")
    assert primary.calls == ["Раз.", "Два.", "Три."]


async def test_no_first_audio_in_time_counts_as_a_failure_and_the_stream_is_closed():
    primary = FakeVoice("cloud", delay=5.0)
    fallback = FakeVoice("local")
    tts = make(primary, fallback, first_chunk_timeout=0.05)

    assert await speak(tts) == [FRAME, FRAME, FRAME]

    assert tts.last_provider == "local" and tts.primary_paused
    assert primary.finished == 1  # cancelled, not left running
    assert "first audio" in tts.stats.recent_errors[0]


async def test_a_primary_that_returns_no_audio_is_a_failure():
    primary = FakeVoice("cloud", chunks=())
    tts = make(primary, FakeVoice("local"))

    await speak(tts)

    assert tts.last_provider == "local" and "no audio" in tts.stats.recent_errors[0]


async def test_any_exception_from_the_primary_is_survived():
    primary = FakeVoice("cloud", error=RuntimeError("bug in a client"))
    tts = make(primary, FakeVoice("local"))

    assert await speak(tts) == [FRAME, FRAME, FRAME]
    assert "RuntimeError" in tts.stats.recent_errors[0]


# --- The primary dies after it started -----------------------------------------------------------


async def test_a_primary_that_dies_mid_sentence_is_not_replaced_by_another_voice():
    primary = FakeVoice("cloud", die_after=2)
    fallback = FakeVoice("local")
    tts = make(primary, fallback)

    audio = await speak(tts, "Раз.")

    assert audio == [FRAME, FRAME]  # what the caller heard; no second voice mid-sentence
    assert fallback.calls == [] and tts.stats.cut_sentences == 1 and tts.primary_paused
    await speak(tts, "Два.")
    assert fallback.calls == ["Два."]  # the next sentence is the fallback's


async def test_a_primary_that_stalls_mid_sentence_is_cut_the_same_way():
    primary = FakeVoice("cloud", stall_after=1)
    tts = make(primary, FakeVoice("local"), chunk_timeout=0.05)

    assert await speak(tts) == [FRAME] and tts.stats.cut_sentences == 1


# --- Nobody can speak ----------------------------------------------------------------------------


async def test_when_the_fallback_fails_too_the_sentence_raises():
    tts = make(FakeVoice("cloud", error=TtsError("x")), FakeVoice("local", error=TtsError("boom")))
    with pytest.raises(TtsError, match="boom"):
        await speak(tts)


@pytest.mark.parametrize("fallback", [None, FakeVoice("local", reason="model missing")])
async def test_without_a_usable_fallback_a_failed_primary_raises(fallback):
    tts = make(FakeVoice("cloud", error=TtsError("x")), fallback)
    with pytest.raises(TtsError, match="no TTS voice"):
        await speak(tts)


async def test_a_lone_primary_that_works_needs_no_fallback():
    tts = make(FakeVoice("cloud"), None)
    assert await speak(tts) == [FRAME, FRAME, FRAME]


# --- Barge-in ------------------------------------------------------------------------------------


async def test_stopping_the_playback_closes_the_voice_and_is_not_a_failure():
    primary = FakeVoice("cloud", chunks=[FRAME] * 50)
    tts = make(primary, FakeVoice("local"))

    stream = tts.synthesize("Длинная фраза.")
    assert await anext(stream) == FRAME
    await stream.aclose()  # the caller interrupted

    assert primary.finished == 1
    assert tts.stats.primary_failures == 0 and not tts.primary_paused


async def test_cancelling_the_task_closes_the_voice():
    primary = FakeVoice("cloud", stall_after=1)
    tts = make(primary, FakeVoice("local"), chunk_timeout=30.0)

    task = asyncio.create_task(speak(tts))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert primary.finished == 1 and not tts.primary_paused


# --- Warm-up and closing -------------------------------------------------------------------------


async def test_warm_up_warms_the_fallback_and_a_configured_primary():
    primary, fallback = FakeVoice("cloud"), FakeVoice("local")
    await make(primary, fallback).warm_up()
    assert (primary.warmed, fallback.warmed) == (1, 1)


async def test_an_unconfigured_primary_is_not_warmed_up():
    primary = FakeVoice("cloud", reason="key: not set")
    await make(primary, FakeVoice("local")).warm_up()
    assert primary.warmed == 0


async def test_a_primary_that_fails_to_warm_up_starts_its_pause_and_does_not_raise():
    primary = FakeVoice("cloud", warm_error=TtsError("unreachable"))
    tts = make(primary, FakeVoice("local"))

    await tts.warm_up()

    assert tts.primary_paused and "unreachable" in tts.stats.recent_errors[0]
    await speak(tts)
    assert primary.calls == []


async def test_a_fallback_that_fails_to_warm_up_is_a_hard_error():
    tts = make(FakeVoice("cloud"), FakeVoice("local", warm_error=TtsError("no model")))
    with pytest.raises(TtsError, match="no model"):
        await tts.warm_up()


async def test_closing_closes_both_voices():
    primary, fallback = FakeVoice("cloud"), FakeVoice("local")
    await make(primary, fallback).aclose()
    assert (primary.closed, fallback.closed) == (1, 1)


def test_the_failover_is_unconfigured_only_when_no_voice_can_speak():
    both = make(FakeVoice("cloud", reason="key: not set"), FakeVoice("local", reason="no model"))
    assert "key: not set" in both.unconfigured_reason
    only_fallback = make(FakeVoice("cloud", reason="key: not set"), FakeVoice("local"))
    assert only_fallback.unconfigured_reason is None
    assert make(FakeVoice("cloud"), None).unconfigured_reason is None
