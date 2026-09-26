"""Text to speech: one interface, several voices, and a fallback that keeps the agent talking.

`TTSClient` turns ONE sentence into 8 kHz mono 16-bit PCM chunks (the telephone format), so the
rest of the app never knows which voice is behind it. `FailoverTTS` puts a primary voice (cloud,
best quality, may be blocked or slow) in front of a fallback (local, always there):

  - a primary that is not configured (no API key) is skipped, and says why;
  - a primary that fails, or gives no first audio within `first_chunk_timeout`, loses THAT
    sentence to the fallback and then sits out `pause_seconds` (the next sentence after the pause
    is the probe: if it fails too, the pause starts again);
  - once audio has started, the primary is never swapped out mid-sentence: a voice change in the
    middle of a word is worse than a cut sentence, so the rest is dropped and the primary paused.

A sentence is only lost if the fallback fails too, and then `synthesize` raises TtsError.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_FIRST_CHUNK_TIMEOUT = 1.5
DEFAULT_CHUNK_TIMEOUT = 4.0  # after the first chunk: how long a stream may stall
DEFAULT_PAUSE_SECONDS = 60.0


class TtsError(Exception):
    """A voice could not produce this sentence. The message never contains an API key."""


class TTSClient(Protocol):
    name: str

    @property
    def unconfigured_reason(self) -> str | None:
        """Why this voice cannot be used at all («key: not set»), or None if it is configured."""
        ...

    async def warm_up(self) -> None:
        """Load the model / open the connection, so the first sentence is not slowed by it.
        Safe to call more than once; a failure raises TtsError."""
        ...

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """One sentence as 8 kHz mono 16-bit PCM chunks (an even number of bytes each), first
        chunk as early as the voice can. Raises TtsError."""
        ...

    async def aclose(self) -> None: ...


@dataclass
class FailoverStats:
    primary_sentences: int = 0
    fallback_sentences: int = 0
    primary_failures: int = 0
    cut_sentences: int = 0  # the primary died after audio had started
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=10))


class FailoverTTS:
    name = "failover"

    def __init__(
        self,
        primary: TTSClient,
        fallback: TTSClient | None,
        *,
        first_chunk_timeout: float = DEFAULT_FIRST_CHUNK_TIMEOUT,
        chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT,
        pause_seconds: float = DEFAULT_PAUSE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._first_chunk_timeout = first_chunk_timeout
        self._chunk_timeout = chunk_timeout
        self._pause_seconds = pause_seconds
        self._clock = clock
        self._paused_until = 0.0
        self._told_unconfigured = False
        self.stats = FailoverStats()
        self.last_provider: str | None = None  # who spoke the latest sentence

    @property
    def unconfigured_reason(self) -> str | None:
        if self._primary_configured() or self._usable_fallback() is not None:
            return None
        return f"no voice is configured (primary: {self._primary.unconfigured_reason})"

    @property
    def primary_paused(self) -> bool:
        return self._clock() < self._paused_until

    def _primary_configured(self) -> bool:
        return self._primary.unconfigured_reason is None

    def _usable_fallback(self) -> TTSClient | None:
        if self._fallback is not None and self._fallback.unconfigured_reason is None:
            return self._fallback
        return None

    async def warm_up(self) -> None:
        """Warm the fallback first (it must work), then try the primary; a primary that fails to
        warm up starts its pause instead of failing the call."""
        if (fallback := self._usable_fallback()) is not None:
            await fallback.warm_up()
        if self._primary_configured():
            try:
                await self._primary.warm_up()
            except Exception as exc:
                self._primary_failed(f"warm-up: {exc}")

    async def aclose(self) -> None:
        for client in (self._primary, self._fallback):
            if client is not None:
                await client.aclose()

    def _primary_failed(self, reason: str) -> None:
        self.stats.primary_failures += 1
        self.stats.recent_errors.append(reason)
        self._paused_until = self._clock() + self._pause_seconds
        logger.warning(
            "primary TTS %s failed (%s); the fallback speaks for %.0f s",
            self._primary.name,
            reason,
            self._pause_seconds,
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if self._primary_configured():
            if not self.primary_paused:
                started = False
                try:
                    async with aclosing(self._primary.synthesize(text)) as stream:
                        chunks = aiter(stream)
                        timeout = self._first_chunk_timeout
                        while True:
                            try:
                                async with asyncio.timeout(timeout):
                                    chunk = await anext(chunks)
                            except StopAsyncIteration:
                                break
                            except TimeoutError:
                                what = "more audio" if started else "first audio"
                                raise TtsError(f"no {what} within {timeout:g} s") from None
                            started, timeout = True, self._chunk_timeout
                            yield chunk
                    if not started:
                        raise TtsError("the voice returned no audio")
                    self.stats.primary_sentences += 1
                    self.last_provider = self._primary.name
                    return
                except Exception as exc:
                    self._primary_failed(f"{type(exc).__name__}: {exc}")
                    if started:
                        # Audio is already out: the rest of this sentence is lost, but the voice
                        # must not change mid-word. The next sentence goes to the fallback.
                        self.stats.cut_sentences += 1
                        logger.error("primary TTS died mid-sentence; %r was cut", text)
                        return
        elif not self._told_unconfigured:
            self._told_unconfigured = True
            logger.info(
                "primary TTS %s is not configured (%s): using the fallback",
                self._primary.name,
                self._primary.unconfigured_reason,
            )

        fallback = self._usable_fallback()
        if fallback is None:
            raise TtsError("no TTS voice is available for this sentence")
        self.stats.fallback_sentences += 1
        self.last_provider = fallback.name
        async with aclosing(fallback.synthesize(text)) as stream:
            async for chunk in stream:
                yield chunk
