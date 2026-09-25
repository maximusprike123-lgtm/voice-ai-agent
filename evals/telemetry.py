"""Per-request telemetry for the agent's LLM: who served it and how long the first event took.

`RecordingLLM` wraps the agent's client. It passes every event through unchanged and on the
same schedule, but it reads the underlying stream in a background task. That matters for the
requests DialogueEngine gives up on (first-event timeout): the engine cancels its wait, and
with it any chance of learning which provider was slow. Here the request is left running
(capped) so its provider and its real first-event latency are still recorded.
"""

import asyncio
import math
import statistics
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from agent.llm import LLMClient, LLMError, Message, StreamEvent, ToolSpec
from evals.cost import current_run, last_provider

BACKGROUND_CAP_SECONDS = 30.0  # how long a request the engine abandoned may keep running


@dataclass
class RequestRecord:
    scenario: str | None
    first_event_s: float | None = None  # None: nothing arrived before the record was closed
    total_s: float | None = None
    provider: str | None = None
    outcome: str = "running"  # completed | error | never_answered
    engine_gave_up: bool = False  # DialogueEngine stopped waiting (timeout or cancellation)
    error: str = ""


@dataclass
class RequestLog:
    records: list[RequestRecord] = field(default_factory=list)
    _background: set[asyncio.Task] = field(default_factory=set)

    def keep_alive(self, task: asyncio.Task) -> None:
        """Let an abandoned request run on for a while, then stop it."""

        async def cap() -> None:
            try:
                await asyncio.wait_for(asyncio.shield(task), BACKGROUND_CAP_SECONDS)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()

        watcher = asyncio.create_task(cap())
        self._background.add(watcher)
        watcher.add_done_callback(self._background.discard)

    async def drain(self) -> None:
        """Wait for the abandoned requests to finish (call before reading the results)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)


class RecordingLLM:
    def __init__(self, inner: LLMClient, log: RequestLog) -> None:
        self._inner = inner
        self._log = log

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamEvent]:
        run = current_run.get()
        record = RequestRecord(scenario=run.split("#")[0] if run else None)
        self._log.records.append(record)
        queue: asyncio.Queue = asyncio.Queue()
        started = time.monotonic()

        async def produce() -> None:
            last_provider.set(None)
            try:
                async for event in self._inner.stream(messages, tools):
                    if record.first_event_s is None:
                        record.first_event_s = time.monotonic() - started
                    queue.put_nowait(("event", event))
                record.outcome = "completed"
                queue.put_nowait(("end", None))
            except LLMError as exc:
                record.outcome, record.error = "error", str(exc)
                queue.put_nowait(("error", exc))
            except asyncio.CancelledError:
                record.outcome = (
                    "completed" if record.first_event_s is not None else "never_answered"
                )
                raise
            finally:
                record.total_s = time.monotonic() - started
                record.provider = last_provider.get()

        producer = asyncio.create_task(produce())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "event":
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    return
        except (asyncio.CancelledError, GeneratorExit):
            record.engine_gave_up = True
            if not producer.done():
                self._log.keep_alive(producer)
            raise


# --- Summaries ------------------------------------------------------------------------------------


@dataclass
class ProviderStats:
    provider: str
    requests: int
    first_p50: float | None
    first_p90: float | None
    first_max: float | None
    slow: int  # first event later than the engine's limit (these are the rounds that time out)
    gave_up: int  # the engine actually stopped waiting


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(values)))  # nearest rank
    return values[rank - 1]


def provider_stats(records: list[RequestRecord], limit: float) -> list[ProviderStats]:
    groups: dict[str, list[RequestRecord]] = {}
    for r in records:
        groups.setdefault(r.provider or "(unknown)", []).append(r)
    stats = []
    for name, group in groups.items():
        firsts = [r.first_event_s for r in group if r.first_event_s is not None]
        stats.append(
            ProviderStats(
                provider=name,
                requests=len(group),
                first_p50=statistics.median(firsts) if firsts else None,
                first_p90=_percentile(firsts, 90) if firsts else None,
                first_max=max(firsts) if firsts else None,
                slow=sum(1 for f in firsts if f > limit)
                + sum(1 for r in group if r.first_event_s is None),
                gave_up=sum(1 for r in group if r.engine_gave_up),
            )
        )
    return sorted(stats, key=lambda s: -s.requests)


def format_providers(records: list[RequestRecord], limit: float) -> str:
    if not records:
        return ""
    lines = ["", f"AGENT LLM REQUESTS BY PROVIDER (first-event limit {limit:g}s)", ""]
    lines.append(
        f"{'provider':<16}{'requests':>9}{'share':>7}{'p50':>7}{'p90':>7}{'max':>7}"
        f"{'>limit':>8}{'gave up':>9}"
    )
    total = len(records)

    def cell(value: float | None) -> str:
        return f"{value:7.2f}" if value is not None else "      -"

    for s in provider_stats(records, limit):
        lines.append(
            f"{s.provider:<16}{s.requests:>9}{s.requests / total:>7.0%}{cell(s.first_p50)}"
            f"{cell(s.first_p90)}{cell(s.first_max)}{s.slow:>8}{s.gave_up:>9}"
        )
    timed_out = [
        r
        for r in records
        if r.engine_gave_up and (r.first_event_s is None or r.first_event_s > limit)
    ]
    if timed_out:
        lines += [
            "",
            f"{len(timed_out)} rounds the engine gave up on, attributed by letting them finish:",
        ]
        for r in timed_out:
            first = (
                f"{r.first_event_s:.1f}s"
                if r.first_event_s is not None
                else "no output within the cap"
            )
            lines.append(
                f"  {r.scenario or '?':<22} provider {r.provider or '(unknown)':<14} "
                f"first event after {first}"
            )
    return "\n".join(lines)
