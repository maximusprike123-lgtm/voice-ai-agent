"""Pick the simulated caller's model by measuring how well it plays a caller.

A caller model that ends the call too early (writes the hang-up marker mid-conversation) or
never ends it makes a sweep measure the CALLER, not the agent. This probes each candidate on
short, fixed conversation prefixes taken from the scenarios and counts:

  premature   the hang-up marker on a line that must not end the call (greeting, «как вас
              зовут?», the read-back, all non-final): lower is better, 0 is the goal;
  terminates  a reply to «Могу ещё чем-то помочь?» that says goodbye and ends the call
              (the call CAN end): higher is better;
  empty       empty replies.

    .venv/bin/python -m evals --calibrate-caller
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field

from agent.llm import LLMClient, LLMError, Message, Role, TextDelta
from evals.caller import END_MARKER, build_persona_prompt, is_farewell
from evals.model import Scenario

GREETING = "Здравствуйте! Детейлинг-центр «Пример». Чем могу помочь?"
READ_BACK = (
    "Проверьте, пожалуйста: Игорь, полировка кузова, автомобиль Тойота Камри, в субботу, "
    "двадцать шестого сентября, днём. Номер телефона заканчивается на четыре пять шесть семь. "
    "Всё верно?"
)
ACCEPTED = (
    "Заявка принята и передана администратору, он перезвонит для подтверждения. "
    "Могу ещё чем-то помочь?"
)
# Scenarios where the caller books, so the read-back / acceptance probes make sense.
BOOKING_SCENARIOS = {
    "happy_path_booking",
    "approximate_time",
    "changes_mind",
    "hidden_caller_id",
    "service_not_listed",
    "sunday_closed",
}


@dataclass(frozen=True)
class Probe:
    label: str  # "start" | "name" | "readback" | "closing"
    history: tuple[Message, ...]  # conversation so far, after the persona prompt
    must_not_end: bool  # a hang-up here is premature


def probes_for(scenario: Scenario) -> list[Probe]:
    start = (Message(Role.USER, GREETING),)
    probes = [Probe("start", start, True)]
    if scenario.id in BOOKING_SCENARIOS:
        name = (
            *start,
            Message(Role.ASSISTANT, "Здравствуйте, хочу записаться."),
            Message(Role.USER, "Как вас зовут?"),
        )
        read_back = (
            *name,
            Message(Role.ASSISTANT, "Игорь."),
            Message(Role.USER, READ_BACK),
        )
        closing = (
            *read_back,
            Message(Role.ASSISTANT, "Да, всё верно."),
            Message(Role.USER, ACCEPTED),
        )
        probes += [
            Probe("name", name, True),
            Probe("readback", read_back, True),
            Probe("closing", closing, False),
        ]
    return probes


@dataclass
class CandidateResult:
    model: str
    premature: int = 0
    premature_of: int = 0
    terminates: int = 0
    terminates_of: int = 0
    empty: int = 0
    total: int = 0
    latencies: list[float] = field(default_factory=list)
    error: str = ""  # the last request error, if any
    errors: int = 0

    @property
    def premature_rate(self) -> float:
        return self.premature / self.premature_of if self.premature_of else 1.0

    @property
    def terminate_rate(self) -> float:
        return self.terminates / self.terminates_of if self.terminates_of else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else float("inf")

    def sort_key(self) -> tuple:
        """Best first: no premature hang-ups, then able to end the call, then no empties."""
        if not self.total:
            return (2.0, 1.0, 1.0, float("inf"))  # nothing usable came back
        return (
            round(self.premature_rate, 3),
            round(1 - self.terminate_rate, 3),
            (self.empty + self.errors) / (self.total + self.errors),
            self.median_latency,
        )


async def _reply(llm: LLMClient, scenario: Scenario, probe: Probe) -> str:
    messages = [Message(Role.SYSTEM, build_persona_prompt(scenario)), *probe.history]
    parts = []
    async for event in llm.stream(messages):
        if isinstance(event, TextDelta):
            parts.append(event.text)
    return "".join(parts)


async def calibrate_model(
    llm: LLMClient, model: str, scenarios: list[Scenario], samples: int, concurrency: int = 8
) -> CandidateResult:
    result = CandidateResult(model)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(scenario: Scenario, probe: Probe) -> None:
        async with semaphore:
            started = time.monotonic()
            try:
                raw = await _reply(llm, scenario, probe)
            except LLMError as exc:
                result.error = str(exc)[:120]
                result.errors += 1
                return
            result.latencies.append(time.monotonic() - started)
            result.total += 1
            if not raw.strip():
                result.empty += 1
                return
            marker = END_MARKER in raw or "[КОНЕЦ" in raw
            if probe.must_not_end:
                result.premature_of += 1
                result.premature += marker
            else:
                result.terminates_of += 1
                result.terminates += marker and is_farewell(raw)

    jobs = [
        one(scenario, probe)
        for scenario in scenarios
        for probe in probes_for(scenario)
        for _ in range(samples)
    ]
    await asyncio.gather(*jobs)
    return result


def format_calibration(results: list[CandidateResult]) -> str:
    ranked = sorted(results, key=CandidateResult.sort_key)
    lines = [
        "",
        "SIMULATED CALLER CALIBRATION (best first)",
        "",
        f"{'model':<44}{'premature':>11}{'terminates':>12}{'empty':>7}{'p50 s':>7}",
    ]
    for r in ranked:
        if not r.total:
            lines.append(f"{r.model:<44}  unusable: {r.error or 'no replies'}")
            continue
        note = f"   ({r.errors} request errors)" if r.errors else ""
        lines.append(
            f"{r.model:<44}{r.premature:>5}/{r.premature_of:<5}{r.terminates:>6}/{r.terminates_of:<5}"
            f"{r.empty:>7}{r.median_latency:>7.1f}{note}"
        )
    usable = [r for r in ranked if r.total]
    if usable:
        lines += ["", "recommended order: " + ", ".join(r.model for r in usable)]
    return "\n".join(lines)
