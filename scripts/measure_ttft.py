"""Measure time to first token with the REAL system prompt, through DialogueEngine.

Not part of `pytest` — it makes real calls to the backend in .env:

    .venv/bin/python scripts/measure_ttft.py

For each turn it records two times, both measured from the moment the caller's text is handed
to the engine:
  - LLM TTFT: first text token from the LLM (what the backend controls);
  - first Say: first complete sentence out of the engine (what TTS will actually wait for).

Scenarios:
  cold          model unloaded first, so this includes loading it into memory;
  warm, miss    model loaded, but the prompt differs from the previous call (clock time and
                caller number changed), like consecutive real calls;
  warm, hit     model loaded, identical prompt prefix as the previous call;
  conversation  one call, three consecutive turns (history grows, prefix stays cached).

The unload step uses Ollama's `keep_alive: 0` and the prompt-token count uses the standard
`usage` field. Both live here in the script only; LLMClient stays vendor-agnostic.
"""

import asyncio
import statistics
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.business import load_business_config  # noqa: E402
from agent.dialogue import DialogueEngine, Say, ToolOutcome  # noqa: E402
from agent.llm import (  # noqa: E402
    LLMClient,
    Message,
    OpenAICompatibleLLMClient,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from agent.prompt import build_system_prompt  # noqa: E402
from agent.settings import get_settings  # noqa: E402

BUDGET_SECONDS = 1.5  # rough phone-call budget, same as check_llm.py
QUESTIONS = [
    "Сколько стоит полировка кузова?",
    "Где вы находитесь и до скольки работаете?",
    "Запишите меня на завтра на керамику.",
]
WARM_RUNS = 3

# Stand-ins for the step-1.6 tools, so the prompt sent to the model has realistic tool schemas.
_OBJ = {"type": "object", "properties": {}}
STUB_TOOLS = [
    ToolSpec(
        "submit_booking",
        "Отправить заявку на запись после подтверждения клиентом.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "phone": {"type": "string"},
                "car": {"type": "string"},
                "service_id": {"type": "string"},
                "preferred_date": {"type": "string", "description": "YYYY-MM-DD"},
                "preferred_time": {"type": "string", "description": "HH:MM"},
            },
            "required": ["name", "phone", "car", "service_id", "preferred_date"],
        },
    ),
    ToolSpec(
        "take_message",
        "Передать вопрос администратору, если ответа нет в сведениях.",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "message": {"type": "string"}},
            "required": ["message"],
        },
    ),
    ToolSpec("end_call", "Завершить звонок после прощания.", _OBJ),
]


class StubTools:
    specs = STUB_TOOLS

    async def execute(self, call: ToolCall) -> ToolOutcome:
        return ToolOutcome("ok", ends_call=call.name == "end_call")


class TimedLLM:
    """Wraps an LLMClient and records when the first text token of each stream arrives."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.first_token_at: float | None = None

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._inner.stream(messages, tools):
            if isinstance(event, TextDelta) and self.first_token_at is None:
                self.first_token_at = time.monotonic()
            yield event


@dataclass
class Sample:
    llm_ttft: float | None
    first_say: float | None
    total: float
    reply: str


@dataclass
class Stats:
    samples: list[Sample] = field(default_factory=list)


async def run_turn(engine: DialogueEngine, timed: TimedLLM, user_text: str) -> Sample:
    timed.first_token_at = None
    start = time.monotonic()
    first_say = None
    said: list[str] = []
    async for event in engine.respond(user_text):
        if isinstance(event, Say):
            if first_say is None:
                first_say = time.monotonic() - start
            said.append(event.text)
    llm_ttft = timed.first_token_at - start if timed.first_token_at else None
    return Sample(llm_ttft, first_say, time.monotonic() - start, " ".join(said))


def fmt(value: float | None) -> str:
    return f"{value:5.2f}s" if value is not None else "  n/a "


def report(title: str, samples: list[Sample], questions: list[str]) -> None:
    print(f"\n--- {title} ---")
    print(f"{'LLM TTFT':>9} {'1st Say':>8} {'total':>7}  question -> reply")
    for sample, question in zip(samples, questions, strict=True):
        reply = sample.reply if len(sample.reply) <= 70 else sample.reply[:67] + "..."
        print(
            f"{fmt(sample.llm_ttft):>9} {fmt(sample.first_say):>8} {fmt(sample.total):>7}  "
            f"{question[:28]!r} -> {reply!r}"
        )
    for label, values in (
        ("LLM TTFT", [s.llm_ttft for s in samples]),
        ("1st Say", [s.first_say for s in samples]),
    ):
        real = [v for v in values if v is not None]
        if len(real) > 1:
            print(
                f"  {label}: min {min(real):.2f}s  median {statistics.median(real):.2f}s  "
                f"max {max(real):.2f}s"
            )


async def unload_model(root_url: str, model: str) -> bool:
    """Ollama-specific: evict the model from memory to force a cold start."""
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.post(
                f"{root_url}/api/generate", json={"model": model, "keep_alive": 0}
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"(could not unload model: {exc}; skipping the cold-start run)")
        return False
    await asyncio.sleep(2)
    return True


async def count_prompt_tokens(settings, system_prompt: str) -> int | None:
    """Standard OpenAI `usage.prompt_tokens` from a 1-token non-streaming completion."""
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUESTIONS[0]},
        ],
        "tools": [t.to_api() for t in STUB_TOOLS],
        "max_tokens": 1,
    }
    if settings.llm_reasoning_effort is not None:
        body["reasoning_effort"] = settings.llm_reasoning_effort
    try:
        async with httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
            timeout=settings.llm_timeout_seconds,
        ) as http:
            response = await http.post("/chat/completions", json=body)
        response.raise_for_status()
        return int(response.json()["usage"]["prompt_tokens"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None


async def main() -> int:
    settings = get_settings()
    business = load_business_config(settings.business_config_path)
    now = datetime.now(settings.tz).replace(second=0, microsecond=0)

    inner = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    timed = TimedLLM(inner)

    def new_engine(at: datetime, phone: str = "+79991234567") -> DialogueEngine:
        return DialogueEngine(timed, StubTools(), build_system_prompt(business, at, phone))

    prompt = build_system_prompt(business, now, "+79991234567")
    print(f"Backend: {settings.llm_base_url}  model: {settings.llm_model}")
    print(f"LLM_REASONING_EFFORT = {settings.llm_reasoning_effort!r}")
    print(f"System prompt: {len(prompt)} characters (+ {len(STUB_TOOLS)} tool schemas)")

    # 1. Cold start.
    root_url = settings.llm_base_url.removesuffix("/v1")
    if await unload_model(root_url, settings.llm_model):
        cold = await run_turn(new_engine(now), timed, QUESTIONS[0])
        report("cold (model just unloaded)", [cold], QUESTIONS[:1])

    # 2. Warm, "new call" each time: different clock time and caller number, as in real life.
    miss = []
    for i, question in enumerate(QUESTIONS[:WARM_RUNS]):
        at = now + timedelta(minutes=i + 1)
        miss.append(await run_turn(new_engine(at, f"+7999000000{i}"), timed, question))
    report("warm, new call (time + caller changed)", miss, QUESTIONS[:WARM_RUNS])

    # 3. Warm, KV-cache hit: identical prompt each time (one untimed priming run first).
    await run_turn(new_engine(now), timed, QUESTIONS[0])
    hit = [await run_turn(new_engine(now), timed, q) for q in QUESTIONS[:WARM_RUNS]]
    report("warm, same prompt (cache hit)", hit, QUESTIONS[:WARM_RUNS])

    # 4. One conversation, consecutive turns.
    engine = new_engine(now)
    convo = [await run_turn(engine, timed, q) for q in QUESTIONS]
    report("one conversation, 3 turns", convo, QUESTIONS)

    tokens = await count_prompt_tokens(settings, prompt)
    print(
        f"\nPrompt size: {tokens if tokens is not None else 'n/a'} tokens (system + tools + 1 turn)"
    )

    await inner.aclose()

    worst = max((s.first_say for s in miss + hit + convo if s.first_say is not None), default=None)
    if worst is not None and worst > BUDGET_SECONDS:
        print(
            f"\nWarm first-sentence time exceeds the {BUDGET_SECONDS}s budget (worst {worst:.2f}s)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
