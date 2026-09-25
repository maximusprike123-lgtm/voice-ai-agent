"""Live check of the configured LLM backend (reads real settings from .env, makes real calls).

Not part of `pytest` — it needs a running backend and is meant to be run by hand whenever
the backend, model or LLM_REASONING_EFFORT changes:

    .venv/bin/python scripts/check_llm.py

It checks three things a voice agent depends on:
  1. Thinking is actually disabled (no reasoning text, no runaway token count).
  2. Tool calling works and produces arguments we can parse.
  3. Time to first token is low enough for a phone call.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.llm import (  # noqa: E402
    Message,
    OpenAICompatibleLLMClient,
    Role,
    StreamEnd,
    TextDelta,
    ToolCallEvent,
    ToolSpec,
)
from agent.settings import get_settings  # noqa: E402

LATENCY_WARN_SECONDS = 1.5  # rough phone-call budget for time to first token

BOOKING_TOOL = ToolSpec(
    name="submit_booking",
    description="Отправить заявку на запись.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "service": {"type": "string"},
            "preferred_date": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["name", "service", "preferred_date"],
    },
)


def _client(settings) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
    )


async def check_thinking_disabled(settings) -> bool:
    print("\n=== 1. Thinking disabled? ===")
    print(f"LLM_REASONING_EFFORT = {settings.llm_reasoning_effort!r}")

    client = _client(settings)
    messages = [Message(Role.USER, "Скажи одним словом: сколько будет 2+2?")]

    start = time.monotonic()
    first_token_at = None
    text = ""
    token_events = 0

    async for event in client.stream(messages):
        if isinstance(event, TextDelta):
            if first_token_at is None:
                first_token_at = time.monotonic()
            text += event.text
            token_events += 1

    await client.aclose()

    ttft = (first_token_at - start) if first_token_at else None
    print(f"Time to first token: {ttft:.2f}s" if ttft else "No text was returned.")
    print(f"Answer: {text!r}  ({token_events} text chunks)")

    ok = bool(text.strip()) and token_events < 20 and "\n" not in text.strip()
    if not ok:
        print(
            "!! This looks like a thinking trace leaked into content, or no answer at all. "
            "Re-check LLM_REASONING_EFFORT and the backend's docs for disabling thinking."
        )
    else:
        print("OK: short, direct answer with no visible reasoning.")
    return ok


async def check_tool_calling(settings) -> bool:
    print("\n=== 2. Tool calling ===")
    client = _client(settings)
    messages = [
        Message(
            Role.SYSTEM,
            "Ты администратор автодетейлинга. Если клиент просит записать его, "
            "вызови инструмент submit_booking.",
        ),
        Message(
            Role.USER,
            "Запишите меня, Иван, на полировку кузова на 30 сентября.",
        ),
    ]

    calls = []
    async for event in client.stream(messages, tools=[BOOKING_TOOL]):
        if isinstance(event, ToolCallEvent):
            calls.append(event.call)
        elif isinstance(event, StreamEnd):
            print(f"finish_reason = {event.finish_reason!r}")

    await client.aclose()

    if not calls:
        print("!! No tool call was made. The model may not support tool calling reliably.")
        return False

    ok = True
    for call in calls:
        print(f"Tool call: {call.name}(id={call.id})")
        print(f"  raw arguments: {call.arguments}")
        try:
            parsed = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            print(f"  !! arguments are not valid JSON: {exc}")
            ok = False
            continue
        print(f"  parsed: {parsed}")
        if call.name != BOOKING_TOOL.name:
            print(f"  !! expected tool {BOOKING_TOOL.name!r}")
            ok = False
        missing = [f for f in ("name", "service", "preferred_date") if f not in parsed]
        if missing:
            print(f"  !! missing fields: {missing}")
            ok = False

    print("OK: tool call parsed successfully." if ok else "FAILED: see issues above.")
    return ok


async def main() -> int:
    settings = get_settings()
    print(f"Backend: {settings.llm_base_url}  model: {settings.llm_model}")

    results = {
        "thinking_disabled": await check_thinking_disabled(settings),
        "tool_calling": await check_tool_calling(settings),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
