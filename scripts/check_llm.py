"""Live check of the configured LLM backend (reads real settings from .env, makes real calls).

Not part of `pytest` — it needs a running backend and is meant to be run by hand whenever
the backend, model or LLM_REASONING_EFFORT changes:

    .venv/bin/python scripts/check_llm.py

It checks what a voice agent depends on:
  1. Thinking is actually disabled: no reasoning text leaks, AND the backend's own token
     accounting shows no hidden reasoning (see check_reasoning_tokens).
  2. Tool calling works and produces arguments we can parse.
  3. Time to first token is low enough for a phone call.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

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
# A one-word answer that costs more completion tokens than this means hidden thinking.
MAX_COMPLETION_TOKENS_FOR_ONE_WORD = 20

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


async def _completion_usage(settings, with_reasoning_setting: bool) -> tuple[dict, str]:
    """One non-streaming completion; returns the standard OpenAI `usage` object and the answer.

    Deliberately bypasses LLMClient: token accounting is not part of its interface, and this
    check must see what the backend reports, not what the client chooses to surface.
    """
    body = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": "Скажи одним словом: сколько будет 2+2?"}],
    }
    if with_reasoning_setting and settings.llm_reasoning_effort is not None:
        body["reasoning_effort"] = settings.llm_reasoning_effort
    try:
        async with httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
            timeout=settings.llm_timeout_seconds,
        ) as http:
            response = await http.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("usage") or {}, data["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"usage probe failed: {type(exc).__name__}: {exc}") from exc


async def check_reasoning_tokens(settings) -> bool:
    """Fail if the backend's token accounting shows the model actually thought.

    The streaming client drops reasoning deltas, so a model that thinks silently can still
    produce a clean-looking answer; it only shows up as completion tokens (and latency/cost).
    """
    print("\n=== 1b. Hidden reasoning tokens (backend's own accounting) ===")
    setting = settings.llm_reasoning_effort
    print(f"LLM_REASONING_EFFORT = {setting!r}")

    usage, answer = await _completion_usage(settings, with_reasoning_setting=True)
    completion = usage.get("completion_tokens")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    print(
        f"with your settings: completion_tokens={completion}  reasoning_tokens={reasoning}  "
        f"answer={answer.strip()[:40]!r}"
    )

    if setting is not None:  # baseline: same request without the setting, for comparison
        base_usage, _ = await _completion_usage(settings, with_reasoning_setting=False)
        base_reasoning = (base_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        print(
            f"without the setting (baseline): completion_tokens="
            f"{base_usage.get('completion_tokens')}  reasoning_tokens={base_reasoning}"
        )

    if completion is None:
        print("!! The backend reports no usage.completion_tokens, so thinking cannot be verified.")
        return False
    if reasoning:
        print(f"!! {reasoning} reasoning tokens were spent: thinking is ON.")
        return False
    if completion > MAX_COMPLETION_TOKENS_FOR_ONE_WORD:
        print(
            f"!! {completion} completion tokens for a one-word answer "
            f"(limit {MAX_COMPLETION_TOKENS_FOR_ONE_WORD}): thinking is probably ON."
        )
        return False
    print("OK: no hidden reasoning tokens.")
    return True


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
        "no_hidden_reasoning": await check_reasoning_tokens(settings),
        "tool_calling": await check_tool_calling(settings),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
