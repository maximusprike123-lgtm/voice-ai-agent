"""One live dialogue against the real LLM in which a booking is completed.

Not part of `pytest` (real backend, real clock):

    .venv/bin/python scripts/live_booking_dialogue.py

A scripted "caller" answers whatever the agent just asked (matched by keywords), so the
transcript stays coherent even if the model asks in a different order. This caller gives only
an approximate time ("в субботу после обеда"), so a correct run leaves preferred_time empty
and puts the caller's words into notes. Records go to an InMemorySink; nothing is persisted.
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.business import load_business_config  # noqa: E402
from agent.dialogue import DialogueEngine, EndCall, Say, ToolResult  # noqa: E402
from agent.llm import OpenAICompatibleLLMClient  # noqa: E402
from agent.prompt import build_system_prompt  # noqa: E402
from agent.records import InMemorySink  # noqa: E402
from agent.settings import get_settings  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

CALLER_PHONE = "+79991234567"
OPENING = "Здравствуйте, хочу записаться на полировку кузова."
MAX_TURNS = 14

# (pattern in the agent's LAST sentence, what the caller says). First match wins.
CALLER_RULES = [
    (r"верно|правильно|подтвержда|всё так", "Да, всё верно."),
    (r"зовут|\bимя\b|имени", "Меня зовут Игорь."),
    (r"этот номер|на номер|телефон|номер|цифр", "Нет, запишите другой: 8 916 123 45 67."),
    (r"марк|модел|автомобил|машин", "Тойота Камри."),
    (r"услуг", "Полировка кузова."),
    (r"\bдат|когда|какой день|удобн|врем", "В субботу после обеда."),
]
GOODBYE = "Спасибо, до свидания."


def caller_reply(agent_sentences: list[str], booked: bool) -> str:
    if booked:
        return GOODBYE
    lowered = agent_sentences[-1].lower() if agent_sentences else ""
    for pattern, answer in CALLER_RULES:
        if re.search(pattern, lowered):
            return answer
    return "Да."


async def main() -> int:
    settings = get_settings()
    business = load_business_config(settings.business_config_path)
    now = datetime.now(settings.tz)
    sink = InMemorySink()

    llm = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    tools = ToolRegistry(
        business, sink, clock=lambda: datetime.now(settings.tz), caller_phone=CALLER_PHONE
    )
    engine = DialogueEngine(llm, tools, build_system_prompt(business, now, CALLER_PHONE))

    print(f"model: {settings.llm_model}   now: {now:%A %Y-%m-%d %H:%M} ({settings.timezone})")
    print(f"caller number: {CALLER_PHONE}\n")
    print(f"АГЕНТ: {business.greeting}")

    user_text, ended = OPENING, False
    for _ in range(MAX_TURNS):
        print(f"\nКЛИЕНТ: {user_text}")
        started = time.monotonic()
        said: list[str] = []
        async for event in engine.respond(user_text):
            if isinstance(event, Say):
                print(f"АГЕНТ: {event.text}")
                said.append(event.text)
            elif isinstance(event, ToolResult):
                args = json.loads(event.call.arguments or "{}")
                print(f"  [tool] {event.call.name}({json.dumps(args, ensure_ascii=False)})")
                print(f"  [tool result] {event.result}")
            elif isinstance(event, EndCall):
                print("  [end_call] agent hangs up")
                ended = True
        print(f"  ({time.monotonic() - started:.1f}s)")
        if ended:
            break
        if not said and not sink.bookings:
            print("  !! empty agent reply")
        user_text = caller_reply(said, booked=bool(sink.bookings))
    else:
        print(f"\n(stopped after {MAX_TURNS} turns without hanging up)")

    await llm.aclose()

    print("\n=== Saved records ===")
    for booking in sink.bookings:
        print(json.dumps(_plain(booking), ensure_ascii=False, indent=2))
    for message in sink.messages:
        print(json.dumps(_plain(message), ensure_ascii=False, indent=2))
    if not sink.bookings and not sink.messages:
        print("(none)")
    print(f"\nbooking completed: {bool(sink.bookings)}   call ended by agent: {ended}")
    return 0 if sink.bookings else 1


def _plain(record) -> dict:
    return {k: (str(v) if v is not None else None) for k, v in vars(record).items()}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
