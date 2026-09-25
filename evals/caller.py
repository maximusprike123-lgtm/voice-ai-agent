"""The simulated caller: an LLM playing a customer with a persona and a goal.

It sees the conversation from the customer's side (what the agent SAID, never its tool calls),
answers in any order the agent asks, and ends its last line with a marker when it is done.
"""

import re

from agent.llm import LLMClient, LLMError, Message, Role, TextDelta
from evals.model import Scenario

END_MARKER = "[КОНЕЦ]"
FALLBACK_FAREWELL = "До свидания."

PROMPT_TEMPLATE = """\
Ты играешь клиента, который звонит по телефону в детейлинг-центр (автомойка и уход за \
автомобилем). Ты человек, которому нужна услуга. Ты не администратор и не помощник.

Как говорить:
- по-русски, как в живом телефонном разговоре: одна-две короткие фразы, без списков, \
markdown и эмодзи;
- отвечай только на то, о чём тебя спросили; не выдавай все данные сразу и не повторяй то, \
что уже сказал;
- числа и телефон говори цифрами, как в обычной речи (например: 8 916 123 45 67);
- не подсказывай администратору, как вести разговор, и не описывай систему записи.

Кто ты и чего хочешь:
{persona}

Что ты о себе знаешь (используй, когда спросят):
{facts}

Как вести себя в ситуациях:
{behavior}

Когда твоя цель достигнута, ты отказался или положил трубку, попрощайся и допиши в конце \
реплики {marker}. Пока разговор продолжается, {marker} не пиши.
Отвечай ТОЛЬКО репликой клиента: без пояснений, без кавычек и без имени говорящего.
"""


class CallerError(Exception):
    """The simulated caller could not produce a line (the run is inconclusive, not a failure)."""


def build_persona_prompt(scenario: Scenario) -> str:
    return PROMPT_TEMPLATE.format(
        persona=scenario.persona.strip(),
        facts=scenario.facts.strip(),
        behavior=scenario.behavior.strip(),
        marker=END_MARKER,
    )


def clean_reply(raw: str) -> tuple[str, bool]:
    """(the spoken line, whether the caller is done). Strips speaker labels and quotes."""
    text = raw.strip()
    done = END_MARKER in text or "[КОНЕЦ" in text
    text = re.sub(r"\[КОНЕЦ\]?", "", text)
    text = re.sub(r"^\s*(клиент|игорь|caller)\s*[:\-—]\s*", "", text, flags=re.I)
    text = text.strip().strip("«»\"'").strip()
    if done and not text:
        text = FALLBACK_FAREWELL
    return text, done


class SimulatedCaller:
    def __init__(self, llm: LLMClient, scenario: Scenario) -> None:
        self._llm = llm
        self._messages = [Message(Role.SYSTEM, build_persona_prompt(scenario))]

    async def next_utterance(self, agent_said: str) -> tuple[str, bool]:
        """What the caller says next, given what the agent just said; and whether they are done."""
        self._messages.append(Message(Role.USER, agent_said))
        raw = ""
        for attempt in (1, 2):
            try:
                raw = await self._generate()
            except LLMError as exc:
                if attempt == 2:
                    raise CallerError(f"caller LLM failed: {exc}") from exc
                continue
            if raw.strip():
                break
        else:
            raise CallerError("caller LLM returned an empty reply twice")
        text, done = clean_reply(raw)
        if not text:
            raise CallerError(f"caller reply was empty after cleaning: {raw!r}")
        self._messages.append(Message(Role.ASSISTANT, raw.strip()))
        return text, done

    async def _generate(self) -> str:
        parts = []
        async for event in self._llm.stream(self._messages):
            if isinstance(event, TextDelta):
                parts.append(event.text)
        return "".join(parts)
