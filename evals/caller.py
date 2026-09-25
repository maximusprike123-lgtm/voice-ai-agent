"""The simulated caller: an LLM playing a customer with a persona and a goal.

It sees the conversation from the customer's side (what the agent SAID, never its tool calls),
answers in any order the agent asks, and ends its last line with a marker when it is done.
"""

import re

from agent.llm import LLMClient, LLMError, Message, Role, TextDelta
from evals.model import Scenario

END_MARKER = "[КОНЕЦ]"
FALLBACK_FAREWELL = "До свидания."
FAREWELL_RE = re.compile(
    r"до свидани|всего доброго|всего хорошего|хорошего дня|доброго дня|всех благ|до встречи"
    r"|прощайте|счастливо|бывайте|\bпока\b",
    re.I,
)


def is_farewell(text: str) -> bool:
    """Does this line say goodbye? Only such a line may end the call."""
    return bool(FAREWELL_RE.search(text))


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

Когда твоя цель достигнута, ты отказался или решил положить трубку, скажи «до свидания» и \
допиши в конце этой реплики {marker}. {marker} пишется ТОЛЬКО вместе с прощанием и только в \
самой последней реплике. Во всех остальных репликах, даже если ты согласился, подтвердил \
или поблагодарил, {marker} не пиши: разговор ещё не закончен, пока ты не попрощался.
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
    """(the spoken line, whether the caller hangs up after it). Strips speaker labels, quotes
    and the end marker. The marker counts ONLY on a farewell line: a caller that writes it
    while still mid-conversation («Да, всё верно. [КОНЕЦ]») is simply carried on."""
    text, hung_up, _ = parse_reply(raw)
    return text, hung_up


def parse_reply(raw: str) -> tuple[str, bool, bool]:
    """(line, hangs up, marker was ignored)."""
    text = raw.strip()
    had_marker = END_MARKER in text or "[КОНЕЦ" in text
    text = re.sub(r"\[КОНЕЦ\]?", "", text)
    text = re.sub(r"^\s*(клиент|игорь|caller)\s*[:\-—]\s*", "", text, flags=re.I)
    text = text.strip().strip("«»\"'").strip()
    if had_marker and not text:
        text = FALLBACK_FAREWELL
    hangs_up = had_marker and is_farewell(text)
    return text, hangs_up, had_marker and not hangs_up


class SimulatedCaller:
    def __init__(self, llm: LLMClient, scenario: Scenario) -> None:
        self._llm = llm
        self._messages = [Message(Role.SYSTEM, build_persona_prompt(scenario))]
        self.markers_ignored = 0  # hang-up markers written without a farewell

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
        text, done, ignored = parse_reply(raw)
        self.markers_ignored += ignored
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
