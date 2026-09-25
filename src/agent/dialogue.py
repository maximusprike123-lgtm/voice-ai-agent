"""DialogueEngine: audio-agnostic conversation loop.

Text in (what the caller said), events out (what to say, which tools ran, when to hang up).
It knows nothing about audio, STT/TTS, or which LLM vendor is behind `LLMClient`, so the same
engine runs in a CLI, with a local mic, and on real phone calls.

Cancellation (barge-in): `respond()` is an async generator. Cancelling the task that consumes
it, or calling `aclose()` on it, stops the LLM stream and leaves the history consistent (no
tool call without a result). Sentences already emitted stay in the history as the assistant's
reply; trimming history to what the caller actually heard is a step-3 concern.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

from agent.llm import (
    LLMClient,
    LLMError,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)

logger = logging.getLogger(__name__)

# Safety cap on LLM<->tools round trips within one user turn (guards against a looping model).
MAX_LLM_ROUNDS = 5

# A "sentence" with fewer letters/digits than this ("Да.", "Ок.") is merged into the next one,
# so TTS is never handed a lone one-word fragment.
MIN_SENTENCE_CHARS = 4

INTERRUPTED_TOOL_RESULT = "Вызов прерван, результат неизвестен."
TOOL_FAILED_RESULT = "Ошибка при выполнении инструмента."


class DialogueError(Exception):
    """Raised when the dialogue cannot continue (not an LLM transport failure)."""


# --- Events out -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Say:
    """One complete sentence of the agent's reply, ready to be spoken."""

    text: str


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    result: str


@dataclass(frozen=True)
class EndCall:
    """The agent is done: say goodbye (already emitted as Say) and hang up."""


DialogueEvent = Say | ToolResult | EndCall


# --- Tools (implemented in step 1.6) ----------------------------------------------------------


@dataclass(frozen=True)
class ToolOutcome:
    result: str  # text handed back to the model as the tool message
    ends_call: bool = False
    # Text the engine speaks to the caller verbatim, bypassing the model. When any tool call of
    # a round has it, the turn ends there: the text goes into the history as an assistant
    # message and the engine waits for the caller instead of calling the LLM again.
    say: str | None = None


class ToolExecutor(Protocol):
    specs: list[ToolSpec]

    async def execute(self, call: ToolCall) -> ToolOutcome:
        """Run one tool call. Should report failures in `result` rather than raise."""
        ...


# --- Sentence splitting -----------------------------------------------------------------------

# A sentence ends at ./!/?/… (plus closing quotes/brackets) followed by whitespace, or at a
# newline. Requiring the whitespace means "3.5" and a terminator still waiting for its next
# token are not split.
_BOUNDARY = re.compile(r"[.!?…]+[»\"')]*(?=\s)|\n")


class SentenceSplitter:
    """Turns a stream of text chunks into complete sentences."""

    def __init__(self) -> None:
        self._buffer = ""
        self._carry = ""  # a too-short sentence waiting to be merged into the next one

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []
        while match := _BOUNDARY.search(self._buffer):
            piece, self._buffer = self._buffer[: match.end()], self._buffer[match.end() :]
            if sentence := self._take(piece):
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        """Return whatever is left at the end of the stream (may lack a final terminator)."""
        rest = f"{self._carry} {self._buffer.strip()}".strip()
        self._buffer = self._carry = ""
        return rest or None

    def _take(self, piece: str) -> str | None:
        candidate = f"{self._carry} {piece.strip()}".strip()
        if not candidate:
            return None
        if sum(ch.isalnum() for ch in candidate) < MIN_SENTENCE_CHARS:
            self._carry = candidate
            return None
        self._carry = ""
        return candidate


def split_sentences(text: str) -> list[str]:
    """Split a complete text into the sentence units TTS speaks."""
    splitter = SentenceSplitter()
    sentences = splitter.feed(text)
    if tail := splitter.flush():
        sentences.append(tail)
    return sentences


# --- Engine -------------------------------------------------------------------------------------


class DialogueEngine:
    def __init__(self, llm: LLMClient, tools: ToolExecutor, system_prompt: str) -> None:
        self._llm = llm
        self._tools = tools
        self._messages: list[Message] = [Message(Role.SYSTEM, system_prompt)]

    @property
    def messages(self) -> list[Message]:
        """A copy of the conversation history, system prompt first."""
        return list(self._messages)

    async def respond(self, user_text: str) -> AsyncIterator[DialogueEvent]:
        """Handle one caller utterance and stream the agent's reaction.

        Raises LLMError if the backend fails. If that happens before anything was added to the
        history besides the user message, the user message is dropped too, so the caller can
        simply retry the same utterance.
        """
        checkpoint = len(self._messages)
        self._messages.append(Message(Role.USER, user_text))

        for round_index in range(MAX_LLM_ROUNDS):
            spoken: list[str] = []
            calls: list[ToolCall] = []
            splitter = SentenceSplitter()
            specs = self._tools.specs or None

            try:
                async with aclosing(self._llm.stream(self._messages, specs)) as stream:
                    async for event in stream:
                        if isinstance(event, TextDelta):
                            for sentence in splitter.feed(event.text):
                                spoken.append(sentence)
                                yield Say(sentence)
                        elif isinstance(event, ToolCallEvent):
                            calls.append(event.call)
                    if tail := splitter.flush():
                        spoken.append(tail)
                        yield Say(tail)
            except LLMError:
                if round_index == 0:
                    del self._messages[checkpoint:]
                raise
            except (asyncio.CancelledError, GeneratorExit):
                if spoken:
                    self._messages.append(Message(Role.ASSISTANT, " ".join(spoken)))
                raise

            if spoken or calls:
                self._messages.append(
                    Message(
                        Role.ASSISTANT,
                        content=" ".join(spoken) or None,
                        tool_calls=tuple(calls) or None,
                    )
                )
            if not calls:
                return

            ends_call = False
            scripted: list[str] = []  # ToolOutcome.say texts, spoken as-is after the tools ran
            done = 0
            try:
                for call in calls:
                    outcome = await self._execute(call)
                    self._messages.append(Message(Role.TOOL, outcome.result, tool_call_id=call.id))
                    done += 1
                    ends_call = ends_call or outcome.ends_call
                    if outcome.say:
                        scripted.append(outcome.say)
                    yield ToolResult(call, outcome.result)
            except (asyncio.CancelledError, GeneratorExit):
                # Never leave a tool call in the history without a matching result.
                for call in calls[done:]:
                    self._messages.append(
                        Message(Role.TOOL, INTERRUPTED_TOOL_RESULT, tool_call_id=call.id)
                    )
                raise

            if scripted:
                text = " ".join(scripted)
                # In the history before it is spoken, so a barge-in mid-sentence still leaves
                # the model knowing what the caller was being asked.
                self._messages.append(Message(Role.ASSISTANT, text))
                for sentence in split_sentences(text):
                    yield Say(sentence)
                if ends_call:
                    yield EndCall()
                return  # wait for the caller: no further LLM round
            if ends_call:
                yield EndCall()
                return

        raise DialogueError(f"model still requesting tools after {MAX_LLM_ROUNDS} rounds")

    async def _execute(self, call: ToolCall) -> ToolOutcome:
        try:
            return await self._tools.execute(call)
        except Exception:
            logger.exception("tool %s raised", call.name)
            return ToolOutcome(TOOL_FAILED_RESULT)
