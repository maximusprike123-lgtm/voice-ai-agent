"""CallSession: one phone call's conversation, with the failure policy around DialogueEngine.

The CLI drives it now; the audio pipeline (step 3) will drive it the same way, so everything
that must be true of every call lives here and not in a front end:
  - the greeting is spoken first (without the LLM) and is in the history;
  - when the LLM fails (stall, error, empty reply) even after the engine's own retry, the
    caller hears a short fallback instead of silence: the first time «повторите, пожалуйста»
    (the failed turn is rolled back, so they simply say it again); the second time in a row an
    apology, the call ends, and a callback message with the caller's last words is saved, so a
    technical failure never silently loses a caller.
"""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime

from agent.dialogue import (
    DialogueEngine,
    DialogueError,
    DialogueEvent,
    EndCall,
    Say,
    ToolResult,
    split_sentences,
)
from agent.llm import LLMError
from agent.records import CallbackMessage, RecordSink

logger = logging.getLogger(__name__)

# NOTE: these use masculine forms («не расслышал»); they must match the gender of the TTS
# voice chosen in step 2.
ASK_TO_REPEAT = "Простите, я не расслышал. Повторите, пожалуйста."
FINAL_APOLOGY = (
    "Извините, у нас возникли технические неполадки. Администратор перезвонит вам. До свидания."
)

# Said when a turn fails AFTER a tool saved something (a booking, a message): the caller must
# never hear «не расслышал» about a request that was in fact taken. Neutral wording; it does not
# say which request, so it is safe whatever tool committed.
COMMITTED_FALLBACK = (
    "Ваша просьба принята и передана администратору, он свяжется с вами. Могу ещё чем-то помочь?"
)

MAX_FAILED_TURNS_IN_A_ROW = 2
CALLBACK_UTTERANCES = 3  # how many of the caller's last lines go into the callback message
MAX_UTTERANCE_CHARS = 300


@dataclass(frozen=True)
class TurnFailed:
    """Not spoken: tells the front end (and the logs) why the fallback phrase follows."""

    reason: str
    consecutive: int


SessionEvent = DialogueEvent | TurnFailed


class CallSession:
    def __init__(
        self,
        engine: DialogueEngine,
        greeting: str,
        sink: RecordSink,
        clock: Callable[[], datetime],
        caller_phone: str | None = None,
        max_failed_turns: int = MAX_FAILED_TURNS_IN_A_ROW,
    ) -> None:
        self._engine = engine
        self._greeting = greeting
        self._sink = sink
        self._clock = clock
        self._caller_phone = caller_phone
        self._max_failed_turns = max_failed_turns
        self._failed_in_a_row = 0
        self._utterances: list[str] = []
        self.ended = False

    def greet(self) -> list[Say]:
        """The opening line, as sentences to speak. Also tells the model it was said."""
        self._engine.add_assistant_message(self._greeting)
        return [Say(sentence) for sentence in split_sentences(self._greeting)]

    async def handle(self, user_text: str) -> AsyncIterator[SessionEvent]:
        """One caller utterance in, the agent's reaction out (fallback phrases included)."""
        self._utterances.append(user_text)
        committed = False  # did a tool save something during this turn?
        try:
            async for event in self._engine.respond(user_text):
                if isinstance(event, EndCall):
                    self.ended = True
                elif isinstance(event, ToolResult) and event.committed:
                    committed = True
                yield event
        except (LLMError, DialogueError) as exc:
            if committed:
                # The work is done and saved; only the model's follow-up failed. Say so in
                # code instead of asking the caller to repeat, and don't count it as a failed
                # turn: the caller got a correct answer.
                logger.warning("turn failed after a tool committed (%s)", exc)
                yield TurnFailed(str(exc), self._failed_in_a_row)
                self._failed_in_a_row = 0
                self._engine.add_assistant_message(COMMITTED_FALLBACK)
                for sentence in split_sentences(COMMITTED_FALLBACK):
                    yield Say(sentence)
                return
            self._failed_in_a_row += 1
            logger.warning("turn failed (%d in a row): %s", self._failed_in_a_row, exc)
            yield TurnFailed(str(exc), self._failed_in_a_row)
            if self._failed_in_a_row < self._max_failed_turns:
                yield Say(ASK_TO_REPEAT)
                return
            await self._save_callback()
            self.ended = True
            for sentence in split_sentences(FINAL_APOLOGY):
                yield Say(sentence)
            yield EndCall()
        else:
            self._failed_in_a_row = 0

    async def _save_callback(self) -> None:
        last = [u[:MAX_UTTERANCE_CHARS] for u in self._utterances[-CALLBACK_UTTERANCES:]]
        quoted = "; ".join(f"«{line}»" for line in last)
        message = CallbackMessage(
            message=(
                "Звонок оборвался из-за технической неполадки, клиент не получил ответа. "
                f"Последние слова клиента: {quoted}. Нужно перезвонить."
            ),
            name=None,
            phone=None,
            caller_phone=self._caller_phone,
            created_at=self._clock(),
        )
        try:
            await self._sink.add_message(message)
        except Exception:
            logger.exception("could not save the callback message after a failed call")
