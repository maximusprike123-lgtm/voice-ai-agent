"""Composition root: builds the long-lived parts from Settings and hands out one CallSession
per call. The CLI uses it now; the call handler of step 3 will use the same thing.

    async with open_runtime(settings, notify=True) as runtime:
        session = runtime.new_call(caller_phone="+79991234567")
        ...

Ownership: `open_runtime` closes everything it created on exit (drains and stops the notifier,
then the notifier's HTTP client, then the database). Things passed in (`llm`, `notifier`) are
used but not closed, since the caller owns them.
"""

import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.business import BusinessConfig, load_business_config
from agent.dialogue import DialogueEngine
from agent.llm import LLMClient, OpenAICompatibleLLMClient
from agent.notifier import Notifier, NotifyingSink, TelegramNotifier
from agent.prompt import build_system_prompt
from agent.records import RecordSink
from agent.session import CallSession
from agent.settings import Settings
from agent.storage import SqliteSink
from agent.tools import ToolRegistry

Clock = Callable[[], datetime]


def offset_clock(start: datetime, real_clock: Clock) -> Clock:
    """A clock that reads `start` now and keeps ticking in real time (for `--now`)."""
    delta = start - real_clock()
    return lambda: real_clock() + delta


@dataclass
class Runtime:
    settings: Settings
    business: BusinessConfig
    llm: LLMClient
    store: SqliteSink  # the database (source of truth)
    sink: RecordSink  # what the tools write to: the store, or the store + notifications
    clock: Clock
    resent_at_start: int = 0  # older unnotified records queued when notifications started

    def new_call(self, caller_phone: str | None = None) -> CallSession:
        """A fresh conversation: its own prompt (date, time, caller), tools and history."""
        now = self.clock()
        engine = DialogueEngine(
            self.llm,
            ToolRegistry(self.business, self.sink, self.clock, caller_phone),
            build_system_prompt(self.business, now, caller_phone),
            first_event_timeout=self.settings.llm_first_event_timeout_seconds,
            event_timeout=self.settings.llm_event_timeout_seconds,
            speech_guard=self.settings.speech_guard,
        )
        return CallSession(engine, self.business.greeting, self.sink, self.clock, caller_phone)


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    notify: bool = False,
    db_path: Path | None = None,
    clock: Clock | None = None,
    llm: LLMClient | None = None,
    notifier: Notifier | None = None,
) -> AsyncIterator[Runtime]:
    clock = clock or (lambda: datetime.now(settings.tz))
    business = load_business_config(settings.business_config_path)
    store = await SqliteSink.open(db_path or settings.db_path)

    owned_notifier: TelegramNotifier | None = None
    owned_llm: OpenAICompatibleLLMClient | None = None
    notifying: NotifyingSink | None = None
    try:
        if llm is None:
            llm = owned_llm = OpenAICompatibleLLMClient(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key.get_secret_value(),
                model=settings.llm_model,
                reasoning_effort=settings.llm_reasoning_effort,
                extra_body=settings.llm_extra_body,
                timeout_seconds=settings.llm_timeout_seconds,
            )

        sink: RecordSink = store
        resent = 0
        if notify:
            if notifier is None:
                notifier = owned_notifier = TelegramNotifier(
                    settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id
                )
            sink = notifying = NotifyingSink(store, notifier)
            resent = await notifying.start()

        yield Runtime(settings, business, llm, store, sink, clock, resent)
    finally:
        with contextlib.suppress(Exception):
            if notifying is not None:
                await notifying.aclose()
        if owned_notifier is not None:
            await owned_notifier.aclose()
        if owned_llm is not None:
            await owned_llm.aclose()
        await store.close()
