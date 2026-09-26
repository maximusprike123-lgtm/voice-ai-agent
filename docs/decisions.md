# Decisions and step log (steps 1.1–1.9)

Moved out of CLAUDE.md verbatim. CLAUDE.md keeps a one-line version of each decision; this file has
the full text with the reasons and measurements. Eval harness / speech guard history is in
[evals.md](evals.md), latency and provider findings in [latency.md](latency.md), open issues in
[open-issues.md](open-issues.md).

## Architecture decisions (and why)

- **Telephony path:** 1ATC (SIP, Russian number, real credentials, foreign IPs blocked) →
  Asterisk in Docker on a Russian VPS → **AudioSocket** (plain TCP, 8kHz/16-bit PCM) → our
  Python app. Asterisk owns all SIP/RTP; **our Python code never implements SIP/RTP itself.**
- **No Pipecat, no LiveKit.** Checked: Pipecat has no built-in AudioSocket transport (the
  community `pipecat-asterisk` package uses Asterisk's chan_websocket instead, not
  AudioSocket) and no built-in Yandex SpeechKit service. Given AudioSocket is a simple
  protocol, we write a minimal asyncio pipeline ourselves (VAD, streaming STT, LLM,
  streaming TTS, barge-in) rather than half-use a framework that doesn't fit the transport.
- **`DialogueEngine` is audio-agnostic.** It only sees text in, and streams text/tool
  events out. This lets the same engine run in a CLI (step 1), with a local mic (step 2),
  and on real phone calls (step 3) unchanged.
- **`LLMClient`, `STTClient`, `TTSClient` are vendor-agnostic interfaces.** Nothing outside
  the client module may depend on a specific vendor's behavior. This is a hard requirement:
  the vendor choice may change later.
  - **LLM:** generic OpenAI-compatible Chat Completions API (`LLM_BASE_URL`, `LLM_API_KEY`,
    `LLM_MODEL`). Currently pointed at **OpenRouter, `deepseek/deepseek-v4.1-flash`** (pinned
    version, never a `-latest` alias, so behavior can't change under us). **Local Ollama
    `qwen3.5:4b` is kept as the offline backup** (commented-out block in `.env`; 8GB Mac, see
    open issues for why it isn't the main backend). Switching = editing `.env` only.
    - **Thinking must be OFF:** `LLM_REASONING_EFFORT=none`, an optional passthrough field in
      `LLMClient` (top-level `reasoning_effort`), omitted entirely when unset so a
      non-reasoning backend is never sent a field it might reject. Verified on both backends
      by completion-token counts, not by looking at the text: Ollama qwen3.5 705 → 4 tokens;
      OpenRouter DeepSeek 64 (59 reasoning) → 4 (0). On OpenRouter `minimal`/`low` do NOT turn
      thinking off; `none` (or `reasoning: {enabled: false}`) does. `LLMClient` also drops any
      `reasoning`/`reasoning_content` delta fields a backend might stream, so thinking text
      can never leak into speech. But a silent thinker still costs tokens and latency, which
      is why `check_llm.py` checks the backend's own `usage` numbers (see Commands).
    - **Optional `extra_body` passthrough** (`LLM_EXTRA_BODY`, a JSON object; `None` = nothing
      added): merged into every request body, for backend-specific options such as OpenRouter
      provider routing (`{"provider": {"order": [...], "allow_fallbacks": true}}`). It may not
      override the fields the client owns (`model`, `messages`, `stream`, `tools`,
      `reasoning_effort`). Verified live end to end. LLMClient defaults are unchanged.
  - **STT/TTS:** vendor not chosen yet (candidates: ElevenLabs, Deepgram). Foreign services
    are acceptable for the MVP despite the RU-only telephony/LLM hosting constraint — this
    may need a network path check once a vendor is picked.
- **Principle: code enforces guarantees, the LLM handles conversation.** Anything the business
  relies on must not depend on the model obeying the prompt. Examples so far: booking data is
  validated and normalized in code; a booking is saved only by `confirm_booking` after code
  has read the draft back; the read-back text is built by code and spoken verbatim by the
  engine (`ToolOutcome.say`), so its phone digits are always exactly the last 4 and its
  numbers/dates/times are always in words; the model is never asked to say digits; an
  approximate time cannot become an invented HH:MM because it goes into `preferred_period`; the
  caller hears that a booking/message was accepted from a code-built sentence
  (`confirm_booking` / `take_message` return `say`), never from a model round that could stall;
  and **a speech guard checks every sentence the model writes before it can be spoken** (phone
  digits, false acceptance claims, «записал» before a save, foreign script).
  When a live run shows the model slipping on something that matters, move it into code
  rather than into more prompt text. The LLM keeps what is conversational: understanding the
  caller, phrasing questions, answering FAQ.
- **Knowledge:** `config/business.yaml` (hours, services, prices, FAQ) loaded straight into
  the system prompt. No RAG — the FAQ is small and fixed.
- **Bookings:** validated, then **written to SQLite before** the Telegram notification is
  sent, so nothing is lost if Telegram fails. The agent never confirms a time slot itself.
  SQLite is also the outbox: every row has `notified_at` (NULL until the owner was told).
  **Notifications never block the conversation:** `NotifyingSink` saves, returns the row id
  at once, and one background worker sends to Telegram and then marks the row. Unnotified rows
  are re-sent at startup and by a sweep every 5 minutes.
  **Delivery is at-least-once, not exactly-once:** if the process dies (or the mark fails)
  between a successful send and marking the row, the message is sent again on the next
  start/sweep, so the owner may occasionally see a duplicate. Accepted on purpose: a duplicate
  is harmless, a lost booking is not. Don't "fix" it without a plan for the lost-message case.
- **Scope:** inbound calls only, Russian only, no call transfer, no LangVerse code reused.

## Status paragraphs (as written when step 1 was completed)

**STEP 1 (the text agent, roadmap items 1.1–1.10) IS COMPLETE — tag `v0.1-text-agent`.** What exists:
a Russian-language, audio-agnostic booking agent (DialogueEngine + CallSession + tools + SQLite +
Telegram notifier + CLI) with code-enforced guarantees (two-step booking, code-built read-back and
acceptance, `end_call` and same-turn-confirm guards, speech guard) and an evaluation harness
(`python -m evals`) with a measured baseline: **95/99 = 96%** raw pass rate on 10 runs per scenario
(old agent 76/95 = 80%). Step 2 (STT/TTS, local mic) has NOT been started. Post-tag follow-ups, both
done: the `role_leakage` speech-guard rule and two relaxed eval checks (see the end of this
section). The detailed step log follows.

## Step log: 1.1–1.9

**Done:** 1.1 (project skeleton, settings), 1.2 (business.yaml schema + loader), 1.3
(Russian system prompt builder, with a 14-day calendar so "в пятницу" resolves to a real
date), 1.4 (`LLMClient` + `OpenAICompatibleLLMClient`, offline tests, live check script),
1.5 (`DialogueEngine` in `src/agent/dialogue.py`: history, LLM→tools→LLM loop capped at 5
rounds, `SentenceSplitter`, `Say`/`ToolResult`/`EndCall` events, cancellable; tools come in
through a `ToolExecutor` protocol that 1.6 implements; `scripts/measure_ttft.py` times the
real prompt through the engine). `mark_spoken` (trim history to what the caller heard) is
deferred to step 3. Follow-up: `build_system_prompt()` now puts all static content first and
the per-call content (time, caller number, 14-day calendar) last, after `VOLATILE_MARKER`;
a test pins the shared prefix. Tool schemas (1.6) must stay static too: no dates, caller data
or other per-call content in tool names/descriptions/parameters.

1.6 / 1.6b (tools, `src/agent/tools.py` + `records.py` + `ru_words.py`; `ToolRegistry`
implements `ToolExecutor`). Booking is **two-step**: `prepare_booking(...)` validates and
normalizes (phone → `+7XXXXXXXXXX`; date required, not in the past, ≤ 90 days ahead, on a
working day; `preferred_time` HH:MM only for an exact time, checked against that day's hours
and not already past today; `preferred_period` ∈ утро/день/вечер/любое; at least one of the
two required, both allowed and both stored, exact time wins in the read-back; `service_id`
from business.yaml or the reserved `other` — `notes` then required; caller's own words go to
`notes`), stores a draft for this call (a new or failed prepare discards the old draft) and
returns `ToolOutcome.say` = a read-back built by code («Проверьте, пожалуйста: … Номер
телефона заканчивается на <ровно 4 цифры словами>. Всё верно?»). `confirm_booking()` (no
args) re-validates the draft against the clock, saves it via the `RecordSink`
(`InMemorySink` for now; a sink failure keeps the draft so confirm can be retried), and has a
duplicate guard per call. Errors go back to the model as `ОШИБКА: …`. Success text says
"don't name a price" only for `other`. `take_message`, `end_call`. Schemas are static. The
old `submit_booking` is gone. `business.yaml` may not use the id `other` (fails at load).
**`end_call` guard:** the engine calls `ToolExecutor.begin_turn()` once per caller utterance;
`end_call` returns `ОШИБКА` if a booking was accepted in the same turn (the caller hasn't heard
it yet), so the model must ask «нужна ли помощь ещё?» and wait; it also covers a duplicate
confirm. The prompt forbids «записал/записала/записано» before `confirm_booking` succeeds
(use «хорошо», «принято»). Live check: the model asked and waited.
**Engine:** `ToolOutcome.say` — when a tool round has it, the engine speaks the text as `Say`
sentences, appends it to the history as an assistant message and ends the turn with no
further LLM call (it waits for the caller). `ru_words.py` holds the reusable Russian
number/date/time word tables and functions (for `text_normalize.py` later). Prompt (static
part): read nothing back yourself, `confirm_booking` only after a clear «да», the model
never says phone digits (asks «Записать вас на номер, с которого вы звоните?»; without
caller ID it asks the caller to dictate a number), approximate period → `preferred_period`,
don't re-ask the time. `scripts/live_booking_dialogue.py` runs a scripted caller against the
real LLM.

1.7 (SQLite, `src/agent/storage.py`): `SqliteSink` implements `RecordSink` on stdlib
`sqlite3` (one shared connection, every call in `asyncio.to_thread` under a lock); WAL +
`synchronous=FULL`, so once `add_*` returns the row is on disk; file mode 0600 (dir 0700,
loose modes on an existing file are tightened); `PRAGMA user_version` + an append-only
`_MIGRATIONS` tuple (a newer-than-known database is refused, untouched; step 3 adds `call_id`
as migration 2). Tables `bookings` and `messages` with a nullable `notified_at` outbox
column. `RecordSink.add_booking/add_message` now return the row id (per-table ids;
`InMemorySink` too). API: `SqliteSink.open(path)` / `close()` / `async with`, `get_booking`,
`list_bookings`, `list_messages`, `list_unnotified()` (merged, oldest first),
`mark_booking_notified(id)` / `mark_message_notified(id)` (idempotent, keeps the first
timestamp). Failures raise `StorageError`; the tools already turn that into an apology plus
`ОШИБКА` and keep the draft so `confirm_booking` can be retried. `Settings.db_path` exists
(`data/agent.db`, gitignored) but nothing opens the database yet: the CLI (1.9) / call
handler (step 3) will.

1.8 (Telegram notifier, `src/agent/notifier.py`): `TelegramNotifier` (`sendMessage`, JSON
`chat_id` + `text`, **no `parse_mode`** so caller text can never break a message; timeouts
3s connect / 5s read+write; text capped at Telegram's 4096; control characters stripped;
errors are `NotifyError(permanent, retry_after)`: 429 → transient with Telegram's
`retry_after` (body or header), 5xx/timeouts/network/odd 200 → transient, other 4xx →
permanent). `NotifyingSink` implements `RecordSink` over an outbox store (`SqliteSink`): save
(a save failure propagates, nothing is sent) → enqueue → return id; a single worker keeps
order and stays under Telegram's per-chat rate limit; retries: 5 attempts per record per round,
backoff 1/2/4/8s (+-25% jitter, cap 30s), a 429 waits `retry_after`+0.5s, a `retry_after` over
120s is left to the sweep; permanent errors are logged at ERROR and not retried; a record that
fails its whole budget stays unnotified for the sweep; `start()` resends `list_unnotified()`
then starts the worker and the 5-minute sweep; `aclose()` drains for up to 10s, then cancels
(unsent rows stay in the DB). Message texts are built by `format_booking` /
`format_message` (plain Russian text). **Token hygiene:** the bot token is in the request URL,
so `NotifyError`s are raised outside `except` blocks (never chained to httpx errors) and httpx's
own INFO log line ("HTTP Request: POST …/bot<TOKEN>/sendMessage") is redacted by a logging
filter on the `httpx` logger — found by a test, it *did* leak the token at the default INFO
level. `scripts/send_test_notification.py` sends one marked test message through the whole
real path (temporary SQLite + `NotifyingSink` + Telegram); verified live: delivered, marked
notified, token absent from all output. Nothing wires `NotifyingSink` into
an app yet: the CLI (1.9) / call handler (step 3) will build store → notifier → sink and call
`start()`/`aclose()`.

1.9 follow-up (found by the live CLI run): **the caller is never told the outcome of a save by
the LLM.** `ToolOutcome` gained `committed` (a record was saved; also on `ToolResult`).
`confirm_booking` success returns a code-built `say`: «Заявка принята и передана администратору,
он перезвонит для подтверждения. [Точное время администратор уточнит при звонке.] Могу ещё
чем-то помочь?» (a duplicate confirm has its own sentence); `take_message` success: «Сообщение
передано администратору, он свяжется с вами. Могу ещё чем-то помочь?» (all neutral, no
gendered verbs). The engine speaks it and ends the turn with no further LLM call, which also
saves a round trip; the prompt tells the model not to add anything. **Safety net** in
`CallSession`: if a turn fails after any tool with `committed=True` succeeded, it says
`COMMITTED_FALLBACK` («Ваша просьба принята и передана администратору, он свяжется с вами. Могу
ещё чем-то помочь?»), adds it to the history, and does not count the turn as a failed turn,
never «не расслышал». `LLMClient` got the optional `extra_body` passthrough (above);
`scripts/latency_study.py` measures TTFT per routing configuration (results in Known open
issues).

Backend switch (after 1.6): main LLM is now OpenRouter DeepSeek (see Architecture);
`check_llm.py` gained the token-based "no hidden reasoning" check (it fails if
`reasoning_tokens > 0` or a one-word answer costs > 20 completion tokens; verified to FAIL with
`LLM_REASONING_EFFORT=high`); `measure_ttft.py` skips its Ollama-only cold-start run on
remote backends.

1.9 (CLI + failure handling): **Engine** (`dialogue.py`): every LLM round has two stall
timeouts, first event 4s (`LLM_FIRST_EVENT_TIMEOUT_SECONDS`) and later events 8s
(`LLM_EVENT_TIMEOUT_SECONDS`); a failed / stalled-before-first-token / empty round is retried
ONCE on the identical messages (nothing duplicated), but not after the caller heard part of the
answer and not after a stall in an already-flowing stream, so the worst case before the
fallback is ~8s (2 x 4s), not 16s; still failing → `LLMError` (an empty reply is an error now,
first-round failures roll the user message back); `add_assistant_message()` puts the greeting
in the history. **`CallSession`** (`session.py`, reusable by step 3): speaks the greeting
(business.yaml, no LLM), and on a failed turn says «Простите, я не расслышал. Повторите,
пожалуйста.» (turn rolled back, caller repeats); the 2nd failed turn in a row → apology + hang-up
+ a `CallbackMessage` with the caller's last 3 lines and caller ID saved through the sink (so a
technical failure never silently loses a caller); a success resets the counter; emits
`TurnFailed(reason)` (not spoken). **`app.py`**: `open_runtime(settings, notify=, db_path=,
clock=, llm=, notifier=)` builds store → (notifier → `NotifyingSink`) → LLM client and closes
what it created; `Runtime.new_call(caller_phone)` = fresh prompt + tools + engine + session;
`offset_clock()` for a fake "now" that keeps ticking. `NotifyingSink.start()` returns how many
old records it queued. **CLI** (`cli.py`): `.venv/bin/python -m agent.cli [--caller PHONE]
[--notify] [--db PATH] [--now 'YYYY-MM-DD HH:MM'] [--script FILE] [--show-tools]`; notifications
are OFF unless `--notify`, database `DB_PATH` (data/agent.db) unless `--db`; `--script` reads the
caller's lines from a file (`#` comments, blank lines skipped; example in
`scripts/scenarios/`), `/quit` / Ctrl-D / Ctrl-C hang up, `/db` shows all saved records; at the
end it prints what was saved during the call exactly as the owner would see it (+ Telegram
state), and with `--notify` waits up to 30s for delivery. `--notify` also resends every older
unnotified record in that database (it says so). Bad arguments → exit 2, bad config / unusable
database → exit 1 without printing secret values.

## What was next, and the remaining roadmap (as of the end of step 1)

**Next (NOT started):** step 2 (STT/TTS, local mic): pick the voice (gender!), wire `SpeechGuard` before TTS,
decide the production LLM/provider order (see the latency issues), and re-run the latency study
from the production VPS.

**Remaining roadmap:** step 2 (STT/TTS, local mic) and step 3 (Asterisk + AudioSocket on the
real VPS).

## Step 1.11 (2026-09-26): grounding of caller data, before step 2

Problem: the model can write the caller's line into its reply («userМеня зовут Дмитрий») and call a
tool with a made-up name and phone in the same reply; the speech guard dropped the sentence but the
call still ran (once a fabricated name+phone was saved through `take_message`). Decisions:
(1) `role_leakage` is a tripwire for the whole round: stop reading the stream at the first such
sentence, discard every tool call, one corrective round with a hidden note («вызовы отменены, не
выдумывай данные клиента»), sentences the caller already heard stay in the history, a second leak →
neutral fallback and no tools (`DialogueEngine.respond`, event `ToolCallsDropped`, shown by the CLI
and the eval harness as an item of kind `dropped`). Other guard rules keep their tool calls.
(2) The real guarantee does not depend on leaks: `ToolExecutor.begin_turn(user_text)` now gets the
caller's words and `ToolRegistry` checks them (`src/agent/grounding.py`, `ru_words.spoken_digits`):
phone must occur in the caller's digits (or equal the caller ID), otherwise «ОШИБКА» that tells the
model to ask the caller; the name is refused once. `take_message` never loses a message over a name.
(3) Not done, on purpose: the dictated-number-vs-caller-ID conflict (own step and sweep), checks of
`car`/`service`/dates. Tests: `tests/test_grounding.py`, new cases in test_tools / test_dialogue /
test_ru_words / test_text_guard; existing tests now feed the caller's words to the registry.

## Step 1.12 (2026-09-26): the caller ID vs a number the caller dictated

Decisions (plan approved as written): (1) `ru_words.spoken_digit_runs()` returns the runs of digits
(any other word ends a run; separators and number words do not; each run knows whether it touches the
start / end of the utterance) and `spoken_digits()` is now its join. (2) `CallerSpeech` keeps every run
and, when a run ends one utterance and another starts the next, also the joined run (the STT cuts
dictation at pauses); a time at the end of one line must not swallow a number at the start of the
next, so the pieces are kept too. (3) `ToolRegistry`: if the model passes the caller ID and the LAST
full number the caller dictated is another one, «ОШИБКА: клиент называл другой номер», once per
dictated number (same rule as the name: a repeat after the caller spoke again is accepted, a repeat in
the same turn is refused); no digits in the error text. Applies to `prepare_booking` and
`take_message`; not to a hidden caller ID. (4) New eval scenario `dictates_other_number`
(evals/scenarios.py, cost table, ideal-run test). Result: docs/evals.md (Step 1.12 sweep): the scenario
does not trigger the old failure, the rule fired twice and was correct, the earlier «failures» were an
eval artifact.

