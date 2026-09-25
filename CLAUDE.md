# Voice AI agent — car detailing center (Moscow)

Demo/pilot inbound phone agent, Russian only. Answers FAQ questions and collects booking
requests (name, phone, car make/model, service, preferred date/time), which go to the
owner via Telegram. No real calendar integration — the owner confirms manually.

## Working rules

- **Analysis → plan → my approval → implementation → tests.** Don't start writing code for
  a step before the plan for that step has been approved.
- **No secrets in code or chat.** Real values live only in `.env` (gitignored). When
  inspecting `.env`, don't print secret values (API keys, tokens) into the conversation —
  non-secret settings (URLs, model names, paths) are fine to show.
- **Small steps.** Build and verify one numbered step at a time (see Roadmap). Don't jump
  ahead to a later step without asking.
- **Don't touch unrelated files.** A step's edits should stay scoped to what that step needs.
- Every step ends with passing tests (`pytest`) and a clean lint (`ruff check .`).

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
  - **STT/TTS:** vendor not chosen yet (candidates: ElevenLabs, Deepgram). Foreign services
    are acceptable for the MVP despite the RU-only telephony/LLM hosting constraint — this
    may need a network path check once a vendor is picked.
- **Principle: code enforces guarantees, the LLM handles conversation.** Anything the business
  relies on must not depend on the model obeying the prompt. Examples so far: booking data is
  validated and normalized in code; a booking is saved only by `confirm_booking` after code
  has read the draft back; the read-back text is built by code and spoken verbatim by the
  engine (`ToolOutcome.say`), so its phone digits are always exactly the last 4 and its
  numbers/dates/times are always in words; the model is never asked to say digits; an
  approximate time cannot become an invented HH:MM because it goes into `preferred_period`.
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
- **Principle: code enforces guarantees, the LLM handles conversation.** Anything the business
  relies on must not depend on the model obeying the prompt. Examples so far: booking data is
  validated and normalized in code; a booking is saved only by `confirm_booking` after code
  has read the draft back; the read-back text is built by code and spoken verbatim by the
  engine (`ToolOutcome.say`), so its phone digits are always exactly the last 4 and its
  numbers/dates/times are always in words; the model is never asked to say digits; an
  approximate time cannot become an invented HH:MM because it goes into `preferred_period`.
  When a live run shows the model slipping on something that matters, move it into code
  rather than into more prompt text. The LLM keeps what is conversational: understanding the
  caller, phrasing questions, answering FAQ.
- **Knowledge:** `config/business.yaml` (hours, services, prices, FAQ) loaded straight into
  the system prompt. No RAG — the FAQ is small and fixed.
- **Bookings:** validated, then **written to SQLite before** the Telegram notification is
  sent, so nothing is lost if Telegram fails. The agent never confirms a time slot itself.
  SQLite is also the outbox: every row has `notified_at` (NULL until the owner was told), so
  after a crash between save and Telegram the notifier re-sends whatever is still NULL.
- **Scope:** inbound calls only, Russian only, no call transfer, no LangVerse code reused.

## Status

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

Backend switch (after 1.6): main LLM is now OpenRouter DeepSeek (see Architecture);
`check_llm.py` gained the token-based "no hidden reasoning" check (it fails if
`reasoning_tokens > 0` or a one-word answer costs > 20 completion tokens; verified to FAIL with
`LLM_REASONING_EFFORT=high`); `measure_ttft.py` skips its Ollama-only cold-start run on
remote backends.

**Next: 1.9 — CLI** (text dialogue in the terminal: builds `SqliteSink` → `TelegramNotifier`
→ `NotifyingSink`, `DialogueEngine`, `ToolRegistry`; handles empty LLM replies and short LLM
timeouts).

**Remaining roadmap (from the original plan):** 1.10 tests incl. scripted scenario dialogues against the real LLM. Then step 2 (STT/TTS,
local mic) and step 3 (Asterisk + AudioSocket on the real VPS).

## Known open issues

- **OpenRouter latency (main backend), `scripts/measure_ttft.py`, real prompt ≈ 2200 tokens
  incl. tool schemas, thinking off.** Time to first *sentence* (what TTS waits for): median
  ≈ 1.3–1.6s, range 1.3–2.0s over 9 turns; LLM first token ≈ 0.95–1.3s. Budget is 1.5s, so
  it is marginal. **Tail risk:** one turn in that run hung for 27s (single sample; likely a
  provider stall), and a separate probe saw one 6.6s first token. The 60s client timeout is
  far too long for a phone call: step 1.9/3 needs a shorter timeout plus a fallback phrase /
  retry. Untried levers: OpenRouter provider routing (sort by latency / pin a provider).
  Prompt-prefix caching on OpenRouter has not been verified either way.
- **Ollama qwen3.5:4b backup is not usable for real calls** (8GB CPU-only Mac; measured with
  the real prompt ≈ 2100 tokens): first sentence cold ≈ 27s, warm with a changed prompt
  ≈ 13–15s (prefill ≈ 165 tok/s), later turns of one conversation 1.9–2.7s; generation ≈ 10
  tok/s. Prompt-prefix reuse does not work on it, so the static-first prompt layout (all
  static content first, per-call time/caller/calendar after `VOLATILE_MARKER`) gave no gain
  there: identical repeat 0.12s · same system prompt + new user text 3.3s · system prompt
  differing only in its last section 12.8s. Hypothesis (unverified): qwen3.5 is a
  hybrid/recurrent-state model that can't rewind to a mid-prompt point. The layout stays: it
  is free and helps backends with real prefix caching. Keep Ollama for offline logic
  development only.
- **Small Qwen models sometimes insert Chinese characters** (seen once on qwen3.5:4b:
  "тридцати五千 рублей"). Step 1.10 scenario tests must check replies for non-Cyrillic/Latin
  scripts, and a guard is needed before TTS. Applies to any backend, not only Qwen.
- **Conversation-level rule compliance is still model-dependent** (`live_booking_dialogue.py`,
  caller: "в субботу после обеда"). qwen3.5:4b (3 runs, old one-step flow): invented
  placeholders, skipped confirmation, invented `preferred_time`, needless `take_message`,
  promised free slots, garbled Russian. DeepSeek v4.1 flash, one-step flow: fine but read
  five digits of the number instead of four and re-asked the time. **Two-step flow on
  DeepSeek, 1 run:** correct end to end (read-back by code with exactly four digits,
  `preferred_period: "день"`, no `preferred_time`, phone normalized, confirm only after
  «да»). Still seen in that run: it asked «днём или ближе к вечеру?» after «после обеда»
  despite the prompt, and hung up (`end_call`) in the same turn as confirming, without
  waiting for the caller's goodbye. The early hang-up is now blocked in code (see the
  `end_call` guard); the time re-ask is left to the 1.10 scenario tests. One run is not reliability: 1.10 scenario tests
  must run several times and assert on tool arguments and on the spoken read-back.
- **Telegram permanent failure = silent pile-up.** If Telegram fails permanently (bad token,
  bot blocked, chat not found), bookings pile up unnotified and the owner doesn't know
  (the only trace is an ERROR log line). Before real customers: add a second alert channel
  (e.g. alert me if unnotified records are older than 30 minutes).
- **Telegram is reachable from Russia only unreliably (measured 2026-09-25, dev Mac in
  Moscow).** The live test message *was* delivered (after `/start` in `@voise_demo_agent_bot`;
  a bot can't message a user who hasn't started it, which first showed up as
  `400 chat not found`), but only on the 4th attempt (17s): TCP connect to `api.telegram.org`
  is instant while the TLS handshake stalls ~10s in about 2 of 8 tries (0.4-0.7s otherwise) —
  the usual signature of DPI throttling. The retry/backoff logic absorbed it, and it is
  invisible to callers because notifications are background work. Still: **re-measure from the
  production VPS before real customers**, expect slow/late owner notifications, and be ready
  with a workaround (an HTTPS proxy for httpx, or a relay; `TelegramNotifier` already takes
  `api_url`). This is another reason for the second alert channel above. The long-lived
  client reuses connections, so steady-state cost is lower than in the one-shot test script.
- **Empty model reply:** seen once (qwen3.5:4b, turn 2 of a conversation): the LLM returned
  no text and no tool call, so `DialogueEngine.respond()` yields no events and the caller
  would hear silence. Decide handling (retry / fallback phrase) in 1.9 or 1.10.
- **Foreign LLM host:** OpenRouter is outside Russia; the RU-only constraint applies to
  telephony. Check the network path and data-handling terms before a real pilot (call audio
  is not sent to the LLM, but transcripts with names and phone numbers are).

## Commands

```bash
.venv/bin/pytest                    # all tests (offline, no network)
.venv/bin/ruff check .               # lint
.venv/bin/ruff format .              # format
.venv/bin/python scripts/check_llm.py   # live check against the real LLM_BASE_URL in .env:
                                          # thinking off (text AND token usage)? tool calls
                                          # parse? TTFT?
.venv/bin/python scripts/measure_ttft.py  # real prompt through DialogueEngine: cold/warm TTFT
.venv/bin/python scripts/live_booking_dialogue.py  # scripted caller books via the real LLM + tools
.venv/bin/python scripts/send_test_notification.py  # ONE real test message to the owner's Telegram
```
