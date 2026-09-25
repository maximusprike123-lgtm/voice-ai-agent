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
- **Knowledge:** `config/business.yaml` (hours, services, prices, FAQ) loaded straight into
  the system prompt. No RAG — the FAQ is small and fixed.
- **Bookings:** validated, then **written to SQLite before** the Telegram notification is
  sent, so nothing is lost if Telegram fails. The agent never confirms a time slot itself.
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

1.6 (tools, `src/agent/tools.py` + `records.py`: `ToolRegistry` implements `ToolExecutor`;
`submit_booking` validates and normalizes — phone → `+7XXXXXXXXXX`, date required, not in the
past, ≤ 90 days ahead, on a working day; `preferred_time` optional but if given must be within
that day's hours (and not already past today); `service_id` from business.yaml or the reserved
`other` (then `notes` required, no price ever returned); duplicate guard per call. Validation
errors go back to the model as `ОШИБКА: …` results. `take_message`, `end_call`. Schemas are
static (built from business.yaml only). Records go to a `RecordSink` — `InMemorySink` for now.
`business.yaml` may not use the id `other` (fails at load). The prompt's static part got the
`other` rule, the "approximate period → no `preferred_time`, words into `notes`" rule and an
`ОШИБКА` rule. `scripts/live_booking_dialogue.py` runs a scripted caller against the real LLM.)

Backend switch (after 1.6): main LLM is now OpenRouter DeepSeek (see Architecture);
`check_llm.py` gained the token-based "no hidden reasoning" check (it fails if
`reasoning_tokens > 0` or a one-word answer costs > 20 completion tokens; verified to FAIL with
`LLM_REASONING_EFFORT=high`); `measure_ttft.py` skips its Ollama-only cold-start run on
remote backends.

**Next: 1.7 — SQLite storage** (a `RecordSink` that writes before anything is notified).

**Remaining roadmap (from the original plan):** 1.8 Telegram notifier · 1.9 CLI ·
1.10 tests incl. scripted scenario dialogues against the real LLM. Then step 2 (STT/TTS,
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
- **Booking-rule compliance is model-dependent** (`scripts/live_booking_dialogue.py`, caller:
  "в субботу после обеда"). qwen3.5:4b, 3 runs: invented `name`/`car` placeholders, submitted
  without the confirmation step, invented `preferred_time` "15:30", a needless `take_message`
  after a validation error, promised "есть свободные места", garbled Russian.
  DeepSeek v4.1 flash, 1 run: followed the whole flow (asked in order, read the request back,
  submitted only after "всё верно", left `preferred_time` empty and put the caller's words
  in `notes`, normalized phone, never confirmed the slot). Only one run, so not proof:
  minor slip seen — it read out five digits of the caller's number instead of four. The tool
  layer cannot detect invented-but-valid values, so 1.10 scenario tests must assert on tool
  arguments (e.g. no invented `preferred_time`), across several runs.
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
```
