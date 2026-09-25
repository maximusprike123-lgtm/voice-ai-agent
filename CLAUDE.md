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
    `LLM_MODEL`). Currently pointed at **local Ollama, model `qwen3.5:4b`** (8GB Mac dev
    machine). `LLM_REASONING_EFFORT=none` disables the model's thinking mode — verified
    empirically (see `scripts/check_llm.py`): without it, a trivial question burned 705
    reasoning tokens; with it, 4. This is an optional passthrough field in `LLMClient`,
    omitted entirely when unset, so a non-reasoning backend is never sent a field it might
    reject. `LLMClient` also unconditionally drops any `reasoning`/`reasoning_content`
    delta fields a backend might stream, so thinking text can never leak into speech.
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
date), 1.4 (`LLMClient` + `OpenAICompatibleLLMClient`, offline tests, live check script).

**Next: 1.5 — DialogueEngine.** Holds message history, drives the LLM-then-tools-then-LLM
loop, splits streamed text into sentences (the units TTS will speak later), emits
`Say`/`ToolResult`/`EndCall` events. Must be cancellable (for barge-in later, step 3).

**Remaining roadmap (from the original plan):** 1.6 tools (`submit_booking`,
`take_message`, `end_call`) · 1.7 SQLite storage · 1.8 Telegram notifier · 1.9 CLI ·
1.10 tests incl. scripted scenario dialogues against the real LLM. Then step 2 (STT/TTS,
local mic) and step 3 (Asterisk + AudioSocket on the real VPS).

## Known open issues

- **Ollama cold start:** ~4.4s time-to-first-token when the model isn't already loaded in
  memory (Ollama unloads idle models after a few minutes); ~1.0s once warm. Needs a
  decision later (`OLLAMA_KEEP_ALIVE`, a keep-warm ping, or accept it for a low-volume
  pilot) — not fixed yet.
- **TTFT hasn't been measured with the real system prompt yet.** `check_llm.py` uses a
  short ad-hoc prompt; `build_system_prompt()` (step 1.3) is much longer (business data +
  14-day calendar) and will add prompt-processing time on this CPU-only 8GB Mac. Re-measure
  once DialogueEngine (1.5) wires the real prompt in.

## Commands

```bash
.venv/bin/pytest                    # all tests (offline, no network)
.venv/bin/ruff check .               # lint
.venv/bin/ruff format .              # format
.venv/bin/python scripts/check_llm.py   # live check against the real LLM_BASE_URL in .env:
                                          # thinking disabled? tool calls parse? TTFT?
```
