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
date), 1.4 (`LLMClient` + `OpenAICompatibleLLMClient`, offline tests, live check script),
1.5 (`DialogueEngine` in `src/agent/dialogue.py`: history, LLM→tools→LLM loop capped at 5
rounds, `SentenceSplitter`, `Say`/`ToolResult`/`EndCall` events, cancellable; tools come in
through a `ToolExecutor` protocol that 1.6 implements; `scripts/measure_ttft.py` times the
real prompt through the engine). `mark_spoken` (trim history to what the caller heard) is
deferred to step 3. Follow-up: `build_system_prompt()` now puts all static content first and
the per-call content (time, caller number, 14-day calendar) last, after `VOLATILE_MARKER`;
a test pins the shared prefix. Tool schemas (1.6) must stay static too: no dates, caller data
or other per-call content in tool names/descriptions/parameters.

**Next: 1.6 — tools** (`submit_booking`, `take_message`, `end_call`) implementing
`ToolExecutor`.

**Remaining roadmap (from the original plan):** 1.7 SQLite storage · 1.8 Telegram notifier · 1.9 CLI ·
1.10 tests incl. scripted scenario dialogues against the real LLM. Then step 2 (STT/TTS,
local mic) and step 3 (Asterisk + AudioSocket on the real VPS).

## Known open issues

- **Latency on qwen3.5:4b / 8GB CPU-only Mac is far over budget with the real prompt**
  (measured with `scripts/measure_ttft.py`; prompt ≈ 2100 tokens incl. tool schemas;
  single run, numbers are noisy). Time to first *sentence* (what TTS waits for):
  cold ≈ 27s · warm with a changed prompt ≈ 13s (prefill ≈ 165 tok/s, no KV-cache reuse) ·
  warm with an identical prompt 1.6–5.5s (inconsistent, unexplained) · later turns of the
  same conversation 1.9–2.7s. Generation is only ≈ 10 tok/s, so the first sentence lands
  ~1–1.3s after the first token. Budget is 1.5s.
- **Prompt-prefix reuse does not work on Ollama + qwen3.5:4b, so the prompt reorder did not
  help here.** Before/after `measure_ttft.py` (new call = changed time + caller): first
  sentence median 15.5s → 14.6s, i.e. no real change. Probe (Ollama `/api/chat`, ~2100-token
  prompt): identical repeat 0.12s · same system prompt, different user text 3.3s · system prompt
  differing only in its last section 12.8s · fully cold 12s. So the runner only resumes from
  (near) the end of the previous request, never from an arbitrary shared prefix. Hypothesis
  (unverified): qwen3.5 is a hybrid/recurrent-state model and can't rewind its state to a
  mid-prompt point. The static-first layout stays: it is free and helps backends with real
  prefix caching (vLLM, hosted APIs, attention-only models on llama.cpp). Re-measure on the
  production backend; that choice (and the cold-start / `OLLAMA_KEEP_ALIVE` question) is
  deliberately deferred until before step 2, since latency on the 8 GB dev Mac isn't
  representative.
- **Small Qwen models sometimes insert Chinese characters** (seen once: "тридцати五千
  рублей"). Step 1.10 scenario tests must check replies for non-Cyrillic/Latin scripts, and a
  guard is needed before TTS.
- **Empty model reply:** seen once in a real run (turn 2 of a conversation): the LLM returned
  no text and no tool call, so `DialogueEngine.respond()` yields no events and the caller
  would hear silence. Decide handling (retry / fallback phrase) in 1.9 or 1.10.

## Commands

```bash
.venv/bin/pytest                    # all tests (offline, no network)
.venv/bin/ruff check .               # lint
.venv/bin/ruff format .              # format
.venv/bin/python scripts/check_llm.py   # live check against the real LLM_BASE_URL in .env:
                                          # thinking disabled? tool calls parse? TTFT?
.venv/bin/python scripts/measure_ttft.py  # real prompt through DialogueEngine: cold/warm TTFT
```
