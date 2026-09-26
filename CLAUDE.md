# Voice AI agent — car detailing center (Moscow)

Demo/pilot inbound phone agent, Russian only. Answers FAQ questions and collects booking
requests (name, phone, car make/model, service, preferred date/time), which go to the
owner via Telegram. No real calendar integration — the owner confirms manually.

Details live in `docs/` (read the relevant file before working on that area):
[docs/decisions.md](docs/decisions.md) (full architecture decisions + step 1.1–1.9 log),
[docs/evals.md](docs/evals.md) (eval harness, sweeps, speech guard, final comparison),
[docs/latency.md](docs/latency.md) (latency studies, providers, Ollama, Telegram reachability),
[docs/open-issues.md](docs/open-issues.md) (open issues in full).

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

## Architecture decisions (one line each; full text and measurements in docs/decisions.md)

- **Telephony:** 1ATC SIP → Asterisk (Docker, Russian VPS) → **AudioSocket** (TCP, 8kHz/16-bit
  PCM) → our Python app. Asterisk owns all SIP/RTP; our code never implements SIP/RTP. (Russian
  number + foreign IPs blocked by the provider.)
- **No Pipecat, no LiveKit:** no AudioSocket transport and no Yandex SpeechKit in Pipecat; the
  protocol is simple, so we write a minimal asyncio pipeline (VAD, streaming STT/LLM/TTS, barge-in).
- **`DialogueEngine` is audio-agnostic** (text in, `Say`/`ToolResult`/`EndCall` events out), so
  the same engine runs in the CLI, on a local mic and on real calls.
- **`LLMClient`/`STTClient`/`TTSClient` are vendor-agnostic** (hard requirement: vendor may
  change). LLM = OpenAI-compatible API, now OpenRouter `deepseek/deepseek-v4.1-flash` (pinned,
  never `-latest`); Ollama `qwen3.5:4b` is the offline backup only. Switching = editing `.env`.
  **Thinking must be OFF** (`LLM_REASONING_EFFORT=none`, verified by token counts, not text);
  `LLM_EXTRA_BODY` passes provider routing. STT/TTS vendor not chosen (ElevenLabs/Deepgram).
- **Code enforces guarantees, the LLM handles conversation:** anything the business relies on
  must not depend on prompt obedience. Two-step booking (`prepare_booking` → code-built read-back
  → `confirm_booking` only after «да»), code-built read-back/acceptance sentences (`say`), the model
  never says phone digits, `end_call` / same-turn-confirm guards, and a **speech guard** on every
  model sentence. When a live run shows a slip that matters, move it into code, not more prompt.
- **Knowledge:** `config/business.yaml` loaded straight into the system prompt; no RAG (FAQ is
  small and fixed). Static prompt part first, per-call part (time, caller, calendar) last; tool
  schemas static too.
- **Bookings:** validated, then written to SQLite **before** Telegram; SQLite is also the outbox
  (`notified_at`). Notifications never block the call (background worker, 5-min sweep,
  resend at startup). **At-least-once** delivery: duplicates are accepted on purpose, a lost
  booking is not. Don't "fix" without a plan for the lost-message case.
- **Scope:** inbound calls only, Russian only, no call transfer, no LangVerse code reused.

## Status

**STEP 1 (text agent, roadmap 1.1–1.10) IS COMPLETE — tag `v0.1-text-agent`.** Russian,
audio-agnostic booking agent: DialogueEngine + CallSession + tools + SQLite + Telegram notifier +
CLI, code-enforced guarantees (two-step booking, code-built read-back/acceptance, `end_call` and
same-turn-confirm guards, speech guard with `role_leakage`), and an eval harness (`python -m evals`)
with a measured result: **95/99 = 96%** raw pass rate, 10 runs per scenario (old agent 76/95 = 80%;
see docs/evals.md). ~980 offline tests. Post-tag follow-ups (role-leakage rule, two relaxed eval
checks) are done.

Code map: `src/agent/{dialogue,session,app,cli,tools,records,storage,notifier,text_guard,
ru_words,prompt,llm,settings,business}.py`, `evals/`, `scripts/`, `config/business.yaml`.

**Next (NOT started, needs a plan and approval first):** step 2 (STT/TTS, local mic): pick the
voice (**gender**: fallback phrases are masculine), wire `SpeechGuard` before TTS, decide the
production LLM/provider order, re-run the latency study from the production VPS.

**Roadmap:** step 2 (STT/TTS, local mic), then step 3 (Asterisk + AudioSocket on the real VPS;
adds `call_id` as SQLite migration 2, and `mark_spoken`).

## Top open issues (details in docs/open-issues.md and docs/latency.md)

- **Latency:** first sentence p50 ≈ 1.2–1.7s vs 1.5s budget; the tail matters (DeepInfra worst,
  Together best p50/p90; ~3–4% of requests > 4s under load). Provider order (Together→Fireworks) is
  set via `LLM_EXTRA_BODY`; re-measure from the production VPS.
- **Ollama backup unusable for calls** (8GB CPU Mac: first sentence 13–27s); offline logic dev only.
- **Telegram:** permanent failure = silent pile-up (needs a second alert channel: unnotified
  records older than 30 min); reachable from Russia only unreliably (TLS stalls ~10s in 2/8 tries).
- **TTS voice gender** must match the masculine fallback phrases (`ASK_TO_REPEAT` etc.) and the
  model's gendered wording; state the agent's gender in the prompt once the voice is chosen.
- **Model still slips:** writes «записал» (blocked by the guard) and recaps before `prepare_booking`
  (13%); hidden-ID «этот номер» offer 1/10; dictated number vs caller ID not enforced in code.
- **Role leakage is blocked in speech but NOT in tool calls** (a fabricated name+phone was once
  saved via `take_message`); nothing verifies name/phone were actually said. Proposed, not approved.
- **Foreign-script chars** (Chinese on qwen3.5): `foreign_script_chars()` guard exists in the
  speech guard; step 2 must decide behavior before TTS.
- **Conversation-level rule compliance** is model-dependent; run several evals, not one dialogue.
- **Account guardrails** (ZDR/no training) exclude first-party `deepseek` and other providers.
- **Gotcha for long sweeps:** start with `subprocess.Popen(..., start_new_session=True)`, track by
  pid (`kill -0`); a plain `&` job can be lost; don't `pgrep -f "python …"` (shows as `Python`).

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
.venv/bin/python scripts/latency_study.py            # TTFT p50/p90/p99 per OpenRouter routing (~4 min, cents)
.venv/bin/python -m evals --dry-run                  # scenario evals: plan + cost estimate, no LLM calls
.venv/bin/python -m evals [--scenarios a,b] [--runs 5]  # REAL LLM calls (~$0.16 per 50 runs), on demand only
.venv/bin/python -m evals --calibrate-caller         # rank candidate caller models (cents)
.venv/bin/python -m evals.compare <ref_dir> <new_dir>  # compare two sweeps: pass rates, failed checks, guard blocks
.venv/bin/python -m agent.cli --show-tools           # talk to the agent in the terminal (see 1.9)
.venv/bin/python -m agent.cli --script scripts/scenarios/booking_saturday_afternoon.txt \
    --caller +79991234567 --show-tools --db /tmp/scratch.db   # scripted call; add --notify for real Telegram
```
