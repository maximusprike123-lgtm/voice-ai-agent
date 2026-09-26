# Latency, providers and infrastructure findings

Moved out of CLAUDE.md verbatim (these were entries of «Known open issues»). Measurements were taken
from the dev Mac in Moscow, NOT the production VPS: re-measure there before deciding anything.
Related: [open-issues.md](open-issues.md), [evals.md](evals.md), [decisions.md](decisions.md).

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
- **OpenRouter first-token latency, measured per routing configuration**
  (`scripts/latency_study.py`: 2 runs x 40 rounds x 5 configurations, real prompt + real tools,
  thinking off, 2026-09-25 ~19:00 MSK, from the dev Mac, so NOT the production VPS; the requests
  of a round are sent concurrently). Two runs, p50 / p90 / max / share over 4s:
  default routing 1.53/2.06/5.3/2% and 1.69/2.49/5.2/5% (served by DeepInfra ~55%, Fireworks
  ~30%, Together ~12%) · `sort=latency` 1.46/2.51/5.3/5% and 1.63/2.54/5.4/5% (same provider mix
  as default: it did not concentrate on the fastest) · pinned **Fireworks** 1.59/2.15/2.7/0% and
  1.63/2.09/3.0/0% · pinned **DeepInfra** 1.58/2.86/11.5/5% and 1.64/4.19/**36.6**/12% · pinned
  **Together** 1.17/1.87/6.1/2% and 1.17/1.92/3.6/0%. So p50 is ~1.2-1.7s everywhere; the
  difference is the TAIL, and **DeepInfra is the tail** (worst in both runs, and default routing
  sends it the most traffic); Together has the best p50/p90, Fireworks the tightest max. With 40
  samples p99 is the maximum, and Together-vs-Fireworks differences are within noise; DeepInfra's
  tail is the one repeated finding. Time until the engine can see a tool-call reply (arguments
  complete) was only ~0.6s (max 1.1s) after the first token in the 6 tool replies observed, so
  tool calls do not explain the 4s timeouts seen in the live CLI run (3 in ~10 rounds, default
  routing, multi-turn history: worse than the study suggests; the study does not cover long
  histories). **Not applied (decision pending; timeouts unchanged):** a preferred order with
  fallbacks (`LLM_EXTRA_BODY={"provider": {"order": ["Together", "Fireworks"],
  "allow_fallbacks": true}}`, verified live: 8/8 requests served by Together) or
  `{"provider": {"ignore": ["DeepInfra"]}}`; re-run the study from the production VPS before
  deciding.
- **OpenRouter account guardrails limit which providers can serve us:** the account's privacy
  settings (zero data retention, no training on prompts) remove `deepseek` (first party),
  `alibaba`, `streamlake`, `gmicloud` and `atlas-cloud` from routing; pinning `DeepSeek` fails
  with HTTP 404 "No endpoints found". Good for customer data (names, phone numbers), but the
  eligible set is US/EU infra providers, and changing those settings changes the latency picture.
- **Infra noise under load:** the final comparison ran both sweeps at once (6 concurrent agent
  streams): 19 of 488 Together requests (3.9%) and 17 of 509 (3.3%) had a first token later than
  4s (p90 2.6-2.9s, max 7-11s), versus 1-2 per ~250 in isolated sweeps; the retry rescued all but
  1 (new) and 5 (ref) turns. A few requests fell back to Fireworks/OpenInference (the latter
  slow: 6.8-11.9s). The tail depends on load and time; re-measure from the production VPS.
- **Infra noise:** sweep 1: 10 first-token timeouts in ~190 requests (6 rescued by the retry, 4
  failed turns = 8% of runs); sweep 2: 1 timeout in 252 requests, 0 failed turns. Both sweeps
  had the Together/Fireworks pin, and in sweep 2 every request was served by Together, so the
  tail varies over time (evening MSK) rather than with the pin; the providers of sweep 1's
  timeouts were not recorded (unrecoverable). Keep the request telemetry on.
