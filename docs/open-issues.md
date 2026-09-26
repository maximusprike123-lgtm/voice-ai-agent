# Open issues (details)

Moved out of CLAUDE.md verbatim. CLAUDE.md lists the top ones in one line each. Latency, provider
and infrastructure issues (OpenRouter latency, Ollama backup, Telegram reachability, provider
routing, guardrails, infra noise) are in [latency.md](latency.md). Eval history is in
[evals.md](evals.md).

- **Foreign-script characters in spoken text** (Chinese seen once on qwen3.5:4b: «тридцати五千
  рублей»): `agent.text_guard.foreign_script_chars()` exists and the 1.10 invariant
  `no_foreign_script` found none in 46 DeepSeek runs, but nothing in the production path uses
  the guard yet. Decide in step 2 what to do on a hit (regenerate / drop the sentence) before
  text reaches TTS. Applies to any backend.
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
- **Fallback phrases use masculine forms («не расслышал»); they must match the gender of the
  TTS voice chosen in step 2** (`ASK_TO_REPEAT` in `session.py`; a test documents it). The
  model's own wording is gendered too («принял», «записал»): the prompt should state the
  agent's gender once the voice is chosen.
- **(Historical: before the guard and the prompt pass; the counts in «Final comparison» supersede
  it) 1.10 baseline 2: what the 14 failures of the second sweep were (triage from transcripts).** *Harness/check artifacts (5 runs, not agent faults):* the caller
  volunteers its phone number together with its name, so `asks_for_a_number` (hidden_caller_id)
  can never fire (0/5 by construction; 2 more runs fail only on it); a caller line that says
  goodbye in the SAME turn as the read-back «да» ends the run before the agent can answer, and
  the agent legitimately cannot hang up in the turn of an accepted save (hidden #1); a run ended
  on the caller's goodbye while the agent's question («Извините, я ещё не отправил заявку…
  Уточните…» / «как вас зовут?») was still pending (hidden #4, address_only #2); the caller
  improvised a metro-route question, so a `take_message` broke `no_messages` (address_only #0).
  *Proposed further harness fixes, NOT applied:* end the run on a farewell only if the agent's
  reply does not end with a question (let the caller answer it); `asks_for_a_number` only when
  the caller had not already given a number; `agent_ended_call` not required when the farewell
  came in a turn with an accepted save; drop `no_messages` from address_only (or accept a
  message for an unanswerable follow-up); add the invariant below.
  *Real agent findings (9 runs):* (1) **the agent said «Заявка принята, администратор
  перезвонит…» WITHOUT calling `confirm_booking` (service_not_listed #0): the caller believes the
  booking is taken, nothing was saved (1/50). No invariant catches it yet (`no_slot_confirmed_claim`
  only looks for «подтверждена»); add `no_acceptance_claim_without_save` (any «заявка принята /
  передана администратору / сообщение передано» spoken before a committed tool of that run).**
  (2) The agent **reads out the full phone number** («Номер — 8 916 123 45 67», «…восемь девятьсот
  шестнадцать…»), 5/50 runs (happy_path 2, approximate_time 2, hidden_caller_id 1), despite the
  prompt rule; the code-built read-back never does. (3) «Записал / Записала (номер)» before the
  booking is confirmed, 3/50. (4) With a HIDDEN caller ID it said «Ваш номер телефона
  определился, записать вас на номер, с которого вы звоните?» (hidden #2). (5) service_not_listed
  #3: the caller asked to book twice, the agent insisted on `take_message` (an `other` booking
  was never offered). (6) Own recap before `prepare_booking` in 19/50 runs (38%; «Всё верно?»,
  «Уточню: …»), which also makes callers answer «да» early (and even say goodbye) before the
  real read-back. The invariants that are code-enforced held everywhere: no foreign script, no
  invented price, no confirm before a read-back, no premature confirm attempt, no hang-up in a
  save turn. These findings suggest moving more into code (e.g. a code-side check of spoken
  text for phone digits / acceptance claims, or dropping the model's own recap by prompt).
- **1.10 status after the guard + prompt pass (what is still open, from the 99 gradable runs):**
  (1) the model **still writes «записал/записала»** at the same rate (6 sentences / 99 runs; the
  guard blocks them all, so callers no longer hear them, but the prompt rule does not work by
  itself). (2) **The model still recaps before `prepare_booking`** in 13% of runs (was 39%).
  (3) **Hidden caller ID:** 1/10 runs still offered «на номер, с которого вы звоните, или на
  номер, который вы продиктовали?» (`never_offers_this_number`). (4) **Dictated number:** 1/10
  sunday_closed run still saved the caller ID instead of the number the caller dictated
  (`saved_phone_is_the_dictated_number`); it is only checked in evals, nothing enforces it in
  code (idea: when the caller dictated digits, validate `prepare_booking`'s phone against the
  digits heard, which needs the raw caller text in the tool layer). (5) **Two eval checks were too strict (fixed 2026-09-26, see the follow-ups).**
  (6) **Role leakage and made-up caller data: DONE in step 1.11 (2026-09-26).** On `role_leakage` the
  engine stops reading the reply, drops ALL tool calls of that round (`ToolCallsDropped`, only when
  a call had already been read; otherwise the closed stream never shows it) and runs a corrective
  round; a second leak speaks «Давайте продолжим.» with no tools. Independently of leaks the tools
  refuse data the caller never said (`agent.grounding.CallerSpeech`, fed by `begin_turn(user_text)`):
  a phone (digits found in the caller's words, in digits or number words, also dictated in pieces;
  the caller ID counts as said) in `prepare_booking` and `take_message` → «ОШИБКА»; a name in
  `prepare_booking` is refused ONCE (accepted on the same name after the caller spoke again, so
  «Дима» → «Дмитрий» costs one turn), in `take_message` it is dropped (`name=None`, warning). Limits:
  colloquial numerals («двойка», «две девятки») are not understood; the name match has no
  diminutive dictionary; `car`, `service_id`, dates are still not checked (the read-back is the
  net). Not measured by the scenarios (they never fired, see evals.md).
  **Step (б) DONE (step 1.12, 2026-09-26): the caller ID is refused once when the caller dictated
  a different full number.** `CallerSpeech.dictated_numbers()` (runs of digits said in a row; the
  pieces of a number dictated in several utterances are also kept joined; exact 10-11 digit runs
  only, so «14:00, 8 916…» with no word between them is missed) and `ToolRegistry._phone_conflict`
  in `prepare_booking` / `take_message`: error «клиент называл другой номер» once per dictated
  number, a repeat after the caller spoke again goes through. Sweep results and the caveat that
  the failures it was meant to remove were mostly eval artifacts: docs/evals.md (Step 1.12 sweep).
  (7) `acceptance_claim` never fired in the 200 runs of the final comparison
  (it happened once in ~250 earlier): rare but the guard covers it.
- **Background-process gotcha for long sweeps in this environment:** a sweep started with a
  plain `&` can be lost between tool calls; use `subprocess.Popen(..., start_new_session=True)`,
  track it by pid (`kill -0`), and don't `pgrep -f "python …"` (the interpreter shows up as
  `Python`, case matters).
- **The model adds redundant checks** («Уточню: … Верно?» about the date before
  `prepare_booking`, or re-asking the time): measured in 1.10 as the warning
  `own_recap_before_prepare` (19/50 runs); not fixed.
- **Empty model reply:** now handled (1.9): the engine retries once, then raises `LLMError`,
  and `CallSession` says the fallback phrase. (Seen once on qwen3.5:4b.)
- **Together stalls on the first token, and the retry goes to the same provider (found 2026-09-26,
  not fixed).** In the grounding sweep (`data/evals/grounding10`) 33 of 468 agent requests (7%, max
  27s) got no first event within the 4s limit; 13 of 100 runs ended as infra errors, most of them
  in the first runs of a scenario. `LLM_EXTRA_BODY` routes `order: [Together, Fireworks]`, so the
  engine's single retry (`DialogueEngine`, same `LLMClient.stream()` call) starts at Together again
  and can stall again; Fireworks is only reached on a provider error, not on a slow first token.
  Idea for step 2 (NOT done now): send the retry to another provider (Fireworks first), e.g. by
  giving `LLMClient.stream()` a per-attempt routing override, and re-measure the stall rate and
  the p90 from the production VPS (see also latency.md). Until then evals and calls see about one
  stalled request in fourteen.

