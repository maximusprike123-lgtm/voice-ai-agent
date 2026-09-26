# Evals, speech guard and prompt pass (step 1.10 and follow-ups)

Moved out of CLAUDE.md verbatim. Run with `python -m evals` (see the Commands in CLAUDE.md).
Related: [decisions.md](decisions.md), [latency.md](latency.md), [open-issues.md](open-issues.md).

## Harness design (1.10) and its fixes

1.10 (scenario evals, `evals/` package, run with `python -m evals`; on demand, NOT in pytest):
an **adaptive simulated caller** (an LLM with a persona, facts and behavior per scenario; it
hears only the agent's spoken sentences and ends with `[КОНЕЦ]`) talks to the real
`CallSession` + `ToolRegistry` + a temporary SQLite database, on a fixed clock (Friday
2026-09-25 15:00 Moscow). 10 scenarios (happy_path_booking, approximate_time, changes_mind,
hidden_caller_id, service_not_listed, question_outside_faq, price_only, rude_offtopic,
address_only, sunday_closed), each run N times (default 5, concurrency 4). Assertions are on
OUTCOMES: 8 invariants for every run (no foreign script; no «записал» before a successful
confirm; no price outside the price list; no full phone number spoken; no «запись
подтверждена» claim; confirm only after a read-back on an earlier turn; no premature confirm
attempt; no `end_call` in the turn of a save) plus per-scenario checks on saved records and
speech. A run passes only if ALL its checks pass; infra errors (LLM failures after the
engine's retry) and inconclusive runs (turn cap, broken caller) are reported separately and
excluded from the rates. Output: per-scenario pass rate, per-check breakdown, invariant
matrix, tokens and approximate cost; `data/evals/<time>/` gets report.txt, results.json,
transcripts.txt. Options: `--dry-run` (plan + cost estimate), `--scenarios`, `--runs`,
`--concurrency`, `--caller-model`, `--max-cost` (default $2, refuses/stops above it),
`--show-failures`, `--min-pass-rate`. The simulated caller defaults to the first working of
gemini-2.5-flash-lite / gpt-4.1-nano / mistral-small-3.2 / llama-3.3-70b (a different family
from the agent, each probed against this account's privacy filters; the agent's own model
only as a last resort). The harness has its own offline tests (checks, "ideal run" per
scenario, caller, harness, cost, report, CLI). Small production additions: `text_guard.
foreign_script_chars()` (the guard for TTS text; not yet wired into the audio path),
`ru_words.spoken_amounts()` / `longest_number_run()` (read prices and phone digits back out
of spoken text), `LLMClient.usage_hook` (opt-in token accounting), and a **same-turn confirm
guard**: `confirm_booking` returns `ОШИБКА` if the draft was prepared in the current turn, so
the model cannot skip the caller's «да» by calling prepare + confirm together. pytest now
puts the repo root on `pythonpath` so tests can import `evals`.

**1.10 harness fixes (after the first baseline; agent/prompt untouched):** the hang-up marker
`[КОНЕЦ]` counts only on a farewell line (otherwise it is stripped, counted as "ignored" and the
call goes on; the persona prompt now says so); `agent_ended_call` is required only if the
caller's last line was a farewell; `changes_mind` asserts outcomes (one booking, the final
slot, `abandoned_slot_not_saved`) instead of `prepared_at_least_2`; `service_not_listed`: the
persona clearly wants to BOOK, asserts an `other`/`ppf` booking and no message; the agent's own
recap before `prepare_booking` is a **warning metric** (reported per scenario, never pass/fail);
cost estimates are per scenario and match the measured cost within ~7%; the caller model is now
chosen by measurement: `python -m evals --calibrate-caller` (7 candidates, 66 probes each:
premature hang-ups on lines that must not end the call, ability to end the call, empties,
latency): gpt-4.1-nano 0/66 premature + 17/18 terminates, llama-3.3-70b 0/66 + 17/18,
gpt-4o-mini 0/66 + 15/18, gemma-3-27b 0/66 + 9/18, gemini-2.5-flash-lite 4/66 + 18/18,
qwen3-30b 6/66, mistral-small-3.2 0/65 (one request error). Default order is now gpt-4.1-nano,
llama-3.3-70b, mistral-small-3.2, gemini-2.5-flash-lite. New request telemetry
(`evals/telemetry.py`): every agent LLM request is logged with the serving provider and its
first-event latency; a request the engine gave up on is left running in the background
(30s cap) so its provider and true latency are still recorded. `LLMClient`'s `usage_hook` now
also receives the serving `provider` when the backend names one.

**Baseline sweep 1 (before the fixes, Gemini caller): 35% raw, dominated by caller artifacts.
Baseline sweep 2 (after the fixes, 2026-09-25, gpt-4.1-nano caller, 50 runs, $0.156, 0 infra
errors, 1 caller marker ignored): raw pass rate 36/50 = 72%** — happy_path_booking 3/5,
approximate_time 3/5, changes_mind 4/5, hidden_caller_id 0/5, service_not_listed 3/5,
question_outside_faq 5/5, price_only 5/5, rude_offtopic 5/5, address_only 3/5,
sunday_closed 5/5. Invariants: no_foreign_script 50/50, no_invented_prices 50/50,
no_slot_confirmed_claim 50/50, confirm_only_after_readback 50/50, no_premature_confirm_attempt
50/50, end_call_not_with_accepted_save 50/50, **no_записал_before_confirm 47/50,
no_full_phone_spoken 45/50**. Warning `own_recap_before_prepare`: 19/50 runs. All 252 agent
requests were served by Together (the `LLM_EXTRA_BODY` pin worked; Fireworks fallback never
needed): first event p50 1.27s, p90 2.22s, max 5.66s; exactly one round exceeded the 4s limit
and was given up on (service_not_listed, Together, 5.7s). Nothing was tuned after either sweep.

**Harness fixes, round 2 (all applied; agent untouched):** a caller farewell is deferred while
the agent's reply asks a question (the caller answers it first; counted as "farewells deferred");
`asks_for_a_number` is satisfied by asking OR by the caller volunteering a number;
`agent_ended_call` is not required when the farewell came in a turn with an accepted save;
`no_messages` dropped from address_only; new invariant `no_acceptance_claim_without_save`
(9 invariants now); the JSON export no longer drops `turn=0`.

## Baseline sweep 3 (fixed harness, agent unchanged)

**Baseline sweep 3 (fixed harness, agent/prompt unchanged, 2026-09-25, 50 runs, $0.151, 0 infra
errors): raw pass rate 42/50 = 84%** — happy_path_booking 4/5, approximate_time 3/5,
changes_mind 4/5, hidden_caller_id 5/5, service_not_listed 2/5, question_outside_faq 5/5,
price_only 5/5, rude_offtopic 5/5, address_only 5/5, sunday_closed 4/5. Invariants 50/50 except
«no_записал_before_confirm» 48/50 (`no_full_phone_spoken` 50/50 and
`no_acceptance_claim_without_save` 50/50 this time). Warning `own_recap_before_prepare` 13/50.
4 caller goodbyes deferred, 0 markers ignored. All 246 requests served by Together (p50 1.13s,
p90 1.71s, max 4.63s; 2 rounds over the 4s limit, both rescued by the retry). Failures: the
saved phone was the CALLER ID although the caller had dictated another number (happy_path, 1),
«записал» (approximate_time, sunday_closed), 3/5 service_not_listed runs ended with a
`take_message` and no booking, 1 changes_mind run and 1 approximate_time run saved nothing.
**Run-to-run noise is large:** the same agent leaked a full phone number in 5/50 runs in
sweep 2 and 0/50 in sweep 3, so compare sweeps on the counts of specific failure kinds and use
more runs per scenario (10 runs = 100 calls ≈ $0.30) before believing a 10-point difference.

## Speech guard, prompt pass, post-tag follow-ups

**Speech guard (1.10 follow-up, `agent/text_guard.py` + `DialogueEngine`):** deterministic, no
LLM call, applied to every sentence the MODEL writes before it becomes a `Say` (code-built
sentences: read-back, acceptance, greeting, session fallbacks are not checked). Rules, in order: `foreign_script`; `role_leakage` (a role label such as
`user:` / «Клиент:» or the caller's words written into the reply); `phone_digits` (a run of more than 4 digits or number words; in a
sentence about money only a digit group of 7+ counts); `acceptance_claim` («заявка
принята/передана/отправлена/оформлена», «сообщение передано», «ваша просьба принята», not in a
question) while nothing has been committed in this call; `written_down` (записал/записала/
записали/записано/записан) while nothing has been committed. «администратор перезвонит/свяжется»
is deliberately NOT a rule (2 of 3 such sentences in real calls were legitimate future
statements). `SpeechGuard` keeps per-call state (`committed`, flipped by any
`ToolOutcome.committed`) and `blocked`. **A blocked sentence is dropped** (a `SentenceBlocked(rule,
text)` event, not spoken; WARNING log) and never enters the history as spoken text, so history
= what the caller heard. Only if a whole reply would be silent (nothing left, no tool call) the
engine asks the model ONCE more with a hidden `system` note that quotes the blocked sentence and
says what to do («заявка НЕ сохранена: если клиент подтвердил, вызови confirm_booking…»; the
note is for that round only, not kept in the history); if that round is blocked or empty too, a
neutral fallback is spoken: `GUARD_FALLBACKS` = phone «Хорошо, номер есть.», acceptance
«Давайте ещё раз проверим данные заявки.», записал «Хорошо.», role leakage «Давайте
продолжим.», foreign script «Простите, уточните, пожалуйста, ваш вопрос.» (gender-neutral, no digits, never blaming the caller; tests
enforce it). A stalled first attempt that is followed by a fully blocked retry goes straight to
the fallback. Verified live on DeepSeek: the mid-conversation `system` note is accepted and the
model continues sensibly (next question, or `prepare_booking`). `SPEECH_GUARD=false` is a
debugging kill switch. The CLI prints `[guard] blocked (rule): …`; evals record blocked sentences
as items and report **"SPEECH GUARD BLOCKS" per scenario and rule** (the model's raw
violation attempts; with the guard on the spoken-text invariants can no longer show them).
`evals/compare.py` compares two sweeps (pass rates, counts of failed checks per 100 runs, guard
blocks; a sweep without the guard is replayed through it offline).

**Prompt pass:** no own recap before `prepare_booking` («СРАЗУ вызови prepare_booking, ничего
не говоря перед этим»; no «всё верно?», no «Уточню…»); with «Номер звонящего: не определён» the
number is UNKNOWN (never «определился», never «номер, с которого вы звоните», ask to dictate);
a number the caller dictates beats the caller ID; a caller who clearly wants to BOOK an unlisted
service gets an `other` booking, not «просто передать вопрос» (`take_message` only for
questions without an answer); and a contradictory old rule was removed («Пиши … номера
телефонов словами» told the model to spell phone numbers out). New invariant
`saved_phone_is_the_dictated_number` (10 invariants now).

**Post-tag follow-ups (2026-09-26):** (1) **`role_leakage` guard rule** (checked right after
`foreign_script`, before `phone_digits`): blocks a sentence that starts with or contains a role
label: Latin `user` / `assistant` / `system` when they open the sentence, carry a colon, or are glued
to Cyrillic («userЗаписывай»), and «Клиент:», «Агент:», «Администратор:», «Ассистент:»,
«Пользователь:», «Система:» with a colon. Ordinary words and Latin brand names («Ecosystem»,
«Системы безопасности», «Администратор перезвонит») are not touched. Fallback «Давайте продолжим.»
(neutral, no digits, no blame; covered by the wording tests) and a correction note for the model.
Tested with the real leaked sentence from the live probe («userЗаписывай на тот, что я продиктовал —
восемь девять один шесть…»), which is reported as `role_leakage` (not as a phone number). New eval
invariant `no_role_leakage` (11 invariants); `SPEECH GUARD BLOCKS` has a `role_leakage` column.
**Replayed over all ~450 recorded runs it found ONE real leak nobody had seen**
(old agent, service_not_listed #0): «userМеня зовут Дмитрий.», the model invented the caller's answer,
and a name «Дмитрий» plus a phone number the caller never gave were then SAVED by `take_message`.
(2) **Two eval checks relaxed:** `other_service_no_price` now allows quoting the price of a LISTED
alternative («у нас есть оклейка защитной плёнкой, от двадцати тысяч рублей»): every amount must be
in the price list and no CLAUSE (split on commas/semicolons, not dashes) about the requested item
(«фар») may carry an amount; `says_master_decides` accepts any wording that the price is not final
(`FINAL_PRICE_HEDGE`: мастер/осмотр/зависит от/уточн…/определ…/ориентировочн…). Both real runs that
failed the strict versions in the final sweep now pass.

## Final comparison (old agent vs guard + prompt pass)

**Final comparison (2026-09-25, 10 runs per scenario = 100 runs each, run simultaneously so the
network conditions match; `python -m evals.compare data/evals/final_ref10 data/evals/final_new10`).
Reference = the old agent (previous `prompt.py`, guard off), graded by the same checks; new = guard +
prompt pass.** Raw pass rate **76/95 = 80% → 95/99 = 96%** (infra errors 5 → 1). Per scenario
(ref → new): happy_path 5/9 → 10/10, approximate_time 7/10 → 10/10, changes_mind 9/10 → 10/10,
hidden_caller_id 6/9 → 9/10, service_not_listed 4/9 → 9/10, question_outside_faq 9/10 → 10/10,
sunday_closed 7/9 → 8/9, address_only 10/10 → 10/10, rude_offtopic 9/9 → 10/10, price_only 10/10 →
9/10. **Counts of specific failure kinds (runs per 100):** full phone number spoken 8 → 0;
«записал» spoken 4 → 0; saved the caller ID instead of the dictated number (`phone_ok`) 4 → 1;
`service_not_listed` ending with a message and no booking (`exactly_one_booking`/`no_messages`) 3 → 0;
`never_offers_this_number` 1 → 1; false acceptance claims 0 → 0 (none occurred in either 100). **Guard blocks
(the model's raw attempts):** `phone_digits` 9 → **0** (the prompt fix, removing the old «номера
телефонов словами» rule, removed the cause: the guard never had to act); `written_down` 5 → 6
(the model still says «записал» at the same rate, prompt or not; now every one is blocked,
before: spoken); `acceptance_claim` 0 → 0; `foreign_script` 0 → 0. **No fully blocked reply
happened, so the corrective round never ran in the sweeps** (it is covered by tests and by the live probe).
Warning `own_recap_before_prepare` 39 → 13 per 100 runs (the prompt cut it by two thirds, not to 0).
Caller side: 3 markers ignored, 11 goodbyes deferred (new); 2 / 9 (ref).
