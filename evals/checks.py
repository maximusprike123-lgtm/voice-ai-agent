"""Grading: pure functions from a finished call (`RunResult`) to pass/fail.

Everything here asserts OUTCOMES (what was saved, which tools ran in which order, what kinds of
things were or were not said), never exact wording. Each check has a stable name so pass rates
can be broken down per check. Two groups:

  - INVARIANTS: true of every call in every scenario (no foreign scripts, no invented prices...);
  - scenario checks, built from the small factories below and attached in scenarios.py.
"""

import re
from collections.abc import Callable, Iterable
from datetime import date, time

from agent.ru_words import longest_number_run, spoken_amounts
from agent.text_guard import ACCEPTANCE_CLAIM, ROLE_LEAKAGE, WRITTEN_DOWN, foreign_script_chars
from agent.tools import normalize_phone
from evals.caller import is_farewell
from evals.model import Check, CheckContext, CheckResult, Item, RunResult

PHONE_RUN_THRESHOLD = 6  # this many consecutive digits/number-words in one sentence = a phone
_MONEY_CUE = re.compile(r"рубл|\bруб\b|₽|тысяч|\bтыс\b|стоит|стоимост|цен[аыуе]\b", re.I)
_WRITTEN_DOWN = WRITTEN_DOWN
_CLAIMS_CONFIRMED = re.compile(r"\bподтвержд[её]н(?:а|о|ы)?\b", re.I)


def named(name: str) -> Callable[[Callable[[RunResult, CheckContext], CheckResult]], Check]:
    """Attach a stable name to a check function (used when reporting, and if it crashes)."""

    def decorate(fn):
        fn.check_name = name
        return fn

    return decorate


def result(name: str, passed: bool, detail: str = "") -> CheckResult:
    return CheckResult(name, passed, "" if passed else detail)


def _sentences_matching(run: RunResult, pattern: re.Pattern | str) -> list[str]:
    regex = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
    return [i.text for i in run.says() if regex.search(i.text)]


# --- Invariants (every run) -----------------------------------------------------------------------


@named("no_foreign_script")
def no_foreign_script(run: RunResult, ctx: CheckContext) -> CheckResult:
    bad = sorted({ch for i in run.says() for ch in foreign_script_chars(i.text)})
    return result("no_foreign_script", not bad, f"characters {bad} in spoken text")


@named("no_role_leakage")
def no_role_leakage(run: RunResult, ctx: CheckContext) -> CheckResult:
    """The agent never speaks a role label or the caller's words («userЗаписывай…»)."""
    for item in run.says():
        if ROLE_LEAKAGE.search(item.text):
            return result("no_role_leakage", False, f"spoke a role marker: {item.text!r}")
    return result("no_role_leakage", True)


@named("no_записал_before_confirm")
def no_written_down_before_confirm(run: RunResult, ctx: CheckContext) -> CheckResult:
    name = "no_записал_before_confirm"
    for item in run.items:
        if item.kind == "tool" and item.tool == "confirm_booking" and item.committed:
            break
        if item.kind == "say" and _WRITTEN_DOWN.search(item.text):
            return result(name, False, f"said {item.text!r} before the booking was confirmed")
    return result(name, True)


@named("no_invented_prices")
def no_invented_prices(run: RunResult, ctx: CheckContext) -> CheckResult:
    allowed = {
        price
        for service in ctx.business.services
        for price in (service.price_from, service.price_to)
        if price is not None
    }
    bad = []
    for item in run.says():
        if _MONEY_CUE.search(item.text):
            bad += [a for a in spoken_amounts(item.text) if a not in allowed]
    return result(
        "no_invented_prices", not bad, f"amounts {sorted(set(bad))} are not in the price list"
    )


@named("no_full_phone_spoken")
def no_full_phone_spoken(run: RunResult, ctx: CheckContext) -> CheckResult:
    for item in run.says():
        if longest_number_run(item.text) >= PHONE_RUN_THRESHOLD:
            return result("no_full_phone_spoken", False, f"digits read out: {item.text!r}")
    return result("no_full_phone_spoken", True)


@named("no_slot_confirmed_claim")
def no_slot_confirmed_claim(run: RunResult, ctx: CheckContext) -> CheckResult:
    for item in run.says():
        for match in _CLAIMS_CONFIRMED.finditer(item.text):
            if item.text[max(0, match.start() - 3) : match.start()].lower() != "не ":
                return result(
                    "no_slot_confirmed_claim", False, f"claimed a confirmation: {item.text!r}"
                )
    return result("no_slot_confirmed_claim", True)


@named("confirm_only_after_readback")
def confirm_only_after_readback(run: RunResult, ctx: CheckContext) -> CheckResult:
    """Every saved booking was preceded by a successful prepare_booking on an EARLIER turn and
    by a spoken read-back in between."""
    name = "confirm_only_after_readback"
    last_prepare: tuple[Item, int] | None = None  # (the draft's tool item, its index)
    for index, item in enumerate(run.items):
        if item.kind != "tool":
            continue
        if item.tool == "prepare_booking" and not item.is_error:
            last_prepare = (item, index)
        elif item.tool == "confirm_booking" and item.committed:
            if last_prepare is None:
                return result(name, False, "booking saved without any prepare_booking")
            prepared, prepared_at = last_prepare
            if prepared.turn >= item.turn:
                return result(name, False, "booking saved in the same turn as its draft")
            if not any(i.kind == "say" for i in run.items[prepared_at + 1 : index]):
                return result(name, False, "booking saved with no read-back spoken")
    return result(name, True)


@named("no_premature_confirm_attempt")
def no_premature_confirm_attempt(run: RunResult, ctx: CheckContext) -> CheckResult:
    """The code refuses confirm_booking in the turn of its prepare_booking; the model should not
    even try (it means it was about to skip the caller's «да»)."""
    tries = [
        i for i in run.tool_calls("confirm_booking") if i.is_error and "ещё не ответил" in i.text
    ]
    return result(
        "no_premature_confirm_attempt",
        not tries,
        f"{len(tries)} confirm attempt(s) in the draft's turn",
    )


@named("end_call_not_with_accepted_save")
def end_call_not_with_accepted_save(run: RunResult, ctx: CheckContext) -> CheckResult:
    for turn in {i.turn for i in run.items if i.kind == "tool" and i.committed}:
        if any(i.kind == "end" and i.turn == turn for i in run.items):
            return result(
                "end_call_not_with_accepted_save",
                False,
                f"hung up in turn {turn}, the turn of a save",
            )
    return result("end_call_not_with_accepted_save", True)


_ACCEPTANCE_CLAIM = ACCEPTANCE_CLAIM  # one definition, shared with the speech guard


@named("no_acceptance_claim_without_save")
def no_acceptance_claim_without_save(run: RunResult, ctx: CheckContext) -> CheckResult:
    """Never say the request was accepted/passed on before something was actually saved.
    (The sentences the code speaks after a save come after the committed tool result.)"""
    saved = False
    for item in run.items:
        if item.kind == "tool" and item.committed:
            saved = True
        elif item.kind == "say" and not saved:
            text = item.text.strip()
            if not text.endswith("?") and _ACCEPTANCE_CLAIM.search(text):
                return result(
                    "no_acceptance_claim_without_save",
                    False,
                    f"claimed acceptance with nothing saved: {item.text!r}",
                )
    return result("no_acceptance_claim_without_save", True)


_PHONE_IN_SPEECH = re.compile(r"\+?\d[\d\s\-()]{9,}\d")


def dictated_numbers(run: RunResult) -> list[str]:
    """Phone numbers the caller dictated, in order, as +7XXXXXXXXXX."""
    numbers = []
    for line in run.caller_lines:
        for match in _PHONE_IN_SPEECH.findall(line):
            if number := normalize_phone(match):
                numbers.append(number)
    return numbers


@named("saved_phone_is_the_dictated_number")
def saved_phone_is_the_dictated_number(run: RunResult, ctx: CheckContext) -> CheckResult:
    """If the caller dictated a phone number, that is the one that must be saved (not the number
    they called from). Nothing to check without a booking or a dictated number."""
    name = "saved_phone_is_the_dictated_number"
    numbers = dictated_numbers(run)
    if not numbers or not run.bookings:
        return result(name, True)
    dictated = numbers[-1]
    wrong = [b.phone for b in run.bookings if b.phone != dictated]
    return result(name, not wrong, f"the caller dictated {dictated} but {wrong} was saved")


INVARIANTS: tuple[Check, ...] = (
    no_foreign_script,
    no_role_leakage,
    no_written_down_before_confirm,
    no_invented_prices,
    no_full_phone_spoken,
    no_slot_confirmed_claim,
    confirm_only_after_readback,
    no_premature_confirm_attempt,
    end_call_not_with_accepted_save,
    no_acceptance_claim_without_save,
    saved_phone_is_the_dictated_number,
)
INVARIANT_NAMES = tuple(c.check_name for c in INVARIANTS)


# --- Factories for scenario checks ----------------------------------------------------------------


def exactly_one_booking() -> Check:
    @named("exactly_one_booking")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        return result(
            "exactly_one_booking", len(run.bookings) == 1, f"{len(run.bookings)} bookings saved"
        )

    return check


def no_bookings() -> Check:
    @named("no_bookings")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        return result("no_bookings", not run.bookings, f"{len(run.bookings)} booking(s) saved")

    return check


def no_messages() -> Check:
    @named("no_messages")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        return result("no_messages", not run.messages, f"{len(run.messages)} message(s) saved")

    return check


def exactly_one_message() -> Check:
    @named("exactly_one_message")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        return result(
            "exactly_one_message", len(run.messages) == 1, f"{len(run.messages)} messages saved"
        )

    return check


def agent_ended_call() -> Check:
    """When the caller said goodbye, the agent must have hung up. Nothing is required if the
    caller never said goodbye (they just thanked) or said it in the very turn of an accepted
    save (the code refuses end_call in that turn)."""

    @named("agent_ended_call")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        last = run.caller_lines[-1] if run.caller_lines else ""
        if not is_farewell(last):
            return result("agent_ended_call", True)
        last_turn = len(run.caller_lines)
        if any(i.kind == "tool" and i.committed and i.turn == last_turn for i in run.items):
            return result("agent_ended_call", True)  # the code forbids hanging up in a save turn
        return result(
            "agent_ended_call", run.ended_call(), "the caller said goodbye, the agent never hung up"
        )

    return check


def booking_field(name: str, predicate: Callable, describe: str) -> Check:
    """The (single) saved booking satisfies `predicate(booking)`. Fails if there is no booking."""

    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        if not run.bookings:
            return result(name, False, "no booking was saved")
        booking = run.bookings[-1]
        return result(name, bool(predicate(booking)), f"{describe}; got {_summary(booking)}")

    return check


def _summary(booking) -> str:
    return (
        f"service={booking.service_id} date={booking.preferred_date} "
        f"time={booking.preferred_time} period={booking.preferred_period} "
        f"phone={booking.phone} caller_phone={booking.caller_phone} notes={booking.notes!r} "
        f"name={booking.name!r} car={booking.car!r}"
    )


def service_in(ids: Iterable[str]) -> Check:
    allowed = set(ids)
    return booking_field(
        "service_ok",
        lambda b: b.service_id in allowed,
        f"service_id should be one of {sorted(allowed)}",
    )


def service_is(service_id: str) -> Check:
    return service_in([service_id])


def date_is(day: date) -> Check:
    return booking_field("date_ok", lambda b: b.preferred_date == day, f"date should be {day}")


def time_is(value: time) -> Check:
    return booking_field(
        "time_ok", lambda b: b.preferred_time == value, f"preferred_time should be {value:%H:%M}"
    )


def time_empty() -> Check:
    return booking_field(
        "time_empty", lambda b: b.preferred_time is None, "no exact time may be invented"
    )


def period_is(value: str) -> Check:
    return booking_field(
        "period_ok", lambda b: b.preferred_period == value, f"preferred_period should be {value!r}"
    )


def morning() -> Check:
    return booking_field(
        "morning",
        lambda b: (
            b.preferred_period == "утро" or (b.preferred_time and b.preferred_time < time(12))
        ),
        "should be a morning slot",
    )


def notes_match(pattern: str) -> Check:
    return booking_field(
        "notes_ok",
        lambda b: b.notes and re.search(pattern, b.notes, re.I),
        f"notes should mention /{pattern}/",
    )


def phone_is(e164: str) -> Check:
    return booking_field("phone_ok", lambda b: b.phone == e164, f"phone should be {e164}")


def caller_phone_is(value: str | None) -> Check:
    return booking_field(
        "caller_id_ok", lambda b: b.caller_phone == value, f"caller_phone should be {value}"
    )


def name_match(pattern: str) -> Check:
    return booking_field(
        "name_ok", lambda b: re.search(pattern, b.name, re.I), f"name should match /{pattern}/"
    )


def car_match(pattern: str) -> Check:
    return booking_field(
        "car_ok", lambda b: re.search(pattern, b.car, re.I), f"car should match /{pattern}/"
    )


def no_booking_on(day: date, name: str = "abandoned_slot_not_saved") -> Check:
    """Nothing was saved for a slot the caller gave up (e.g. after changing their mind)."""

    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        hit = [b for b in run.bookings if b.preferred_date == day]
        return result(name, not hit, f"a booking for the abandoned day {day} was saved")

    return check


def prepared_at_least(times: int) -> Check:
    name = f"prepared_at_least_{times}"

    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        ok_calls = [i for i in run.tool_calls("prepare_booking") if not i.is_error]
        return result(
            name, len(ok_calls) >= times, f"only {len(ok_calls)} successful prepare_booking"
        )

    return check


# «The quoted price is a range; the exact price is decided later»: the master, an inspection, or
# any equivalent wording («точная цена зависит от размера и состояния автомобиля»).
FINAL_PRICE_HEDGE = (
    r"мастер|осмотр|зависит от|окончательн|уточн|определ\w+|"
    r"точн\w+\s+(?:цен|стоимост)|ориентировочн|примерн"
)


def speech_matches(name: str, pattern: str) -> Check:
    """Some spoken sentence matches the pattern (case-insensitive)."""

    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        return result(
            name, bool(_sentences_matching(run, pattern)), f"nothing said matches /{pattern}/"
        )

    return check


def speech_never_matches(name: str, pattern: str) -> Check:
    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        hits = _sentences_matching(run, pattern)
        return result(name, not hits, f"said {hits[0]!r}" if hits else "")

    return check


def message_mentions(pattern: str) -> Check:
    @named("message_ok")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        ok = any(re.search(pattern, m.message, re.I) for m in run.messages)
        return result("message_ok", ok, f"no saved message mentions /{pattern}/")

    return check


def quotes_amount(amount: int) -> Check:
    name = f"quotes_{amount}"

    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        said = [a for i in run.says() for a in spoken_amounts(i.text)]
        return result(name, amount in said, f"amounts said: {sorted(set(said))}")

    return check


def no_amounts_spoken(name: str = "no_amounts_spoken") -> Check:
    @named(name)
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        said = sorted(
            {a for i in run.says() if _MONEY_CUE.search(i.text) for a in spoken_amounts(i.text)}
        )
        return result(name, not said, f"amounts said: {said}")

    return check


def no_terms_invented() -> Check:
    """No durations (years, months) in sentences about a warranty: the agent has no such facts."""

    @named("no_terms_invented")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        for item in run.says():
            mentions_warranty = re.search(r"гаранти", item.text, re.I)
            has_duration = re.search(r"\b(лет|год[аы]?|месяц\w*)\b", item.text, re.I)
            if mentions_warranty and has_duration and longest_number_run(item.text) > 0:
                return result("no_terms_invented", False, f"invented a term: {item.text!r}")
        return result("no_terms_invented", True)

    return check


def other_service_has_no_price(item_pattern: str | None = None) -> Check:
    """If the request went in as «other», the agent must not have priced the unlisted service.

    Quoting the price of a LISTED alternative is fine («такой услуги нет, есть оклейка защитной
    плёнкой от двадцати тысяч рублей»): every amount spoken must be in the price list (that is
    also the no_invented_prices invariant), and, if `item_pattern` names what the caller asked
    for («фар»), no clause about that item may carry an amount at all."""

    @named("other_service_no_price")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        if not any(b.service_id == "other" for b in run.bookings):
            return result("other_service_no_price", True)
        listed = {
            price
            for service in ctx.business.services
            for price in (service.price_from, service.price_to)
            if price is not None
        }
        for item in run.says():
            amounts = spoken_amounts(item.text) if _MONEY_CUE.search(item.text) else []
            if any(a not in listed for a in amounts):
                return result(
                    "other_service_no_price",
                    False,
                    f"an amount not in the price list: {item.text!r}",
                )
            if amounts and item_pattern:
                # Clause by clause (commas, semicolons; NOT dashes: «оклейка фар — от 20 000 ₽»
                # prices the item), so «оклейка плёнкой — от 20 000 ₽, а фары уточнит мастер»
                # does not count as pricing the headlights.
                for clause in re.split(r"[,;]", item.text):
                    if re.search(item_pattern, clause, re.I) and spoken_amounts(clause):
                        return result(
                            "other_service_no_price",
                            False,
                            f"priced the unlisted service /{item_pattern}/: {item.text!r}",
                        )
        return result("other_service_no_price", True)

    return check


def other_service_notes_match(pattern: str) -> Check:
    @named("other_service_notes")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        others = [b for b in run.bookings if b.service_id == "other"]
        if not others:
            return result("other_service_notes", True)
        ok = all(b.notes and re.search(pattern, b.notes, re.I) for b in others)
        return result("other_service_notes", ok, f"notes of an «other» booking lack /{pattern}/")

    return check


# --- sunday_closed --------------------------------------------------------------------------------


def no_booking_on_closed_day() -> Check:
    @named("no_booking_on_closed_day")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        from agent.business import Weekday

        bad = [
            b.preferred_date
            for b in run.bookings
            if ctx.business.hours[Weekday.from_index(b.preferred_date.weekday())] is None
        ]
        return result("no_booking_on_closed_day", not bad, f"booked on closed day(s) {bad}")

    return check


def one_booking_on_an_open_day() -> Check:
    @named("one_booking_open_day")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        from agent.business import Weekday

        if len(run.bookings) != 1:
            return result("one_booking_open_day", False, f"{len(run.bookings)} bookings saved")
        day = run.bookings[0].preferred_date
        open_day = ctx.business.hours[Weekday.from_index(day.weekday())] is not None
        return result(
            "one_booking_open_day",
            open_day and day >= ctx.now.date(),
            f"booking on {day}, open={open_day}",
        )

    return check


def says_closed_that_day() -> Check:
    return speech_matches(
        "says_closed",
        r"воскресень[^.?!]*(не работа|выходн|закрыт)|(не работа|выходн|закрыт)[^.?!]*воскресень",
    )


def offers_another_day() -> Check:
    return speech_matches(
        "offers_another_day",
        r"друг\w+ (день|дат)|понедельник|суббот|пятниц|в любой (другой )?день|на какой (день|дату)"
        r"|какой день|какую дату",
    )


_ASKS_FOR_NUMBER = re.compile(
    r"продикт|(?:назов|подскаж|укаж|сообщ|оставьте)\w*[^.?!]*(?:номер|телефон)"
    r"|какой[^.?!]*(?:номер|телефон)|на какой номер|номер (?:телефона )?для связи",
    re.I,
)


def _gives_number(line: str) -> bool:
    return len(re.sub(r"\D", "", line)) >= 10


def asks_to_dictate_number() -> Check:
    """With a hidden caller ID the agent must get a number from the caller. Satisfied if it
    asked for one, or if the caller volunteered a number (then there was nothing to ask)."""

    @named("asks_for_a_number")
    def check(run: RunResult, ctx: CheckContext) -> CheckResult:
        if any(_gives_number(line) for line in run.caller_lines):
            asked = True  # volunteered; whether or not it was asked for first
        else:
            asked = bool(_sentences_matching(run, _ASKS_FOR_NUMBER))
        return result("asks_for_a_number", asked, "no number was asked for or given")

    return check


def never_offers_this_number() -> Check:
    return speech_never_matches(
        "never_offers_this_number",
        r"с которого вы звоните|на этот номер|ваш номер (определ|высвечива)",
    )


# --- Warning metrics (reported, never pass/fail) --------------------------------------------------

_OWN_RECAP = re.compile(r"\b(?:верно|правильно)\s*\?\s*$|^\s*уточню\b", re.I)


@named("own_recap_before_prepare")
def own_recap_before_prepare(run: RunResult, ctx: CheckContext) -> CheckResult:
    """The model recapped the details itself («Уточню: … Верно?») before calling prepare_booking,
    although the system reads the request back. Harmless but redundant, so only a warning:
    `passed` means "no warning"."""
    name = "own_recap_before_prepare"
    for index, item in enumerate(run.items):
        if item.kind == "tool" and item.tool == "prepare_booking" and not item.is_error:
            for earlier in run.items[:index]:
                if earlier.kind == "say" and _OWN_RECAP.search(earlier.text):
                    return CheckResult(name, False, f"said {earlier.text!r} before prepare_booking")
            return CheckResult(name, True)
    return CheckResult(name, True)  # no booking was prepared: not applicable


WARNINGS: tuple[Check, ...] = (own_recap_before_prepare,)
WARNING_NAMES = tuple(w.check_name for w in WARNINGS)


def run_warnings(run: RunResult, ctx: CheckContext) -> list[CheckResult]:
    results = []
    for warning in WARNINGS:
        try:
            results.append(warning(run, ctx))
        except Exception as exc:
            results.append(CheckResult(warning.check_name, False, f"crashed: {exc}"))
    return results


# --- Running checks -------------------------------------------------------------------------------


def grade(run: RunResult, scenario_checks: Iterable[Check], ctx: CheckContext) -> list[CheckResult]:
    """Run the invariants and the scenario's checks. A check that crashes counts as failed."""
    results = []
    for check in (*INVARIANTS, *scenario_checks):
        name = getattr(check, "check_name", getattr(check, "__name__", "check"))
        try:
            results.append(check(run, ctx))
        except Exception as exc:  # a grader bug must not hide as a pass
            results.append(CheckResult(name, False, f"check crashed: {type(exc).__name__}: {exc}"))
    return results
