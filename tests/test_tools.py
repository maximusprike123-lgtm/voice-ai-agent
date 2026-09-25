"""Offline tests for the tools: two-step booking, validation, read-back, static schemas."""

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import OTHER_SERVICE_ID, load_business_config
from agent.dialogue import DialogueEngine, EndCall, Say, ToolResult
from agent.llm import Message, Role, StreamEnd, TextDelta, ToolCall, ToolCallEvent
from agent.records import Booking, InMemorySink
from agent.tools import (
    ERROR_PREFIX,
    MAX_HORIZON_DAYS,
    PERIODS,
    ToolRegistry,
    build_tool_specs,
    normalize_phone,
)

REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"
MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 24, 17, 5, tzinfo=MOSCOW)  # Thursday; Mon-Fri 10-21, Sat 10-20, Sun off
FRIDAY, SATURDAY, SUNDAY = "2026-09-25", "2026-09-26", "2026-09-27"

CONFIRMED_SAY = (
    "Заявка принята и передана администратору, он перезвонит для подтверждения. "
    "Могу ещё чем-то помочь?"
)
DEFAULT_READ_BACK = (
    "Проверьте, пожалуйста: Игорь, полировка кузова, автомобиль Toyota Camry, "
    "в пятницу, двадцать пятого сентября, в четырнадцать часов. "
    "Номер телефона заканчивается на четыре пять шесть семь. "
    "Всё верно?"
)


@pytest.fixture(scope="module")
def business():
    return load_business_config(REPO_CONFIG)


@pytest.fixture
def sink():
    return InMemorySink()


@pytest.fixture
def tools(business, sink):
    return ToolRegistry(business, sink, clock=lambda: NOW, caller_phone="+79991234567")


def valid_args(**overrides):
    args = {
        "name": "Игорь",
        "phone": "8 (916) 123-45-67",
        "car": "Toyota Camry",
        "service_id": "polishing",
        "preferred_date": FRIDAY,
        "preferred_time": "14:00",
    }
    args.update(overrides)
    return {k: v for k, v in args.items() if v is not None}


async def call(tools, name, args=None, raw=None):
    arguments = raw if raw is not None else json.dumps(args or {}, ensure_ascii=False)
    return await tools.execute(ToolCall("call_1", name, arguments))


async def prepare(tools, **overrides):
    return await call(tools, "prepare_booking", valid_args(**overrides))


async def confirm(tools):
    """confirm_booking as the model calls it: on the turn AFTER the read-back («да»)."""
    tools.begin_turn()
    return await call(tools, "confirm_booking")


async def book(tools, **overrides):
    """prepare + confirm; returns the prepare error if validation failed, else confirm's outcome."""
    prepared = await prepare(tools, **overrides)
    return prepared if is_error(prepared) else await confirm(tools)


def is_error(outcome) -> bool:
    return outcome.result.startswith(ERROR_PREFIX)


# --- Phone normalization --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "+79161234567",
        "89161234567",
        "79161234567",
        "9161234567",
        "8 (916) 123-45-67",
        "+7 916 123 45 67",
        "8-916-123-45-67",
        "+7(916)123-45-67",
    ],
)
def test_phone_formats_normalize(raw):
    assert normalize_phone(raw) == "+79161234567"


@pytest.mark.parametrize(
    "raw", ["", "abc", "12345", "916123456", "891612345678", "+19161234567", "0161234567"]
)
def test_bad_phones_are_rejected(raw):
    assert normalize_phone(raw) is None


# --- Step 1: prepare_booking ---------------------------------------------------------------------


async def test_prepare_returns_a_code_built_read_back_and_saves_nothing(tools, sink):
    outcome = await prepare(tools)

    assert not is_error(outcome) and not outcome.ends_call
    assert outcome.say == DEFAULT_READ_BACK
    assert sink.bookings == []  # nothing is saved before confirm_booking


async def test_prepare_result_tells_the_model_not_to_repeat_the_read_back(tools):
    result = (await prepare(tools)).result
    assert "уже зачитан" in result and "confirm_booking" in result
    assert DEFAULT_READ_BACK not in result


async def test_read_back_has_no_digits_at_all(tools):
    assert not any(ch.isdigit() for ch in (await prepare(tools)).say)


@pytest.mark.parametrize(
    ("phone", "spoken"),
    [
        ("+79161234567", "четыре пять шесть семь"),
        ("8 916 000 00 01", "ноль ноль ноль один"),
        ("89990001122", "один один два два"),
        ("9161234500", "четыре пять ноль ноль"),
    ],
)
async def test_read_back_always_says_exactly_the_last_four_digits(tools, phone, spoken):
    say = (await prepare(tools, phone=phone)).say
    tail = say.split("заканчивается на ")[1].split(".")[0]
    assert tail == spoken
    assert len(tail.split()) == 4


async def test_read_back_with_a_period_instead_of_a_time(tools):
    say = (
        await prepare(tools, preferred_date=SATURDAY, preferred_time=None, preferred_period="день")
    ).say
    assert "в субботу, двадцать шестого сентября, днём." in say
    assert " в четырнадцать" not in say


@pytest.mark.parametrize(
    ("period", "phrase"),
    [("утро", "утром"), ("день", "днём"), ("вечер", "вечером"), ("любое", "в любое время")],
)
async def test_every_period_is_read_back_in_words(tools, period, phrase):
    say = (await prepare(tools, preferred_time=None, preferred_period=period)).say
    assert f"двадцать пятого сентября, {phrase}." in say


async def test_exact_time_wins_in_the_read_back_when_both_are_given(tools, sink):
    outcome = await prepare(tools, preferred_time="14:30", preferred_period="день")

    assert "в четырнадцать тридцать." in outcome.say and "днём" not in outcome.say
    await confirm(tools)
    assert sink.bookings[0].preferred_time == time(14, 30)  # both are stored
    assert sink.bookings[0].preferred_period == "день"


async def test_read_back_for_other_uses_the_callers_description_and_no_price(tools):
    outcome = await prepare(tools, service_id="other", notes="убрать запах после собаки")

    assert "другая услуга, вы описали так: убрать запах после собаки" in outcome.say
    assert "₽" not in outcome.say and "руб" not in outcome.say


async def test_second_prepare_replaces_the_draft(tools, sink):
    await prepare(tools, car="Toyota Camry")
    await prepare(tools, car="BMW X5")
    await confirm(tools)

    assert [b.car for b in sink.bookings] == ["BMW X5"]


async def test_failed_prepare_discards_the_previous_draft(tools, sink):
    await prepare(tools)
    failed = await prepare(tools, phone="12345")
    assert is_error(failed)

    assert is_error(await confirm(tools))  # the stale draft must not be confirmable
    assert sink.bookings == []


# --- Step 2: confirm_booking ----------------------------------------------------------------------


async def test_confirm_saves_the_prepared_draft_with_normalized_fields(tools, sink):
    await prepare(tools)
    outcome = await confirm(tools)

    assert not is_error(outcome) and not outcome.ends_call
    assert outcome.say == CONFIRMED_SAY and outcome.committed
    assert sink.bookings == [
        Booking(
            name="Игорь",
            phone="+79161234567",
            car="Toyota Camry",
            service_id="polishing",
            service_name="Полировка кузова",
            preferred_date=date(2026, 9, 25),
            preferred_time=time(14, 0),
            preferred_period=None,
            notes=None,
            caller_phone="+79991234567",
            created_at=NOW,
        )
    ]


async def test_confirm_without_a_draft_is_an_error(tools, sink):
    outcome = await confirm(tools)
    assert is_error(outcome) and "prepare_booking" in outcome.result
    assert sink.bookings == []


async def test_confirm_takes_no_arguments_and_ignores_any(tools, sink):
    await prepare(tools)
    tools.begin_turn()
    outcome = await call(tools, "confirm_booking", {"name": "Другой человек"})
    assert not is_error(outcome)
    assert sink.bookings[0].name == "Игорь"


async def test_confirm_consumes_the_draft(tools, sink):
    await prepare(tools)
    await confirm(tools)

    assert is_error(await confirm(tools))
    assert len(sink.bookings) == 1


async def test_confirm_revalidates_and_rejects_a_draft_that_went_stale(business, sink):
    clock = {"now": NOW}
    tools = ToolRegistry(business, sink, clock=lambda: clock["now"])
    await prepare(tools, preferred_date="2026-09-24", preferred_time="18:00")

    clock["now"] = NOW + timedelta(hours=2)  # 19:05: 18:00 has passed while the caller thought
    outcome = await confirm(tools)

    assert is_error(outcome) and "больше не действительна" in outcome.result
    assert "уже прошло" in outcome.result
    assert sink.bookings == []
    assert is_error(await confirm(tools))  # the draft is gone


async def test_what_the_caller_hears_never_claims_the_slot_is_confirmed(tools):
    await prepare(tools)
    say = (await confirm(tools)).say

    assert "перезвонит для подтверждения" in say
    for claim in ("подтверждена", "подтверждено", "подтверждён", "записан"):
        assert claim not in say


async def test_price_restriction_applies_only_to_the_other_service(tools):
    listed = (await book(tools)).result
    assert "Цену не называй" not in listed and "₽" not in listed

    other = (await book(tools, service_id="other", notes="нужна консультация")).result
    assert "Цену не называй" in other and "₽" not in other


async def test_result_mentions_missing_exact_time_only_when_there_is_none(tools):
    assert "Точное время не указано" not in (await book(tools)).result
    period_only = await book(tools, car="BMW", preferred_time=None, preferred_period="утро")
    assert "Точное время не указано" in period_only.result


# --- Duplicates and sink failure ------------------------------------------------------------------


async def test_identical_booking_is_saved_once(tools, sink):
    first = await book(tools)
    second = await book(tools)

    assert not is_error(first) and not is_error(second)
    assert "уже принята" in second.result
    assert len(sink.bookings) == 1


async def test_different_booking_in_the_same_call_is_saved_too(tools, sink):
    await book(tools)
    await book(tools, car="BMW X5")
    assert len(sink.bookings) == 2


class BrokenSink:
    async def add_booking(self, booking):
        raise RuntimeError("disk full")

    async def add_message(self, message):
        raise RuntimeError("disk full")


async def test_sink_failure_is_reported_keeps_the_draft_and_allows_a_retry(business):
    tools = ToolRegistry(business, BrokenSink(), clock=lambda: NOW)
    await prepare(tools)
    failed = await confirm(tools)
    assert is_error(failed) and "не удалось" in failed.result

    working = InMemorySink()
    tools._sink = working
    assert not is_error(await confirm(tools))  # same draft, no need to prepare again
    assert len(working.bookings) == 1


# --- Validation (through prepare_booking) ---------------------------------------------------------


async def test_exact_time_is_optional_when_a_period_is_given(tools):
    assert not is_error(await prepare(tools, preferred_time=None, preferred_period="день"))


async def test_neither_time_nor_period_is_rejected(tools):
    outcome = await prepare(tools, preferred_time=None)
    assert is_error(outcome)
    assert "preferred_time" in outcome.result and "preferred_period" in outcome.result
    assert "не придумывай" in outcome.result.lower()


@pytest.mark.parametrize("empty", ["", "   "])
async def test_blank_time_and_period_count_as_missing(tools, empty):
    outcome = await prepare(tools, preferred_time=empty, preferred_period=empty)
    assert is_error(outcome) and "preferred_period" in outcome.result


async def test_unknown_period_is_rejected_and_the_choices_are_listed(tools):
    outcome = await prepare(tools, preferred_time=None, preferred_period="после обеда")
    assert is_error(outcome)
    for allowed in PERIODS:
        assert allowed in outcome.result


async def test_bad_time_alone_does_not_also_report_a_missing_time(tools):
    outcome = await prepare(tools, preferred_time="25:00")
    assert is_error(outcome) and "HH:MM" in outcome.result
    assert "нужно указать время" not in outcome.result


async def test_time_boundaries_open_inclusive_close_exclusive(tools):
    assert not is_error(await prepare(tools, preferred_time="10:00"))
    assert not is_error(await prepare(tools, preferred_time="20:59"))
    assert is_error(await prepare(tools, preferred_time="21:00"))
    assert is_error(await prepare(tools, preferred_time="09:59"))


async def test_saturday_closes_earlier(tools):
    assert not is_error(await prepare(tools, preferred_date=SATURDAY, preferred_time="19:30"))
    outcome = await prepare(tools, preferred_date=SATURDAY, preferred_time="20:00")
    assert is_error(outcome) and "10:00–20:00" in outcome.result


async def test_single_digit_hour_and_numeric_phone_are_accepted(tools, sink):
    outcome = await prepare(tools, preferred_time="9:30", phone=89161234567)
    assert is_error(outcome)  # 09:30 is before opening...
    assert "phone" not in outcome.result  # ...but the numeric phone itself was fine
    assert not is_error(await book(tools, preferred_time="11:05", phone=89161234567))
    assert sink.bookings[0].preferred_time == time(11, 5)
    assert sink.bookings[0].phone == "+79161234567"


async def test_past_date_is_rejected(tools, sink):
    outcome = await prepare(tools, preferred_date="2026-09-23")
    assert is_error(outcome) and "прошла" in outcome.result
    assert sink.bookings == []


async def test_today_is_allowed_but_a_time_already_passed_is_not(tools):
    assert not is_error(await prepare(tools, preferred_date="2026-09-24", preferred_time="18:00"))
    outcome = await prepare(tools, preferred_date="2026-09-24", preferred_time="12:00")
    assert is_error(outcome) and "уже прошло" in outcome.result
    assert is_error(await prepare(tools, preferred_date="2026-09-24", preferred_time="17:05"))


async def test_today_with_a_period_is_allowed(tools):
    assert not is_error(
        await prepare(
            tools, preferred_date="2026-09-24", preferred_time=None, preferred_period="вечер"
        )
    )


async def test_closed_day_is_rejected(tools):
    outcome = await prepare(
        tools, preferred_date=SUNDAY, preferred_time=None, preferred_period="любое"
    )
    assert is_error(outcome)
    assert "воскресенье" in outcome.result and "не работаем" in outcome.result


async def test_horizon_limit(tools):
    # NOW + 90 days is Wednesday 2026-12-23; the day after is the first one over the limit.
    assert not is_error(await prepare(tools, preferred_date="2026-12-23"))
    outcome = await prepare(tools, preferred_date="2026-12-24")
    assert is_error(outcome) and str(MAX_HORIZON_DAYS) in outcome.result


async def test_year_typo_is_caught_by_horizon(tools):
    assert is_error(await prepare(tools, preferred_date="2027-09-25"))


@pytest.mark.parametrize("bad", ["25.09.2026", "завтра", "2026-9-25", "2026-02-30", "20260925"])
async def test_unparsable_date_is_rejected(tools, bad):
    outcome = await prepare(tools, preferred_date=bad)
    assert is_error(outcome) and "YYYY-MM-DD" in outcome.result


@pytest.mark.parametrize("bad", ["после обеда", "25:00", "14:60", "14", "два часа"])
async def test_unparsable_time_is_rejected_with_a_hint_about_period(tools, bad):
    outcome = await prepare(tools, preferred_time=bad)
    assert is_error(outcome) and "preferred_period" in outcome.result


async def test_unknown_service_is_rejected_and_valid_ids_are_listed(tools):
    outcome = await prepare(tools, service_id="rocket_wash")
    assert is_error(outcome)
    assert "rocket_wash" in outcome.result and "polishing" in outcome.result
    assert OTHER_SERVICE_ID in outcome.result


async def test_bad_phone_is_rejected(tools, sink):
    outcome = await prepare(tools, phone="12345")
    assert is_error(outcome) and "phone" in outcome.result


@pytest.mark.parametrize("field", ["name", "phone", "car", "service_id", "preferred_date"])
async def test_missing_required_field_is_rejected(tools, field):
    outcome = await prepare(tools, **{field: None})
    assert is_error(outcome) and field in outcome.result


async def test_all_problems_are_reported_together(tools):
    outcome = await prepare(tools, name=None, phone="1", preferred_date=SUNDAY, service_id="nope")
    for field in ("name", "phone", "service_id", "preferred_date"):
        assert field in outcome.result


async def test_overlong_name_is_rejected(tools):
    assert is_error(await prepare(tools, name="И" * 101))


async def test_non_string_field_is_rejected(tools):
    outcome = await prepare(tools, car={"make": "Toyota"})
    assert is_error(outcome) and "car" in outcome.result


async def test_other_service_records_the_description_and_invents_no_price(tools, sink):
    outcome = await book(
        tools, service_id="other", notes="нужно убрать запах после перевозки собаки"
    )

    assert not is_error(outcome)
    booking = sink.bookings[0]
    assert booking.service_id == OTHER_SERVICE_ID
    assert booking.notes == "нужно убрать запах после перевозки собаки"


async def test_other_service_requires_notes(tools, sink):
    outcome = await prepare(tools, service_id="other")
    assert is_error(outcome) and "notes" in outcome.result


async def test_callers_words_are_kept_in_notes(tools, sink):
    await book(tools, preferred_time=None, preferred_period="день", notes="после обеда")
    assert sink.bookings[0].notes == "после обеда"
    assert sink.bookings[0].preferred_time is None


# --- take_message / end_call ----------------------------------------------------------------------


async def test_take_message_saves_a_record(tools, sink):
    outcome = await call(
        tools,
        "take_message",
        {"message": "Хочу узнать про скидки", "name": "Анна", "phone": "8 916 123 45 67"},
    )

    assert not is_error(outcome) and not outcome.ends_call
    record = sink.messages[0]
    assert (record.message, record.name, record.phone) == (
        "Хочу узнать про скидки",
        "Анна",
        "+79161234567",
    )
    assert record.caller_phone == "+79991234567" and record.created_at == NOW


async def test_take_message_requires_message(tools, sink):
    outcome = await call(tools, "take_message", {"name": "Анна"})
    assert is_error(outcome) and "message" in outcome.result
    assert sink.messages == []


async def test_take_message_keeps_the_text_even_if_the_phone_is_garbage(tools, sink):
    outcome = await call(tools, "take_message", {"message": "Перезвоните", "phone": "как-нибудь"})
    assert not is_error(outcome)
    assert sink.messages[0].phone is None
    assert "как-нибудь" in sink.messages[0].message


async def test_end_call_ends_the_call(tools):
    outcome = await call(tools, "end_call", {})
    assert outcome.ends_call and not is_error(outcome)


@pytest.mark.parametrize("raw", ["", "  "])
async def test_end_call_accepts_empty_arguments(tools, raw):
    assert (await call(tools, "end_call", raw=raw)).ends_call


# --- end_call guard: not in the same turn as an accepted booking ----------------------------------


async def test_end_call_is_rejected_in_the_same_turn_as_a_successful_confirm(tools, sink):
    tools.begin_turn()
    await prepare(tools)
    tools.begin_turn()  # the caller said «да»
    assert not is_error(await confirm(tools))

    outcome = await call(tools, "end_call")

    assert is_error(outcome) and not outcome.ends_call
    assert "дождись ответа" in outcome.result
    assert len(sink.bookings) == 1  # the booking itself is unaffected


async def test_end_call_is_allowed_once_the_caller_has_spoken_again(tools):
    tools.begin_turn()
    await book(tools)
    tools.begin_turn()  # «нет, больше ничего, спасибо»

    outcome = await call(tools, "end_call")

    assert outcome.ends_call and not is_error(outcome)


async def test_end_call_stays_rejected_for_the_rest_of_the_confirming_turn(tools):
    tools.begin_turn()
    await book(tools)

    assert is_error(await call(tools, "end_call"))
    assert is_error(await call(tools, "end_call"))  # retrying in the same turn changes nothing


async def test_end_call_without_any_booking_is_always_allowed(tools):
    tools.begin_turn()
    assert (await call(tools, "end_call")).ends_call


async def test_end_call_is_allowed_after_a_failed_confirm(business):
    tools = ToolRegistry(business, BrokenSink(), clock=lambda: NOW)
    tools.begin_turn()
    await prepare(tools)
    assert is_error(await confirm(tools))  # nothing was accepted

    assert (await call(tools, "end_call")).ends_call


async def test_end_call_is_allowed_after_a_rejected_confirm_without_a_draft(tools):
    tools.begin_turn()
    assert is_error(await confirm(tools))
    assert (await call(tools, "end_call")).ends_call


async def test_a_duplicate_confirm_counts_as_accepted_for_the_guard(tools):
    tools.begin_turn()
    await book(tools)
    tools.begin_turn()  # the caller changes their mind and asks again with the same data
    assert "уже принята" in (await book(tools)).result

    assert is_error(await call(tools, "end_call"))


async def test_a_second_booking_later_in_the_call_is_guarded_again(tools):
    tools.begin_turn()
    await book(tools)
    tools.begin_turn()  # «да, ещё одну машину»
    tools.begin_turn()
    await book(tools, car="BMW X5")

    assert is_error(await call(tools, "end_call"))
    tools.begin_turn()
    assert (await call(tools, "end_call")).ends_call


async def test_the_success_result_tells_the_model_the_caller_was_already_told_and_to_wait(tools):
    await prepare(tools)
    result = (await confirm(tools)).result
    assert "Клиенту уже сообщено" in result and "Ничего не добавляй" in result
    assert "end_call вызывать нельзя" in result


def test_the_end_call_schema_stays_static_and_mentions_the_rule(business):
    spec = build_tool_specs(business)[3]
    assert spec.name == "end_call" and spec.parameters == {"type": "object", "properties": {}}
    assert "дождись ответа" in spec.description


# --- What the caller hears after a save is built by code ------------------------------------------


async def test_time_note_is_added_when_only_a_period_was_given(tools):
    await prepare(tools, preferred_time=None, preferred_period="день")
    outcome = await confirm(tools)

    assert outcome.say == (
        "Заявка принята и передана администратору, он перезвонит для подтверждения. "
        "Точное время администратор уточнит при звонке. Могу ещё чем-то помочь?"
    )


async def test_the_other_service_says_the_same_and_never_names_a_price(tools):
    await prepare(tools, service_id="other", notes="нужна консультация", preferred_time="14:00")
    outcome = await confirm(tools)

    assert outcome.say == CONFIRMED_SAY
    assert "₽" not in outcome.say and "руб" not in outcome.say


async def test_the_confirmation_wording_is_gender_neutral_and_has_no_digits(tools):
    await prepare(tools, preferred_time=None, preferred_period="утро")
    say = (await confirm(tools)).say
    await call(tools, "take_message", {"message": "вопрос"})

    from agent.text_guard import GUARD_FALLBACKS
    from agent.tools import (
        BOOKING_ACCEPTED_SAY,
        BOOKING_ALREADY_ACCEPTED_SAY,
        MESSAGE_TAKEN_SAY,
    )

    everything_the_code_says = (
        say,
        BOOKING_ACCEPTED_SAY,
        BOOKING_ALREADY_ACCEPTED_SAY,
        MESSAGE_TAKEN_SAY,
        *GUARD_FALLBACKS.values(),  # what is said when the speech guard blocked a whole reply
    )
    for text_ in everything_the_code_says:
        assert not any(ch.isdigit() for ch in text_)
        for gendered in ("принял", "записал", "передал", "понял", "расслышал", "уточнил"):
            assert gendered not in text_.lower()


def test_the_guard_fallbacks_never_blame_the_caller():
    from agent.text_guard import GUARD_FALLBACKS

    for phrase in GUARD_FALLBACKS.values():
        for blaming in ("ошиб", "неправильно", "неверно", "вы не ", "вы сказали", "вы неясно"):
            assert blaming not in phrase.lower()


async def test_a_duplicate_confirm_also_speaks_from_code(tools):
    await book(tools)
    outcome = await book(tools)

    assert outcome.say == (
        "Эта заявка уже принята, администратор перезвонит для подтверждения. "
        "Могу ещё чем-то помочь?"
    )
    assert outcome.committed and not is_error(outcome)


async def test_take_message_speaks_from_code_and_commits(tools, sink):
    outcome = await call(tools, "take_message", {"message": "Хочу узнать про скидки"})

    assert outcome.say == (
        "Сообщение передано администратору, он свяжется с вами. Могу ещё чем-то помочь?"
    )
    assert outcome.committed and not outcome.ends_call
    assert "Клиенту уже сообщено" in outcome.result and "Ничего не добавляй" in outcome.result
    assert len(sink.messages) == 1


async def test_only_saves_are_committed(tools, business):
    assert not (await prepare(tools)).committed  # a draft is not a change
    assert (await confirm(tools)).committed
    assert not (await call(tools, "end_call")).committed
    assert not (await confirm(tools)).committed  # no draft any more: an error
    assert not (await call(tools, "take_message", {})).committed  # missing message: an error


async def test_failures_have_no_say_and_are_not_committed(business):
    tools = ToolRegistry(business, BrokenSink(), clock=lambda: NOW)
    await prepare(tools)
    failed_booking = await confirm(tools)
    failed_message = await call(tools, "take_message", {"message": "вопрос"})

    for outcome in (failed_booking, failed_message):
        assert is_error(outcome) and outcome.say is None and not outcome.committed


async def test_validation_errors_and_prepare_have_no_committed_flag(tools):
    assert not (await prepare(tools, phone="12")).committed
    assert (await prepare(tools)).say is not None  # the read-back, but still not a commit


async def test_take_message_through_the_engine_ends_the_turn_without_another_llm_call(
    business, sink
):
    llm = ScriptedLLM(
        [
            ToolCallEvent(ToolCall("c1", "take_message", '{"message": "Есть ли скидки?"}')),
            StreamEnd("tool_calls"),
        ],
    )
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    events = [e async for e in engine.respond("Есть ли скидки?")]

    assert isinstance(events[0], ToolResult) and events[0].committed
    assert " ".join(e.text for e in events if isinstance(e, Say)) == (
        "Сообщение передано администратору, он свяжется с вами. Могу ещё чем-то помочь?"
    )
    assert len(llm.calls) == 1 and len(sink.messages) == 1


async def test_the_caller_can_hang_up_on_the_turn_after_the_acceptance(business, sink):
    llm = ScriptedLLM(
        tool_round(prepare_call()),
        tool_round(ToolCall("c2", "confirm_booking", "{}")),
        [
            TextDelta("Всего доброго!"),
            ToolCallEvent(ToolCall("c3", "end_call", "{}")),
            StreamEnd("tool_calls"),
        ],
    )
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    await collect_events(engine, "Запишите меня")
    await collect_events(engine, "Да, всё верно")
    last = await collect_events(engine, "Нет, спасибо")

    assert last[-1] == EndCall() and Say("Всего доброго!") in last


# --- confirm_booking only after the caller had a turn to answer the read-back ---------------------


async def test_confirm_in_the_same_turn_as_prepare_is_refused_and_saves_nothing(tools, sink):
    tools.begin_turn()
    prepared = await call(tools, "prepare_booking", valid_args())
    outcome = await call(tools, "confirm_booking")  # the model calls both in one go

    assert not is_error(prepared) and prepared.say == DEFAULT_READ_BACK
    assert is_error(outcome) and not outcome.committed and outcome.say is None
    assert "ещё не ответил" in outcome.result and "в следующей реплике" in outcome.result
    assert sink.bookings == []


async def test_the_refused_draft_is_kept_and_confirms_on_the_next_turn(tools, sink):
    tools.begin_turn()
    await call(tools, "prepare_booking", valid_args())
    assert is_error(await call(tools, "confirm_booking"))

    tools.begin_turn()  # the caller says «да»
    outcome = await call(tools, "confirm_booking")

    assert not is_error(outcome) and outcome.committed
    assert len(sink.bookings) == 1


async def test_a_refused_confirm_does_not_arm_the_end_call_guard(tools):
    tools.begin_turn()
    await call(tools, "prepare_booking", valid_args())
    await call(tools, "confirm_booking")  # refused: nothing was accepted
    assert (await call(tools, "end_call")).ends_call


async def test_a_reprepared_draft_needs_its_own_later_turn(tools, sink):
    tools.begin_turn()
    await call(tools, "prepare_booking", valid_args(car="Toyota Camry"))
    tools.begin_turn()  # the caller corrects something instead of confirming
    await call(tools, "prepare_booking", valid_args(car="BMW X5"))

    assert is_error(await call(tools, "confirm_booking"))  # same turn as the new prepare
    tools.begin_turn()
    assert not is_error(await call(tools, "confirm_booking"))
    assert [b.car for b in sink.bookings] == ["BMW X5"]


async def test_the_model_cannot_skip_the_callers_yes_through_the_engine(business, sink):
    """prepare + confirm in ONE model round: only the read-back is spoken, nothing is saved."""
    llm = ScriptedLLM(
        [
            ToolCallEvent(prepare_call()),
            ToolCallEvent(ToolCall("c2", "confirm_booking", "{}")),
            StreamEnd("tool_calls"),
        ],
        tool_round(ToolCall("c3", "confirm_booking", "{}")),
    )
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    first = await collect_events(engine, "Запишите меня, всё как обычно")

    results = [e for e in first if isinstance(e, ToolResult)]
    assert results[0].committed is False and results[1].result.startswith(ERROR_PREFIX)
    assert " ".join(e.text for e in first if isinstance(e, Say)) == DEFAULT_READ_BACK
    assert sink.bookings == [] and len(llm.calls) == 1

    second = await collect_events(engine, "Да, всё верно")  # now the caller really answered
    assert len(sink.bookings) == 1 and second[0].committed


# --- Malformed calls ------------------------------------------------------------------------------


async def test_invalid_json_is_an_error_result_not_an_exception(tools):
    outcome = await call(tools, "prepare_booking", raw='{"name": "Игорь"')
    assert is_error(outcome) and "JSON" in outcome.result


async def test_non_object_arguments_are_rejected(tools):
    assert is_error(await call(tools, "prepare_booking", raw='["Игорь"]'))


async def test_submit_booking_no_longer_exists(tools):
    outcome = await call(tools, "submit_booking", valid_args())
    assert is_error(outcome) and "prepare_booking" in outcome.result


async def test_unknown_tool_is_an_error_listing_the_real_ones(tools):
    outcome = await call(tools, "cancel_booking", {})
    assert is_error(outcome)
    assert "confirm_booking" in outcome.result and "end_call" in outcome.result


# --- Static schemas -------------------------------------------------------------------------------


def test_specs_are_exactly_the_four_tools(business):
    assert [s.name for s in build_tool_specs(business)] == [
        "prepare_booking",
        "confirm_booking",
        "take_message",
        "end_call",
    ]


def test_specs_do_not_depend_on_caller_time_or_call(business):
    a = ToolRegistry(business, InMemorySink(), clock=lambda: NOW, caller_phone="+79990000001")
    b = ToolRegistry(
        business,
        InMemorySink(),
        clock=lambda: datetime(2027, 1, 1, 3, 0, tzinfo=MOSCOW),
        caller_phone="+74950000002",
    )
    assert a.specs == b.specs == build_tool_specs(business)
    assert json.dumps([s.to_api() for s in a.specs]) == json.dumps([s.to_api() for s in b.specs])


async def test_specs_do_not_change_while_a_draft_exists(business, sink):
    tools = ToolRegistry(business, sink, clock=lambda: NOW)
    before = json.dumps([s.to_api() for s in tools.specs])
    await prepare(tools)
    assert json.dumps([s.to_api() for s in tools.specs]) == before


def test_service_enum_matches_config_plus_other(business):
    schema = build_tool_specs(business)[0].parameters["properties"]["service_id"]
    assert schema["enum"] == [s.id for s in business.services] + [OTHER_SERVICE_ID]


def test_period_enum_is_exactly_the_four_values(business):
    schema = build_tool_specs(business)[0].parameters["properties"]["preferred_period"]
    assert schema["enum"] == ["утро", "день", "вечер", "любое"]


def test_required_fields_leave_time_period_and_notes_optional(business):
    required = build_tool_specs(business)[0].parameters["required"]
    assert "preferred_date" in required
    for optional in ("preferred_time", "preferred_period", "notes"):
        assert optional not in required


def test_confirm_booking_takes_no_parameters(business):
    confirm_spec = build_tool_specs(business)[1]
    assert confirm_spec.name == "confirm_booking"
    assert confirm_spec.parameters == {"type": "object", "properties": {}}


async def test_clock_must_be_timezone_aware(business, sink):
    registry = ToolRegistry(business, sink, clock=lambda: datetime(2026, 9, 24, 17, 5))
    with pytest.raises(ValueError, match="timezone-aware"):
        await call(registry, "prepare_booking", valid_args())


# --- With DialogueEngine --------------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream(self, messages, tools=None):
        self.calls.append(list(messages))
        for event in self._scripts.pop(0):
            yield event


def tool_round(call):
    return [ToolCallEvent(call), StreamEnd("tool_calls")]


def prepare_call(call_id="c1", **overrides):
    return ToolCall(call_id, "prepare_booking", json.dumps(valid_args(**overrides)))


async def test_read_back_is_spoken_verbatim_and_the_turn_ends_without_another_llm_call(
    business, sink
):
    llm = ScriptedLLM(tool_round(prepare_call()))
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    events = [e async for e in engine.respond("Запишите меня на полировку")]

    assert [type(e) for e in events] == [ToolResult, Say, Say, Say]
    assert " ".join(e.text for e in events if isinstance(e, Say)) == DEFAULT_READ_BACK
    assert len(llm.calls) == 1  # no second round trip
    assert engine.messages[-1] == Message(Role.ASSISTANT, DEFAULT_READ_BACK)
    assert sink.bookings == []


async def test_full_flow_prepare_yes_confirm_through_the_engine(business, sink):
    llm = ScriptedLLM(
        tool_round(prepare_call()), tool_round(ToolCall("c2", "confirm_booking", "{}"))
    )
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    [e async for e in engine.respond("Запишите меня")]
    assert sink.bookings == []
    events = [e async for e in engine.respond("Да, всё верно")]

    assert len(sink.bookings) == 1 and sink.bookings[0].preferred_time == time(14, 0)
    # The acceptance is spoken by code and the turn ends: no third LLM call.
    assert [type(e) for e in events] == [ToolResult, Say, Say]
    assert events[0].committed is True
    assert " ".join(e.text for e in events if isinstance(e, Say)) == CONFIRMED_SAY
    assert len(llm.calls) == 2
    assert engine.messages[-1] == Message(Role.ASSISTANT, CONFIRMED_SAY)
    # On the "yes" turn the model saw the read-back in its history:
    assert Message(Role.ASSISTANT, DEFAULT_READ_BACK) in llm.calls[1]


async def test_validation_error_goes_back_to_the_model_which_retries(business, sink):
    bad = prepare_call("c1", preferred_date=SUNDAY)
    llm = ScriptedLLM(
        tool_round(bad),
        [TextDelta("В воскресенье мы не работаем. Подойдёт суббота?"), StreamEnd("stop")],
        tool_round(prepare_call("c2", preferred_date=SATURDAY)),
    )
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    first = [e async for e in engine.respond("Запишите на воскресенье")]
    assert first[0].result.startswith(ERROR_PREFIX)
    assert not any(isinstance(e, Say) and "Проверьте" in e.text for e in first)
    assert llm.calls[1][-1].role is Role.TOOL and "не работаем" in llm.calls[1][-1].content

    second = [e async for e in engine.respond("Тогда в субботу")]
    assert any(
        isinstance(e, Say) and "в субботу, двадцать шестого сентября" in e.text for e in second
    )


async def test_model_that_confirms_and_hangs_up_in_one_turn_is_stopped_then_may_end_later(
    business, sink
):
    confirm_call = ToolCall("c2", "confirm_booking", "{}")
    hangup_early = ToolCall("c3", "end_call", "{}")
    hangup_later = ToolCall("c4", "end_call", "{}")
    llm = ScriptedLLM(
        tool_round(prepare_call()),
        # turn 2: the caller says «да»; the model confirms and tries to hang up in the same round
        [ToolCallEvent(confirm_call), ToolCallEvent(hangup_early), StreamEnd("tool_calls")],
        # turn 3: «нет, спасибо»
        [TextDelta("Всего доброго!"), ToolCallEvent(hangup_later), StreamEnd("tool_calls")],
    )
    engine = DialogueEngine(llm, ToolRegistry(business, sink, clock=lambda: NOW), "SYS")

    await collect_events(engine, "Запишите меня")
    second = await collect_events(engine, "Да, всё верно")
    third = await collect_events(engine, "Нет, спасибо")

    assert len(sink.bookings) == 1
    assert not any(isinstance(e, EndCall) for e in second)  # the call was NOT ended
    early_result = [e for e in second if isinstance(e, ToolResult) and e.call.name == "end_call"]
    assert early_result[0].result.startswith(ERROR_PREFIX)
    # The caller still hears the acceptance and the question, from code, in that same turn:
    assert Say("Могу ещё чем-то помочь?") in second
    assert len(llm.calls) == 3  # no LLM round was needed to tell the caller
    assert isinstance(third[-1], EndCall)


async def collect_events(engine, user_text):
    return [event async for event in engine.respond(user_text)]
