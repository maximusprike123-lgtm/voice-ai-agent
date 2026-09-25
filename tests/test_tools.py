"""Offline tests for the tools: validation, records, static schemas, engine round trip."""

import json
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import OTHER_SERVICE_ID, load_business_config
from agent.dialogue import DialogueEngine, Say, ToolResult
from agent.llm import Role, StreamEnd, TextDelta, ToolCall, ToolCallEvent
from agent.records import Booking, InMemorySink
from agent.tools import (
    ERROR_PREFIX,
    MAX_HORIZON_DAYS,
    ToolRegistry,
    build_tool_specs,
    normalize_phone,
)

REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"
MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 24, 17, 5, tzinfo=MOSCOW)  # Thursday; Mon-Fri 10-21, Sat 10-20, Sun off
FRIDAY, SATURDAY, SUNDAY = "2026-09-25", "2026-09-26", "2026-09-27"


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


async def book(tools, **overrides):
    return await call(tools, "submit_booking", valid_args(**overrides))


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


# --- submit_booking: happy paths --------------------------------------------------------------


async def test_valid_booking_is_saved_with_normalized_fields(tools, sink):
    outcome = await book(tools)

    assert not is_error(outcome) and not outcome.ends_call
    assert sink.bookings == [
        Booking(
            name="Игорь",
            phone="+79161234567",
            car="Toyota Camry",
            service_id="polishing",
            service_name="Полировка кузова",
            preferred_date=date(2026, 9, 25),
            preferred_time=time(14, 0),
            notes=None,
            caller_phone="+79991234567",
            created_at=NOW,
        )
    ]


async def test_success_result_never_confirms_the_slot_or_names_a_price(tools):
    result = (await book(tools)).result
    assert "перезвонит для подтверждения" in result
    assert "Не говори, что время подтверждено" in result
    assert "₽" not in result and "руб" not in result


async def test_booking_without_time_is_valid_and_keeps_the_callers_words(tools, sink):
    outcome = await book(tools, preferred_date=SATURDAY, preferred_time=None, notes="после обеда")

    assert not is_error(outcome)
    assert sink.bookings[0].preferred_time is None
    assert sink.bookings[0].notes == "после обеда"
    assert "Точное время не указано" in outcome.result


@pytest.mark.parametrize("empty", ["", "   ", None])
async def test_blank_time_counts_as_no_time(tools, sink, empty):
    args = valid_args(preferred_time=None)
    if empty is not None:
        args["preferred_time"] = empty
    outcome = await call(tools, "submit_booking", args)
    assert not is_error(outcome)
    assert sink.bookings[0].preferred_time is None


async def test_time_boundaries_open_inclusive_close_exclusive(tools, sink):
    assert not is_error(await book(tools, preferred_time="10:00"))
    assert not is_error(await book(tools, preferred_time="20:59"))
    assert is_error(await book(tools, preferred_time="21:00"))
    assert is_error(await book(tools, preferred_time="09:59"))


async def test_saturday_closes_earlier(tools):
    assert not is_error(await book(tools, preferred_date=SATURDAY, preferred_time="19:30"))
    outcome = await book(tools, preferred_date=SATURDAY, preferred_time="20:00")
    assert is_error(outcome) and "10:00–20:00" in outcome.result


async def test_single_digit_hour_and_numeric_phone_are_accepted(tools, sink):
    outcome = await book(tools, preferred_time="9:30", phone=89161234567)
    assert is_error(outcome)  # 09:30 is before opening...
    assert "phone" not in outcome.result  # ...but the numeric phone itself was fine
    assert not is_error(await book(tools, preferred_time="11:05", phone=89161234567))
    assert sink.bookings[0].preferred_time == time(11, 5)
    assert sink.bookings[0].phone == "+79161234567"


# --- submit_booking: validation errors --------------------------------------------------------


async def test_past_date_is_rejected(tools, sink):
    outcome = await book(tools, preferred_date="2026-09-23")
    assert is_error(outcome) and "прошла" in outcome.result
    assert sink.bookings == []


async def test_today_is_allowed_but_a_time_already_passed_is_not(tools):
    assert not is_error(await book(tools, preferred_date="2026-09-24", preferred_time="18:00"))
    outcome = await book(tools, preferred_date="2026-09-24", preferred_time="12:00")
    assert is_error(outcome) and "уже прошло" in outcome.result
    assert is_error(await book(tools, preferred_date="2026-09-24", preferred_time="17:05"))


async def test_today_without_time_is_allowed(tools):
    assert not is_error(await book(tools, preferred_date="2026-09-24", preferred_time=None))


async def test_closed_day_is_rejected(tools):
    outcome = await book(tools, preferred_date=SUNDAY, preferred_time=None)
    assert is_error(outcome)
    assert "воскресенье" in outcome.result and "не работаем" in outcome.result


async def test_horizon_limit(tools):
    # NOW + 90 days is Wednesday 2026-12-23; the day after is the first one over the limit.
    assert not is_error(await book(tools, preferred_date="2026-12-23", preferred_time=None))
    outcome = await book(tools, preferred_date="2026-12-24", preferred_time=None)
    assert is_error(outcome) and str(MAX_HORIZON_DAYS) in outcome.result


async def test_year_typo_is_caught_by_horizon(tools):
    assert is_error(await book(tools, preferred_date="2027-09-25"))


@pytest.mark.parametrize("bad", ["25.09.2026", "завтра", "2026-9-25", "2026-02-30", "20260925"])
async def test_unparsable_date_is_rejected(tools, bad):
    outcome = await book(tools, preferred_date=bad)
    assert is_error(outcome) and "YYYY-MM-DD" in outcome.result


@pytest.mark.parametrize("bad", ["после обеда", "25:00", "14:60", "14", "два часа"])
async def test_unparsable_time_is_rejected_with_a_hint_about_notes(tools, bad):
    outcome = await book(tools, preferred_time=bad)
    assert is_error(outcome) and "notes" in outcome.result


async def test_unknown_service_is_rejected_and_valid_ids_are_listed(tools):
    outcome = await book(tools, service_id="rocket_wash")
    assert is_error(outcome)
    assert "rocket_wash" in outcome.result and "polishing" in outcome.result
    assert OTHER_SERVICE_ID in outcome.result


async def test_bad_phone_is_rejected(tools, sink):
    outcome = await book(tools, phone="12345")
    assert is_error(outcome) and "phone" in outcome.result
    assert sink.bookings == []


@pytest.mark.parametrize("field", ["name", "phone", "car", "service_id", "preferred_date"])
async def test_missing_required_field_is_rejected(tools, field):
    outcome = await book(tools, **{field: None})
    assert is_error(outcome) and field in outcome.result


async def test_all_problems_are_reported_together(tools, sink):
    outcome = await book(tools, name=None, phone="1", preferred_date=SUNDAY, service_id="nope")
    for field in ("name", "phone", "service_id", "preferred_date"):
        assert field in outcome.result
    assert sink.bookings == []


async def test_overlong_name_is_rejected(tools):
    assert is_error(await book(tools, name="И" * 101))


async def test_non_string_field_is_rejected(tools):
    outcome = await book(tools, car={"make": "Toyota"})
    assert is_error(outcome) and "car" in outcome.result


# --- "other" service ------------------------------------------------------------------------------


async def test_other_service_records_the_description_and_invents_no_price(tools, sink):
    outcome = await book(
        tools, service_id="other", notes="нужно убрать запах после перевозки собаки"
    )

    assert not is_error(outcome)
    booking = sink.bookings[0]
    assert booking.service_id == OTHER_SERVICE_ID
    assert booking.notes == "нужно убрать запах после перевозки собаки"
    assert "₽" not in outcome.result and "руб" not in outcome.result


async def test_other_service_requires_notes(tools, sink):
    outcome = await book(tools, service_id="other")
    assert is_error(outcome) and "notes" in outcome.result
    assert sink.bookings == []


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


async def test_sink_failure_is_reported_to_the_model_and_not_marked_as_submitted(business):
    tools = ToolRegistry(business, BrokenSink(), clock=lambda: NOW)
    first = await book(tools)
    assert is_error(first) and "не удалось" in first.result

    working = InMemorySink()
    tools._sink = working
    assert not is_error(await book(tools))  # a retry is not treated as a duplicate
    assert len(working.bookings) == 1


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


# --- Malformed calls ------------------------------------------------------------------------------


async def test_invalid_json_is_an_error_result_not_an_exception(tools):
    outcome = await call(tools, "submit_booking", raw='{"name": "Игорь"')
    assert is_error(outcome) and "JSON" in outcome.result


async def test_non_object_arguments_are_rejected(tools):
    assert is_error(await call(tools, "submit_booking", raw='["Игорь"]'))


async def test_unknown_tool_is_an_error_listing_the_real_ones(tools):
    outcome = await call(tools, "cancel_booking", {})
    assert is_error(outcome)
    assert "submit_booking" in outcome.result and "end_call" in outcome.result


# --- Static schemas -------------------------------------------------------------------------------


def test_specs_are_exactly_the_three_tools(business):
    assert [s.name for s in build_tool_specs(business)] == [
        "submit_booking",
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


def test_service_enum_matches_config_plus_other(business):
    schema = build_tool_specs(business)[0].parameters["properties"]["service_id"]
    assert schema["enum"] == [s.id for s in business.services] + [OTHER_SERVICE_ID]


def test_required_fields_leave_time_and_notes_optional(business):
    required = build_tool_specs(business)[0].parameters["required"]
    assert "preferred_date" in required
    assert "preferred_time" not in required and "notes" not in required


async def test_clock_must_be_timezone_aware(business, sink):
    registry = ToolRegistry(business, sink, clock=lambda: datetime(2026, 9, 24, 17, 5))
    with pytest.raises(ValueError, match="timezone-aware"):
        await call(registry, "submit_booking", valid_args())


# --- With DialogueEngine --------------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream(self, messages, tools=None):
        self.calls.append(list(messages))
        for event in self._scripts.pop(0):
            yield event


async def test_validation_error_goes_back_to_the_model_which_retries(business, sink):
    bad = ToolCall("c1", "submit_booking", json.dumps(valid_args(preferred_date=SUNDAY)))
    good = ToolCall("c2", "submit_booking", json.dumps(valid_args(preferred_date=SATURDAY)))
    llm = ScriptedLLM(
        [ToolCallEvent(bad), StreamEnd("tool_calls")],
        [TextDelta("В воскресенье мы не работаем. Подойдёт суббота?"), StreamEnd("stop")],
        [ToolCallEvent(good), StreamEnd("tool_calls")],
        [TextDelta("Заявка передана администратору."), StreamEnd("stop")],
    )
    registry = ToolRegistry(business, sink, clock=lambda: NOW)
    engine = DialogueEngine(llm, registry, "SYS")

    first = [e async for e in engine.respond("Запишите на воскресенье")]
    assert first[0].result.startswith(ERROR_PREFIX)
    assert sink.bookings == []
    assert llm.calls[1][-1].role is Role.TOOL and "не работаем" in llm.calls[1][-1].content

    second = [e async for e in engine.respond("Тогда в субботу")]
    assert [type(e) for e in second] == [ToolResult, Say]
    assert len(sink.bookings) == 1 and sink.bookings[0].preferred_date == date(2026, 9, 26)
