import os.path
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import load_business_config
from agent.prompt import VOLATILE_MARKER, build_system_prompt, format_date_ru, format_price

REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"
MOSCOW = ZoneInfo("Europe/Moscow")
# Thursday
NOW = datetime(2026, 9, 24, 17, 5, tzinfo=MOSCOW)


@pytest.fixture(scope="module")
def business():
    return load_business_config(REPO_CONFIG)


def test_format_date_ru():
    assert format_date_ru(date(2026, 9, 24)) == "четверг, 24 сентября 2026"
    assert format_date_ru(date(2027, 1, 3)) == "воскресенье, 3 января 2027"


def test_format_price(business):
    wash = business.service_by_id("detailing_wash")
    ceramic = business.service_by_id("ceramic_coating")
    assert format_price(wash) == "от 3 000 до 5 000 ₽"
    assert format_price(ceramic) == "от 25 000 ₽"


def test_prompt_has_current_date_and_time(business):
    prompt = build_system_prompt(business, NOW)
    assert "Сейчас четверг, 24 сентября 2026, 17:05" in prompt


def test_prompt_calendar_resolves_relative_days(business):
    prompt = build_system_prompt(business, NOW)
    assert "2026-09-24 — четверг, 24 сентября 2026 (сегодня): 10:00–21:00" in prompt
    assert "2026-09-25 — пятница, 25 сентября 2026 (завтра): 10:00–21:00" in prompt
    assert "2026-09-27 — воскресенье, 27 сентября 2026: выходной" in prompt
    assert "2026-10-07" in prompt
    assert "2026-10-08" not in prompt


def test_prompt_has_business_data(business):
    prompt = build_system_prompt(business, NOW)
    assert business.name in prompt
    assert business.address in prompt
    for service in business.services:
        assert f"[{service.id}] {service.name}" in prompt
    for item in business.faq:
        assert item.answer in prompt


def test_prompt_caller_phone(business):
    assert "Номер звонящего: +79161234567." in build_system_prompt(business, NOW, "+79161234567")
    assert "Номер звонящего: не определён." in build_system_prompt(business, NOW)


def test_prompt_names_the_tools(business):
    prompt = build_system_prompt(business, NOW)
    for tool in ("submit_booking", "take_message", "end_call"):
        assert tool in prompt


def test_prompt_includes_extra_rules(business):
    custom = business.model_copy(update={"extra_rules": ["Не обсуждай конкурентов."]})
    assert "- Не обсуждай конкурентов." in build_system_prompt(custom, NOW)


def test_static_prefix_is_identical_across_times_and_callers(business):
    """The prompt prefix must not depend on the call, so a prompt cache can reuse it."""
    prompts = [
        build_system_prompt(business, NOW, "+79161234567"),
        build_system_prompt(business, NOW),
        build_system_prompt(business, datetime(2026, 10, 3, 9, 41, tzinfo=MOSCOW), "+79990001122"),
        build_system_prompt(business, datetime(2027, 1, 1, 0, 0, tzinfo=MOSCOW), "+74951112233"),
    ]
    assert all(p.count(VOLATILE_MARKER) == 1 for p in prompts)

    static_parts = [p.split(VOLATILE_MARKER)[0] for p in prompts]
    assert len(set(static_parts)) == 1
    assert len(set(prompts)) == len(prompts)  # ...while the full prompts do differ

    # The property that matters: the byte-identical shared prefix covers the whole static part.
    static = static_parts[0]
    assert len(os.path.commonprefix(prompts)) >= len(static)


def test_static_part_holds_all_business_knowledge_and_nothing_per_call(business):
    prompt = build_system_prompt(business, NOW, "+79161234567")
    static, volatile = prompt.split(VOLATILE_MARKER)

    assert static.startswith("Ты — голосовой администратор")
    for expected in (business.name, business.address, "# Услуги и цены", "# Частые вопросы"):
        assert expected in static
    for tool in ("submit_booking", "take_message", "end_call"):
        assert tool in static
    for per_call in ("Сейчас", "Номер звонящего", "Календарь", "+79161234567", "2026-09-24"):
        assert per_call not in static
    for per_call in ("17:05", "+79161234567", "2026-09-24", "Календарь"):
        assert per_call in volatile


def test_static_part_explains_other_service_and_approximate_time(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "service_id other" in static
    assert "не придумывай точное время" in static
    assert "preferred_time" in static and "notes" in static
    assert "ОШИБКА" in static


def test_naive_datetime_is_rejected(business):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_system_prompt(business, datetime(2026, 9, 24, 17, 5))
