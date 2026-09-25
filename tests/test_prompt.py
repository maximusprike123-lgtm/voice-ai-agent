from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import load_business_config
from agent.prompt import build_system_prompt, format_date_ru, format_price

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


def test_naive_datetime_is_rejected(business):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_system_prompt(business, datetime(2026, 9, 24, 17, 5))
