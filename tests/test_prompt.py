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
    for tool in ("prepare_booking", "confirm_booking", "take_message", "end_call"):
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
    for tool in ("prepare_booking", "confirm_booking", "take_message", "end_call"):
        assert tool in static
    # The static part may QUOTE the label «Номер звонящего: …» in its instructions; it must not
    # contain the actual number, the clock or the calendar.
    for per_call in ("Сейчас", "Календарь", "+79161234567", "2026-09-24"):
        assert per_call not in static
    for per_call in ("17:05", "+79161234567", "2026-09-24", "Календарь"):
        assert per_call in volatile


def test_static_part_explains_other_service_and_approximate_time(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "service_id other" in static
    assert "не придумывай точное время" in static
    assert "preferred_time" in static and "preferred_period" in static and "notes" in static
    assert "ОШИБКА" in static


def test_static_part_describes_the_two_step_booking(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "submit_booking" not in build_system_prompt(business, NOW)
    assert "prepare_booking" in static and "confirm_booking" in static
    assert "не пересказывай" in static  # the read-back is spoken by the system
    assert "ясно сказал «да»" in static


def test_model_is_told_never_to_say_phone_digits(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "никогда не произноси цифры номера, ни целиком, ни частями" in static
    assert "последние четыре цифры" not in static


def test_the_prompt_no_longer_asks_for_phone_numbers_to_be_written_in_words(business):
    """It used to say «Пиши числа, даты, время, цены и номера телефонов словами»: an instruction
    to spell out phone numbers, contradicting the never-say-digits rule."""
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "цены и номера телефонов словами" not in static
    assert "номера телефонов не произноси вообще" in static


def test_hidden_caller_id_is_explicit_the_number_is_unknown(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "«Номер звонящего: не определён»" in static
    assert "номер клиента тебе НЕИЗВЕСТЕН" in static
    assert "не говори, что номер определился" in static
    assert "никогда не предлагай «номер, с которого вы звоните»" in static
    assert "сразу попроси клиента продиктовать номер" in static
    hidden = build_system_prompt(business, NOW, None)
    assert "Номер звонящего: не определён." in hidden.split(VOLATILE_MARKER)[1]


def test_the_dictated_number_wins_over_the_caller_id(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "ДРУГОЙ номер" in static
    assert "именно названный им номер, а не номер звонящего" in static


def test_no_own_recap_before_prepare_booking(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "СРАЗУ вызови prepare_booking, ничего не говоря перед этим" in static
    assert "не пересказывай данные заявки" in static
    assert "не спрашивай «всё верно?»" in static
    assert "не начинай фразу со слов «Уточню»" in static


def test_a_caller_who_wants_to_book_an_unlisted_service_gets_an_other_booking(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "ясно хочет ЗАПИСАТЬСЯ на услугу, которой нет в списке" in static
    assert "не предлагай «просто передать вопрос»" in static
    assert "оформи запись с service_id other" in static
    assert "take_message вызывай, только когда клиент ничего не заказывает" in static


def test_static_part_forbids_saying_recorded_before_confirmation(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "Пока confirm_booking не вернул успех" in static
    for word in ("«записал»", "«записала»", "«записано»"):
        assert word in static
    assert "«хорошо»" in static and "«принято»" in static


def test_static_part_says_the_system_announces_results_and_to_wait_before_ending_the_call(business):
    static = build_system_prompt(business, NOW).split(VOLATILE_MARKER)[0]
    assert "система сама сообщит клиенту, что заявка принята" in static
    assert "спросит, нужна ли помощь ещё: ничего не добавляй и дождись ответа клиента" in static
    assert "система сама сообщит клиенту, что сообщение передано" in static  # take_message
    assert "Не завершай звонок в том же ответе, в котором заявка или сообщение приняты" in static
    assert "Если end_call вернул ОШИБКА" in static


def test_naive_datetime_is_rejected(business):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_system_prompt(business, datetime(2026, 9, 24, 17, 5))


def test_the_prompt_tells_the_model_its_gender(business):
    male = build_system_prompt(business, NOW)  # male is the default
    female = build_system_prompt(business, NOW, gender="female")

    assert "О себе говори в мужском роде" in male and "женском роде" not in male
    assert "О себе говори в женском роде" in female and "мужском роде" not in female
    assert build_system_prompt(business, NOW, gender="male") == male


def test_the_gender_line_is_in_the_static_part_and_keeps_the_prefix_stable(business):
    male = [build_system_prompt(business, NOW, phone) for phone in (None, "+79161234567")]
    female = build_system_prompt(business, NOW, gender="female")

    static = male[0].split(VOLATILE_MARKER)[0]
    assert "О себе говори в мужском роде" in static
    assert len(os.path.commonprefix(male)) >= len(static)  # the caller does not break the cache
    assert female.split(VOLATILE_MARKER)[0] != static  # one gender per deployment, so it may differ


def test_an_unknown_gender_is_rejected(business):
    with pytest.raises(ValueError, match="gender"):
        build_system_prompt(business, NOW, gender="other")
