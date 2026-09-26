"""Offline tests for the text stage before TTS: normalization, transliteration, the guard."""

import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import Weekday, load_business_config
from agent.dialogue import split_sentences
from agent.llm import ToolCall
from agent.records import InMemorySink
from agent.ru_words import cardinal_genitive, genitive_noun
from agent.session import ASK_TO_REPEAT, COMMITTED_FALLBACK, FINAL_APOLOGY
from agent.speech.text import (
    LATIN_WORDS,
    SpeechText,
    normalize_for_speech,
    transliterate,
)
from agent.text_guard import GUARD_FALLBACKS, SpeechGuard
from agent.tools import (
    ANYTHING_ELSE_SAY,
    BOOKING_ACCEPTED_SAY,
    BOOKING_ALREADY_ACCEPTED_SAY,
    EXACT_TIME_NOTE_SAY,
    MESSAGE_TAKEN_SAY,
    ToolRegistry,
)
from tests.test_text_guard import REAL_LEGIT, REAL_VIOLATIONS

REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"
MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 24, 17, 5, tzinfo=MOSCOW)


def spoken(text):
    return normalize_for_speech(text)[0]


# --- Genitive numerals (ru_words) -------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "ноля"),
        (1, "одного"),
        (2, "двух"),
        (10, "десяти"),
        (21, "двадцати одного"),
        (100, "ста"),
        (200, "двухсот"),
        (1000, "одной тысячи"),
        (2000, "двух тысяч"),
        (3500, "трёх тысяч пятисот"),
        (18000, "восемнадцати тысяч"),
        (25000, "двадцати пяти тысяч"),
        (999999, "девятисот девяноста девяти тысяч девятисот девяноста девяти"),
    ],
)
def test_cardinal_genitive(n, words):
    assert cardinal_genitive(n) == words


def test_genitive_noun_agrees_with_the_last_digits():
    assert genitive_noun(1, "рубля", "рублей") == "рубля"
    assert genitive_noun(21, "рубля", "рублей") == "рубля"
    assert genitive_noun(11, "рубля", "рублей") == "рублей"
    assert genitive_noun(5, "рубля", "рублей") == "рублей"


# --- Numbers, times, dates, money -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("в 14:30", "в четырнадцать тридцать"),
        ("в 10:00", "в десять часов"),
        ("в 9:05", "в девять ноль пять"),
        ("с 10:00 до 21:00", "с десяти часов до двадцати одного часа"),
        ("после 18:30", "после восемнадцати тридцати"),
        ("26.09", "двадцать шестого сентября"),
        ("1 октября", "один октября"),
        ("01.10.2026", "первого октября"),
        ("цена 25 000 ₽", "цена двадцать пять тысяч рублей"),
        ("от 10 000 до 18 000 ₽", "от десяти тысяч до восемнадцати тысяч рублей"),
        ("от 3 000 до 5 000 руб.", "от трёх тысяч до пяти тысяч рублей"),
        ("от 3 до 5 тыс. рублей", "от трёх до пяти тысяч рублей"),
        ("1 ₽", "один рубль"),
        ("21 ₽", "двадцать один рубль"),
        ("до 21 ₽", "до двадцати одного рубля"),
        ("скидка 10%", "скидка десять процентов"),
        ("от 5 до 10 %", "от пяти до десяти процентов"),
        ("2 часа", "два часа"),
        ("5 дней", "пять дней"),
        ("3,5 часа", "три запятая пять часа"),
        ("№5", "номер пять"),
        ("5мм", "пять мм"),
        ("1000000 рублей", "один ноль ноль ноль ноль ноль ноль рублей"),
    ],
)
def test_numbers_become_words(text, expected):
    assert spoken(text) == expected


def test_a_price_sentence_from_a_real_call_is_normalized():
    real = "Химчистка салона обойдётся от 10 000 до 18 000 ₽."
    assert (
        spoken(real) == "Химчистка салона обойдётся от десяти тысяч до восемнадцати тысяч рублей."
    )


def test_a_phone_number_becomes_a_run_of_number_words_the_guard_then_catches():
    text = spoken("Номер — 8 916 123 45 67.")
    assert not any(ch.isdigit() for ch in text)
    assert SpeechGuard().check(text).rule == "phone_digits"


def test_a_time_or_date_out_of_range_is_left_alone_by_its_own_rule():
    assert spoken("32.13") != "тридцать второго"  # not a date: no such day and month
    assert "двадцать пятого" not in spoken("в 25:99")  # not a time


def test_changes_are_reported_by_kind():
    assert normalize_for_speech("в 14:30, цена 5 000 ₽, Toyota")[1] == ["time", "money", "latin"]
    assert normalize_for_speech("Всё готово.")[1] == []


def test_text_that_needs_no_work_is_returned_unchanged():
    text = "Полировка — от пятнадцати до тридцати пяти тысяч рублей, «под ключ»."
    assert normalize_for_speech(text) == (text, [])


# --- Markup -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Хорошо**, ждём.", "Хорошо, ждём."),
        ("Хорошо 😀 ждём", "Хорошо ждём"),
        ("# Заголовок", "Заголовок"),
        ("`код` и ~зачёркнутое~", "код и зачёркнутое"),
        ("Здравствуйте\u200b!", "Здравствуйте!"),
        ("[ссылка](адрес)", "ссылка (адрес)"),
    ],
)
def test_markup_and_emoji_are_removed(text, expected):
    result, changes = normalize_for_speech(text)
    assert result == expected and "markup" in changes


# --- Latin words ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Toyota Camry", "Тойота Камри"),
        ("автомобиль BMW X5", "автомобиль бэ эм вэ икс пять"),
        ("Mercedes-Benz", "Мерседес Бенц"),
        ("Land Rover", "Лэнд Ровер"),
        ("RAV4", "РАВ четыре"),
        ("оклейка PPF", "оклейка пэ пэ эф"),
        ("kia rio", "Киа Рио"),
        ("Skoda Octavia 2015", "Шкода Октавия две тысячи пятнадцать"),
    ],
)
def test_latin_words_are_read_the_russian_way(text, expected):
    assert spoken(text) == expected


def test_an_unknown_latin_word_is_transliterated_and_reported(caplog):
    with caplog.at_level(logging.WARNING):
        text, changes = normalize_for_speech("Quattro Fjord")
    assert text == "Куаттро Фджорд" or "Квaттро" not in text
    assert not any("a" <= ch.lower() <= "z" for ch in text)
    assert "latin" in changes and "Quattro" in caplog.text


@pytest.mark.parametrize(
    ("word", "expected"),
    [("shchuka", "щука"), ("chery", "чери"), ("cinema", "синема"), ("Zhiguli", "Жигули")],
)
def test_transliteration_handles_the_common_digraphs(word, expected):
    assert transliterate(word) == expected


def test_the_word_table_has_only_lower_case_keys_and_no_latin_in_the_readings():
    for key, reading in LATIN_WORDS.items():
        assert key == key.lower()
        assert not any("a" <= ch.lower() <= "z" for ch in reading), key


# --- The stage: normalization, then the guard -------------------------------------------------


def test_a_clean_model_sentence_passes_untouched():
    result = SpeechText(SpeechGuard()).prepare("Как вас зовут?")
    assert (result.text, result.changes, result.violation) == ("Как вас зовут?", (), None)


def test_numbers_in_a_model_sentence_are_spoken_as_words_and_logged(caplog):
    stage = SpeechText(SpeechGuard())
    with caplog.at_level(logging.WARNING):
        result = stage.prepare("Полировка стоит от 15 000 до 35 000 ₽.")
    assert result.text == "Полировка стоит от пятнадцати тысяч до тридцати пяти тысяч рублей."
    assert result.changes == ("money", "number") and result.violation is None
    assert "normalized model sentence" in caplog.text


def test_a_phone_number_in_digits_is_dropped_after_normalization():
    guard = SpeechGuard()
    result = SpeechText(guard).prepare("Номер — 8 916 123 45 67.")
    assert result.text is None and result.violation.rule == "phone_digits"
    assert [v.rule for v in guard.blocked] == ["phone_digits"]


@pytest.mark.parametrize(("sentence", "rule"), REAL_VIOLATIONS)
def test_every_real_violation_is_still_dropped_at_this_stage(sentence, rule):
    result = SpeechText(SpeechGuard()).prepare(sentence)
    assert result.text is None and result.violation.rule == rule


@pytest.mark.parametrize("sentence", REAL_LEGIT)
def test_every_real_legitimate_sentence_is_spoken(sentence):
    result = SpeechText(SpeechGuard()).prepare(sentence)
    assert result.text is not None and result.violation is None
    if not any(ch.isdigit() for ch in sentence):
        assert result.text == sentence and result.changes == ()


def test_a_sentence_that_is_only_markup_is_dropped_without_a_violation():
    result = SpeechText(SpeechGuard()).prepare("**")
    assert result.text is None and result.violation is None


def test_without_a_guard_only_normalization_runs():
    result = SpeechText(None).prepare("Хорошо, записал. Номер 8 916 123 45 67.")
    assert result.text is not None and result.violation is None


def test_the_stage_uses_the_call_guards_knowledge_of_saves():
    guard = SpeechGuard()
    stage = SpeechText(guard)
    assert stage.prepare("Заявка принята.").text is None  # nothing saved yet
    guard.note_commit()
    assert stage.prepare("Заявка принята.").text == "Заявка принята."


# --- Text written by code: spoken with a log line, except two rules ---------------------------


def test_a_code_sentence_that_trips_a_soft_rule_is_still_spoken_and_logged(caplog):
    stage = SpeechText(SpeechGuard())
    with caplog.at_level(logging.ERROR):
        result = stage.prepare("Хорошо, записал.", scripted=True)
    assert result.text == "Хорошо, записал." and result.violation.rule == "written_down"
    assert "spoken anyway" in caplog.text
    assert [v.rule for v in stage.code_violations] == ["written_down"]


@pytest.mark.parametrize(
    ("sentence", "rule"),
    [
        ("Номер 8 916 123 45 67.", "phone_digits"),
        ("Оформлю заявку. 五千 рублей.", "foreign_script"),
    ],
)
def test_a_code_sentence_with_a_phone_number_or_foreign_script_is_dropped(sentence, rule, caplog):
    stage = SpeechText(SpeechGuard())
    with caplog.at_level(logging.ERROR):
        result = stage.prepare(sentence, scripted=True)
    assert result.text is None and result.violation.rule == rule
    assert "dropped" in caplog.text


def test_latin_car_names_in_a_code_read_back_are_only_an_info_line(caplog):
    with caplog.at_level(logging.INFO):
        result = SpeechText(SpeechGuard()).prepare("автомобиль Toyota Camry", scripted=True)
    assert result.text == "автомобиль Тойота Камри"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- Every sentence the code can build passes the guard ---------------------------------------

# Spoken while nothing is saved (a failed turn, a guard fallback): must pass either way.
PHRASES_BEFORE_A_SAVE = [ASK_TO_REPEAT, FINAL_APOLOGY, *GUARD_FALLBACKS.values()]
# Only ever spoken after a tool saved something (the engine notes the commit first).
PHRASES_AFTER_A_SAVE = [
    ANYTHING_ELSE_SAY,
    BOOKING_ACCEPTED_SAY,
    BOOKING_ALREADY_ACCEPTED_SAY,
    EXACT_TIME_NOTE_SAY,
    MESSAGE_TAKEN_SAY,
    COMMITTED_FALLBACK,
]


@pytest.mark.parametrize("sentence", PHRASES_BEFORE_A_SAVE)
@pytest.mark.parametrize("saved", [False, True])
def test_the_fixed_phrases_pass_the_guard_before_and_after_a_save(sentence, saved):
    guard = SpeechGuard()
    if saved:
        guard.note_commit()
    result = SpeechText(guard).prepare(sentence, scripted=True)
    assert result.violation is None and result.text == sentence


@pytest.mark.parametrize("sentence", PHRASES_AFTER_A_SAVE)
def test_the_phrases_spoken_after_a_save_pass_the_guard_once_it_knows(sentence):
    guard = SpeechGuard()
    guard.note_commit()
    result = SpeechText(guard).prepare(sentence, scripted=True)
    assert result.violation is None and result.text == sentence


def test_the_acceptance_phrases_do_trip_the_guard_when_nothing_was_saved():
    """Why the engine notes a commit before it speaks them: this is the guard doing its job."""
    first_sentence_of_the_fallback = split_sentences(COMMITTED_FALLBACK)[0]
    for sentence in (BOOKING_ACCEPTED_SAY, MESSAGE_TAKEN_SAY, first_sentence_of_the_fallback):
        assert SpeechGuard().check(sentence).rule == "acceptance_claim", sentence


def test_the_greeting_passes_the_guard():
    greeting = load_business_config(REPO_CONFIG).greeting
    for sentence in split_sentences(greeting):
        assert SpeechText(SpeechGuard()).prepare(sentence, scripted=True).violation is None


PHONES = [
    "+79161234567",
    "89990001122",
    "9161234500",
    "8 (916) 000-00-01",
    "+7 903 555 12 34",
    "+79995550000",
    "+79271112233",
    "+74951234567",
]
CARS = [
    "Toyota Camry",
    "BMW X5",
    "Тойота Камри",
    "Kia Rio",
    "Mercedes-Benz E200",
    "Land Rover Discovery",
    "Lada 2107",
    "Skoda Octavia 2015",
    "Лада Веста",
]


def open_days(business):
    day, found = NOW.date() + timedelta(days=1), []
    while len(found) < 30:
        if business.hours[Weekday.from_index(day.weekday())] is not None:
            found.append(day)
        day += timedelta(days=1)
    return found


async def test_every_read_back_the_code_can_build_passes_the_guard():
    business = load_business_config(REPO_CONFIG)
    days = open_days(business)
    services = [s.id for s in business.services] + ["other"]
    times = [time(h, m) for h in range(10, 20) for m in (0, 30)]
    checked = 0
    for p, phone in enumerate(PHONES):
        tools = ToolRegistry(business, InMemorySink(), clock=lambda: NOW)
        tools.begin_turn(f"Меня зовут Игорь, номер {phone}")
        for i, car in enumerate(CARS):
            args = {
                "name": "Игорь",
                "phone": phone,
                "car": car,
                "service_id": services[(p + i) % len(services)],
                "preferred_date": days[(p * 3 + i) % len(days)].isoformat(),
                "notes": "нужно 3 фары, PPF на бампер, до 15 000 ₽ или 10%",
            }
            if (p + i) % 2:
                args["preferred_time"] = times[(p * 5 + i * 7) % len(times)].strftime("%H:%M")
            else:
                args["preferred_period"] = ["утро", "день", "вечер", "любое"][(p + i) % 4]
            outcome = await tools.execute(ToolCall("c", "prepare_booking", json.dumps(args)))
            assert outcome.say, outcome.result
            for sentence in split_sentences(outcome.say):
                result = SpeechText(SpeechGuard()).prepare(sentence, scripted=True)
                assert result.violation is None, (sentence, result.violation)
                assert result.text and not any(ch.isdigit() for ch in result.text), sentence
                assert set(result.changes) <= {"latin", "number", "money", "percent"}, sentence
            checked += 1
    assert checked == len(PHONES) * len(CARS)
