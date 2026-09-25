import pytest

from agent.text_guard import foreign_script_chars


@pytest.mark.parametrize(
    "text",
    [
        "Полировка кузова стоит от пятнадцати до тридцати пяти тысяч рублей.",
        "Мы находимся в Москве, улица Примерная, дом один. Работаем с десяти до двадцати.",
        "Toyota Camry, BMW X5, PPF — оклейка «плёнкой» №1, 15 000 ₽ (примерно)!",
        "Ёлка, ёж; Ё.",
        "",
        "   \n\t",
        "2026-09-26, 14:30 - 20:00 / 100%",
    ],
)
def test_normal_russian_speech_has_no_foreign_characters(text):
    assert foreign_script_chars(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("от пятнадцати до тридцати五千 рублей", ["五", "千"]),
        ("Цена 三万 рублей", ["三", "万"]),
        ("Здравствуйте、чем помочь", ["、"]),  # CJK punctuation
        ("Мы работаем ежедневно，без выходных", ["，"]),  # full-width comma
        ("Привет مرحبا", list("مرحبا")),
        ("Привет こんにちは", list("こんにちは")),
        ("привет 안녕", list("안녕")),
        ("Сколько стоит α-полировка", ["α"]),
    ],
)
def test_characters_from_other_scripts_are_found_in_order(text, expected):
    assert foreign_script_chars(text) == expected


def test_repeats_are_reported_each_time():
    assert foreign_script_chars("五 и 五") == ["五", "五"]


def test_latin_and_cyrillic_mixed_are_fine():
    assert foreign_script_chars("Скажите model и марку: Kia Rio, Лада Веста") == []
