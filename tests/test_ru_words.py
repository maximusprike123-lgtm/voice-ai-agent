from datetime import date, time

import pytest

from agent.ru_words import (
    cardinal,
    date_on_phrase,
    date_words,
    day_ordinal_genitive,
    digits_words,
    plural_form,
    time_words,
)


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "ноль"),
        (1, "один"),
        (11, "одиннадцать"),
        (20, "двадцать"),
        (21, "двадцать один"),
        (99, "девяносто девять"),
        (100, "сто"),
        (105, "сто пять"),
        (999, "девятьсот девяносто девять"),
        (1000, "одна тысяча"),
        (2000, "две тысячи"),
        (3500, "три тысячи пятьсот"),
        (15000, "пятнадцать тысяч"),
        (21000, "двадцать одна тысяча"),
        (35000, "тридцать пять тысяч"),
        (112000, "сто двенадцать тысяч"),
        (999999, "девятьсот девяносто девять тысяч девятьсот девяносто девять"),
    ],
)
def test_cardinal(n, words):
    assert cardinal(n) == words


@pytest.mark.parametrize("n", [-1, 1_000_000])
def test_cardinal_out_of_range(n):
    with pytest.raises(ValueError):
        cardinal(n)


@pytest.mark.parametrize(
    ("n", "form"),
    [(1, "час"), (2, "часа"), (4, "часа"), (5, "часов"), (11, "часов"), (14, "часов")]
    + [(21, "час"), (22, "часа"), (25, "часов"), (111, "часов"), (101, "час")],
)
def test_plural_form(n, form):
    assert plural_form(n, "час", "часа", "часов") == form


def test_digits_words_reads_each_digit_and_ignores_other_characters():
    assert digits_words("4567") == "четыре пять шесть семь"
    assert digits_words("+7 (916) 0") == "семь девять один шесть ноль"
    assert digits_words("") == ""


@pytest.mark.parametrize(
    ("day", "words"),
    [
        (1, "первого"),
        (2, "второго"),
        (3, "третьего"),
        (4, "четвёртого"),
        (9, "девятого"),
        (10, "десятого"),
        (11, "одиннадцатого"),
        (19, "девятнадцатого"),
        (20, "двадцатого"),
        (21, "двадцать первого"),
        (26, "двадцать шестого"),
        (30, "тридцатого"),
        (31, "тридцать первого"),
    ],
)
def test_day_ordinal_genitive(day, words):
    assert day_ordinal_genitive(day) == words


@pytest.mark.parametrize("day", [0, 32])
def test_day_ordinal_out_of_range(day):
    with pytest.raises(ValueError):
        day_ordinal_genitive(day)


def test_every_day_of_the_month_has_an_ordinal():
    assert all(day_ordinal_genitive(d).endswith(("ого", "его")) for d in range(1, 32))


def test_date_words_and_on_phrase():
    assert date_words(date(2026, 9, 26)) == "двадцать шестого сентября"
    assert date_on_phrase(date(2026, 9, 26)) == "в субботу, двадцать шестого сентября"  # Saturday
    assert date_on_phrase(date(2026, 9, 29)) == "во вторник, двадцать девятого сентября"
    assert date_on_phrase(date(2026, 9, 30)) == "в среду, тридцатого сентября"
    assert date_on_phrase(date(2026, 10, 1)) == "в четверг, первого октября"


def test_every_weekday_has_a_preposition_form():
    forms = {date_on_phrase(date(2026, 9, 21 + i)).split(",")[0] for i in range(7)}
    assert forms == {
        "в понедельник",
        "во вторник",
        "в среду",
        "в четверг",
        "в пятницу",
        "в субботу",
        "в воскресенье",
    }


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (time(14, 30), "четырнадцать тридцать"),
        (time(10, 0), "десять часов"),
        (time(14, 0), "четырнадцать часов"),
        (time(11, 0), "одиннадцать часов"),
        (time(12, 0), "двенадцать часов"),
        (time(13, 0), "тринадцать часов"),
        (time(21, 0), "двадцать один час"),
        (time(22, 0), "двадцать два часа"),
        (time(2, 0), "два часа"),
        (time(1, 0), "час"),
        (time(10, 5), "десять ноль пять"),
        (time(9, 9), "девять ноль девять"),
        (time(15, 45), "пятнадцать сорок пять"),
        (time(19, 59), "девятнадцать пятьдесят девять"),
    ],
)
def test_time_words(value, words):
    assert time_words(value) == words
