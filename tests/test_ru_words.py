from datetime import date, time

import pytest

from agent.ru_words import (
    DigitRun,
    cardinal,
    date_on_phrase,
    date_words,
    day_ordinal_genitive,
    digits_words,
    longest_number_run,
    plural_form,
    spoken_amounts,
    spoken_digit_runs,
    spoken_digits,
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


# --- spoken_amounts / longest_number_run ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "amounts"),
    [
        ("Полировка кузова стоит от пятнадцати до тридцати пяти тысяч рублей.", [15000, 35000]),
        ("Это от трёх до пяти тысяч рублей.", [3000, 5000]),
        ("Керамика — от двадцати пяти тысяч рублей.", [25000]),
        ("От десяти до восемнадцати тысяч.", [10000, 18000]),
        ("Две тысячи пятьсот рублей", [2500]),
        ("Одна тысяча рублей", [1000]),
        ("Сто рублей", [100]),
        ("от 3 000 до 5 000 ₽", [3000, 5000]),
        ("от 3\u00a0000 до 5\u00a0000 руб.", [3000, 5000]),
        ("15000 рублей", [15000]),
        ("около 15 тысяч", [15000]),
        ("Итого 3 500 рублей.", [3500]),
        ("Стоит 50 рублей", [50]),
    ],
)
def test_spoken_amounts_reads_prices(text, amounts):
    assert spoken_amounts(text) == amounts


@pytest.mark.parametrize(
    "text",
    [
        "Приезжайте в четырнадцать тридцать.",
        "В субботу, двадцать шестого сентября, днём.",
        "Работаем с десяти до двадцати часов.",
        "Номер заканчивается на четыре пять шесть семь.",
        "Займёт два часа.",
        "Это займёт от одного до двух дней.",
        "Сегодня 26 сентября 2026 года, 14:30.",
        "В две тысячи двадцать шестом году.",
        "Позвоните +7 916 123 45 67 или 89161234567.",
        "",
    ],
)
def test_spoken_amounts_ignores_times_dates_counts_years_and_phones(text):
    assert spoken_amounts(text) == []


def test_spoken_amounts_lists_each_amount_once_in_order():
    text = (
        "От пятнадцати тысяч рублей, ещё раз: от пятнадцати тысяч рублей и до тридцати пяти тысяч"
    )
    assert spoken_amounts(text) == [15000, 35000]


@pytest.mark.parametrize(
    ("text", "run"),
    [
        ("Номер заканчивается на четыре пять шесть семь.", 4),
        ("восемь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь", 10),
        ("Позвоните 8 916 123 45 67", 11),
        ("+79161234567", 11),
        ("в субботу, двадцать шестого сентября, в четырнадцать тридцать", 2),
        ("от пятнадцати до тридцати пяти тысяч рублей", 3),
        ("Здравствуйте, чем могу помочь?", 0),
        ("Приезжайте 26.09.2026 в 14:30", 0),
        ("Позвоните на +7 (916) 123-45-67.", 11),
        ("восемь девятьсот шестнадцать, сто двадцать три, сорок пять, шестьдесят семь", 10),
        ("Четыре пять. Шесть семь.", 2),
        ("", 0),
    ],
)
def test_longest_number_run_separates_phone_dictation_from_normal_numbers(text, run):
    assert longest_number_run(text) == run


def test_a_read_back_run_stays_below_the_phone_threshold():
    assert longest_number_run("Номер телефона заканчивается на четыре пять шесть семь.") < 6


# --- spoken_digits ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("8 (916) 123-45-67", "89161234567"),
        ("плюс 7 916 123 45 67", "79161234567"),
        ("восемь девять один шесть один два три четыре пять шесть семь", "89161234567"),
        ("восемь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь", "89161234567"),
        ("девятьсот пять", "905"),
        ("девятьсот десять", "910"),
        ("двадцать два ноль ноль", "2200"),
        ("девять шестнадцать", "916"),
        ("Меня зовут Игорь, номер 916 123 45 67, спасибо", "9161234567"),
        ("восемь девятьсот шестнадцать, сто двадцать три", "8916123"),  # commas do not matter
        ("нет цифр", ""),
        ("", ""),
    ],
)
def test_spoken_digits(text, digits):
    assert spoken_digits(text) == digits


def test_spoken_digits_does_not_glue_numerals_that_cannot_belong_together():
    # «сорок» + «двадцать»: two tens are two numbers; «пять» closes a group of units
    assert spoken_digits("сорок двадцать") == "4020"
    assert spoken_digits("пять пять") == "55"
    assert spoken_digits("сто двести") == "100200"


# --- spoken_digit_runs --------------------------------------------------------------------------


def runs(text):
    return [(r.digits, r.at_start, r.at_end) for r in spoken_digit_runs(text)]


def test_a_run_is_broken_by_any_other_word_but_not_by_separators():
    assert runs("+7 (916) 123-45-67") == [("79161234567", True, True)]
    assert runs("завтра в 14:00, номер 8 916 123 45 67") == [
        ("1400", False, False),
        ("89161234567", False, True),
    ]


def test_a_run_of_number_words_and_digits_is_one_run():
    assert runs("восемь 916 сто двадцать три") == [("8916123", True, True)]


def test_time_with_a_colon_is_two_pieces_of_one_run():
    assert runs("в 14:00") == [("1400", False, True)]


def test_at_start_and_at_end_mark_the_edges_of_the_utterance():
    assert runs("8 916 123") == [("8916123", True, True)]
    assert runs("номер 8 916") == [("8916", False, True)]
    assert runs("8 916 это номер") == [("8916", True, False)]
    assert runs("да") == []


def test_plus_and_latin_words_and_the_word_plus():
    assert runs("плюс семь девятьсот шестнадцать") == [("7916", True, True)]
    assert runs("Camry 2015") == [("2015", False, True)]
    assert runs("2015 Camry") == [("2015", True, False)]


def test_spoken_digits_is_the_runs_joined():
    text = "в 14:00, номер восемь девятьсот шестнадцать 123 45 67, Камри 2015"
    assert spoken_digits(text) == "".join(r.digits for r in spoken_digit_runs(text))
    assert isinstance(spoken_digit_runs(text)[0], DigitRun)
