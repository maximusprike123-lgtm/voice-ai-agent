"""Russian number, date and time words, for text that will be spoken by TTS.

Pure functions and tables with no dependency on the rest of the app, so both the tools (code-built
read-backs) and, later, text normalization before TTS can use them.
"""

from datetime import date, time

WEEKDAYS_RU = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
# "on <weekday>", preposition included ("во вторник", "в среду").
WEEKDAYS_ON_RU = (
    "в понедельник",
    "во вторник",
    "в среду",
    "в четверг",
    "в пятницу",
    "в субботу",
    "в воскресенье",
)
MONTHS_RU_GENITIVE = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

_UNITS = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_UNITS_FEMININE = {1: "одна", 2: "две"}
_TEENS = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)

# Day of month as a genitive ordinal, for "двадцать шестого сентября".
_ORDINAL_UNITS_GENITIVE = (
    "",
    "первого",
    "второго",
    "третьего",
    "четвёртого",
    "пятого",
    "шестого",
    "седьмого",
    "восьмого",
    "девятого",
)
_ORDINAL_TEENS_GENITIVE = (
    "десятого",
    "одиннадцатого",
    "двенадцатого",
    "тринадцатого",
    "четырнадцатого",
    "пятнадцатого",
    "шестнадцатого",
    "семнадцатого",
    "восемнадцатого",
    "девятнадцатого",
)


def _below_thousand(n: int, feminine: bool = False) -> str:
    words = []
    if n >= 100:
        words.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
        if n:
            words.append(_UNITS_FEMININE[n] if feminine and n in _UNITS_FEMININE else _UNITS[n])
    elif n >= 10:
        words.append(_TEENS[n - 10])
    elif n or not words:
        words.append(_UNITS_FEMININE[n] if feminine and n in _UNITS_FEMININE else _UNITS[n])
    return " ".join(words)


def plural_form(n: int, one: str, few: str, many: str) -> str:
    """Pick the noun form for a count: 1 час, 2 часа, 5 часов, 11 часов, 21 час."""
    if 11 <= n % 100 <= 14:
        return many
    if n % 10 == 1:
        return one
    if 2 <= n % 10 <= 4:
        return few
    return many


def cardinal(n: int) -> str:
    """0..999999 as words, nominative masculine ('двадцать один', 'две тысячи пятьсот')."""
    if not 0 <= n <= 999_999:
        raise ValueError(f"cardinal supports 0..999999, got {n}")
    thousands, rest = divmod(n, 1000)
    if not thousands:
        return _below_thousand(rest)
    words = [
        _below_thousand(thousands, feminine=True),
        plural_form(thousands, "тысяча", "тысячи", "тысяч"),
    ]
    if rest:
        words.append(_below_thousand(rest))
    return " ".join(words)


def digits_words(text: str) -> str:
    """Every digit read on its own, non-digits ignored: '4567' -> 'четыре пять шесть семь'."""
    return " ".join(_UNITS[int(ch)] for ch in text if ch.isdigit())


def day_ordinal_genitive(day: int) -> str:
    """1..31 as a genitive ordinal: 26 -> 'двадцать шестого', 30 -> 'тридцатого'."""
    if not 1 <= day <= 31:
        raise ValueError(f"day of month must be 1..31, got {day}")
    if day < 10:
        return _ORDINAL_UNITS_GENITIVE[day]
    if day < 20:
        return _ORDINAL_TEENS_GENITIVE[day - 10]
    tens, unit = divmod(day, 10)
    if unit == 0:
        return f"{_TENS[tens][:-1]}ого" if tens == 3 else "двадцатого"
    return f"{_TENS[tens]} {_ORDINAL_UNITS_GENITIVE[unit]}"


def date_words(day: date) -> str:
    """'двадцать шестого сентября' (no year, no weekday)."""
    return f"{day_ordinal_genitive(day.day)} {MONTHS_RU_GENITIVE[day.month - 1]}"


def date_on_phrase(day: date) -> str:
    """'в субботу, двадцать шестого сентября'."""
    return f"{WEEKDAYS_ON_RU[day.weekday()]}, {date_words(day)}"


def time_words(value: time) -> str:
    """Time of day as it is said aloud: 14:30 -> 'четырнадцать тридцать', 10:00 -> 'десять часов'.

    No preposition; callers add 'в' ('в десять часов', 'в час').
    """
    hour, minute = value.hour, value.minute
    if minute == 0:
        if hour == 1:
            return "час"
        return f"{cardinal(hour)} {plural_form(hour, 'час', 'часа', 'часов')}"
    minutes = f"ноль {_UNITS[minute]}" if minute < 10 else cardinal(minute)
    return f"{cardinal(hour)} {minutes}"
