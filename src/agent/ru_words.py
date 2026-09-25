"""Russian number, date and time words, for text that will be spoken by TTS.

Pure functions and tables with no dependency on the rest of the app, so both the tools (code-built
read-backs) and, later, text normalization before TTS can use them.
"""

import re
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


# --- Reading numbers back out of text -------------------------------------------------------------
#
# Used by the scenario checks to verify what the agent SAID (prices, phone digits). It understands
# number words in the nominative and genitive («три тысячи», «от трёх до пяти тысяч»), the case
# forms a model uses for prices, and digit forms («15 000 ₽»). Callers that only care about
# money should look at sentences that contain a money cue (рубл*, тысяч*, ₽): a phone number
# read in groups adds up to a plausible-looking "amount".

_NUMBER_WORDS: dict[str, int] = {}
for _value, _forms in {
    0: "ноль ноля",
    1: "один одна одно одного одной одну",
    2: "два две двух",
    3: "три трёх трех",
    4: "четыре четырёх четырех",
    5: "пять пяти",
    6: "шесть шести",
    7: "семь семи",
    8: "восемь восьми",
    9: "девять девяти",
    10: "десять десяти",
    11: "одиннадцать одиннадцати",
    12: "двенадцать двенадцати",
    13: "тринадцать тринадцати",
    14: "четырнадцать четырнадцати",
    15: "пятнадцать пятнадцати",
    16: "шестнадцать шестнадцати",
    17: "семнадцать семнадцати",
    18: "восемнадцать восемнадцати",
    19: "девятнадцать девятнадцати",
    20: "двадцать двадцати",
    30: "тридцать тридцати",
    40: "сорок сорока",
    50: "пятьдесят пятидесяти",
    60: "шестьдесят шестидесяти",
    70: "семьдесят семидесяти",
    80: "восемьдесят восьмидесяти",
    90: "девяносто девяноста",
    100: "сто ста",
    200: "двести двухсот",
    300: "триста трёхсот трехсот",
    400: "четыреста четырёхсот четырехсот",
    500: "пятьсот пятисот",
    600: "шестьсот шестисот",
    700: "семьсот семисот",
    800: "восемьсот восьмисот",
    900: "девятьсот девятисот",
}.items():
    for _form in _forms.split():
        _NUMBER_WORDS[_form] = _value

_WORD_RE = re.compile(r"[а-яёА-ЯЁ]+|\d+")
_THOUSAND_RE = re.compile(r"тысяч\w*|тыс")
_MONEY_RE = re.compile(r"рубл\w*|руб|₽")
_YEAR_RE = re.compile(r"год\w*")
_SEPARATORS = {"до", "-", "–", "—"}
_DIGIT_GROUP_RE = re.compile(r"\d{1,3}(?:[ \u00a0]\d{3})+|\d+")


def _norm(word: str) -> str:
    return word.lower().replace("ё", "е")


def spoken_amounts(text: str) -> list[int]:
    """Money-like amounts mentioned in `text`, in order: 'от трёх до пяти тысяч рублей' ->
    [3000, 5000]; 'двадцать пять тысяч' -> [25000]; '15 000 ₽' -> [15000]; '3 500 рублей' ->
    [3500]. Small numbers (times, dates, counts) are ignored unless a rouble word follows them;
    numbers followed by «год» (years) are ignored."""
    amounts: list[int] = []

    # digit forms: "15 000", "15000", "15 тысяч"
    for match in _DIGIT_GROUP_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) > 6:
            continue  # a phone number, not an amount
        after = text[match.end() : match.end() + 12]
        tail = after.lstrip(" \u00a0").lower()
        value = int(digits)
        if _THOUSAND_RE.match(tail):
            value *= 1000
        elif value < 100 and not _MONEY_RE.match(tail):
            continue
        if _YEAR_RE.match(tail):
            continue
        if value >= 100 or _MONEY_RE.match(tail):
            amounts.append(value)

    # number words
    tokens = [_norm(t) for t in _WORD_RE.findall(text)]
    found: list[tuple[int, bool, int]] = []  # (value, has_thousands, index of the last token)
    total = group = 0
    seen_number = has_thousands = previous_was_digits = False
    last_index = -1

    def flush() -> None:
        nonlocal total, group, seen_number, has_thousands
        if seen_number:
            found.append((total + group, has_thousands, last_index))
        total = group = 0
        seen_number = has_thousands = False

    for index, token in enumerate(tokens):
        if token.isdigit():
            flush()
            previous_was_digits = True
            continue
        was_digits, previous_was_digits = previous_was_digits, False
        if token in _NUMBER_WORDS:
            group += _NUMBER_WORDS[token]
            seen_number = True
            last_index = index
        elif _THOUSAND_RE.fullmatch(token) and was_digits and not seen_number:
            continue  # «15 тысяч»: already read by the digit pass above
        elif _THOUSAND_RE.fullmatch(token) and (seen_number or token.startswith("тысяч")):
            total += (group or 1) * 1000
            group = 0
            seen_number = has_thousands = True
            last_index = index
        else:
            flush()
    flush()

    for position, (value, has_thousands, last) in enumerate(found):
        followed_by = tokens[last + 1] if last + 1 < len(tokens) else ""
        if any(_YEAR_RE.fullmatch(t) for t in tokens[last + 1 : last + 3]):
            continue  # «две тысячи двадцать шестой год»
        # «от трёх до пяти тысяч»: the first number shares the thousands of the second.
        if not has_thousands and value < 1000 and position + 1 < len(found):
            nxt_value, nxt_thousands, _ = found[position + 1]
            between = tokens[last + 1 : last + 2]
            if nxt_thousands and between and between[0] in _SEPARATORS and nxt_value >= 1000:
                value *= 1000
        if value >= 100 or _MONEY_RE.fullmatch(followed_by):
            amounts.append(value)
    return sorted(set(amounts), key=amounts.index)


_DIGITS_ONLY_RE = re.compile(r"[+\d()\-–]+")


def longest_number_run(text: str) -> int:
    """The longest run of consecutive numbers, counted in number words / digits, within a
    sentence. A phone number read aloud is a long run (>= 6), while the four digits of a
    read-back, a date or a price are short ('четыре пять шесть семь' = 4). Dates and times
    written with punctuation ('26.09.2026', '14:30') do not count as digits."""
    longest = run = 0
    for raw in text.split():
        core = raw.strip(",.;:!?«»\"'")
        norm = _norm(core)
        if _DIGITS_ONLY_RE.fullmatch(core) and any(ch.isdigit() for ch in core):
            run += sum(ch.isdigit() for ch in core)
        elif norm in _NUMBER_WORDS or _THOUSAND_RE.fullmatch(norm):
            run += 1
        else:
            run = 0
        longest = max(longest, run)
        if raw.endswith((".", "!", "?")):
            run = 0  # a sentence ends the run
    return longest
