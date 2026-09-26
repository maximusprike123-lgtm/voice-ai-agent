"""Text on its way to the voice: numbers and Latin words as spoken Russian, then the speech guard.

Every sentence that reaches TTS goes through `SpeechText.prepare`:
  1. `normalize_for_speech`: digits, times, dates, prices and percentages become words, Latin
     words (car makes, «PPF») become Russian, markup and emoji go. The model is told to write
     numbers in words, so this rarely does anything (2 of 6796 real sentences had digits); it is
     the safety net, and every change is reported so a model that drifts shows up in the logs.
  2. The speech guard checks BOTH the original and the final text: rules that look at the raw
     form (a Latin role label such as «user», digits) must see it before normalization turns
     it into something else, and it understands number words, so a phone number is caught after
     normalization too. Text the model wrote is dropped on a violation. Text the
     code wrote (`scripted`: read-back, acceptance, greeting, fallbacks) is spoken anyway, with
     an ERROR in the log, because silence in the middle of a booking is worse than a slip: the
     tests check that code-built sentences pass the guard. Two rules stay strict even there:
     a phone number read out and characters of another script.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import time

from agent.ru_words import (
    MONTHS_RU_GENITIVE,
    cardinal,
    cardinal_genitive,
    day_ordinal_genitive,
    digits_words,
    genitive_noun,
    plural_form,
    time_words,
)
from agent.text_guard import SpeechGuard, Violation

logger = logging.getLogger(__name__)

# Rules that silence a sentence even when the code wrote it.
STRICT_RULES = frozenset({"phone_digits", "foreign_script"})

# --- Latin words ------------------------------------------------------------------------------

# How Russian speech says them. Lower case keys; a hyphenated word is looked up whole first, then
# by its parts. Abbreviations are spelled the way a Russian speaker reads the letters.
LATIN_WORDS = {
    "toyota": "Тойота",
    "camry": "Камри",
    "corolla": "Королла",
    "rav4": "РАВ четыре",
    "land": "Лэнд",
    "cruiser": "Крузер",
    "prado": "Прадо",
    "highlander": "Хайлендер",
    "mercedes": "Мерседес",
    "benz": "Бенц",
    "bmw": "бэ эм вэ",
    "audi": "Ауди",
    "volkswagen": "Фольксваген",
    "vw": "фау вэ",
    "skoda": "Шкода",
    "octavia": "Октавия",
    "kia": "Киа",
    "rio": "Рио",
    "sportage": "Спортейдж",
    "hyundai": "Хендай",
    "solaris": "Солярис",
    "creta": "Крета",
    "nissan": "Ниссан",
    "qashqai": "Кашкай",
    "honda": "Хонда",
    "mazda": "Мазда",
    "lexus": "Лексус",
    "porsche": "Порше",
    "ford": "Форд",
    "focus": "Фокус",
    "chevrolet": "Шевроле",
    "opel": "Опель",
    "renault": "Рено",
    "peugeot": "Пежо",
    "citroen": "Ситроен",
    "subaru": "Субару",
    "mitsubishi": "Мицубиси",
    "volvo": "Вольво",
    "rover": "Ровер",
    "range": "Рейндж",
    "tesla": "Тесла",
    "lada": "Лада",
    "geely": "Джили",
    "chery": "Чери",
    "haval": "Хавейл",
    "jeep": "Джип",
    "mini": "Мини",
    "suzuki": "Сузуки",
    "infiniti": "Инфинити",
    "genesis": "Дженесис",
    "x5": "икс пять",
    "x3": "икс три",
    "x6": "икс шесть",
    "ppf": "пэ пэ эф",
    "pdr": "пэ дэ эр",
    "suv": "эс ю ви",
    "led": "лэд",
    "abs": "а бэ эс",
    "vin": "вин",
}

_TRANSLIT_DIGRAPHS = (
    ("shch", "щ"),
    ("sch", "щ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("yo", "ё"),
    ("yu", "ю"),
    ("ya", "я"),
    ("ye", "е"),
    ("ph", "ф"),
    ("th", "т"),
    ("ck", "к"),
    ("oo", "у"),
    ("ee", "и"),
)
_TRANSLIT_LETTERS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
    "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}  # fmt: skip


def transliterate(word: str) -> str:
    """A rough Russian spelling of an unknown Latin word (letter by letter, common digraphs)."""
    text, out, i = word.lower(), [], 0
    while i < len(text):
        for latin, cyrillic in _TRANSLIT_DIGRAPHS:
            if text.startswith(latin, i):
                out.append(cyrillic)
                i += len(latin)
                break
        else:
            char = text[i]
            if char == "c" and text[i + 1 : i + 2] in ("e", "i", "y"):
                out.append("с")
            else:
                out.append(_TRANSLIT_LETTERS.get(char, char))
            i += 1
    return "".join(out).capitalize() if word[:1].isupper() and not word.isupper() else "".join(out)


_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")


def _latin_replacement(match: re.Match, unknown: list[str]) -> str:
    word = match.group()
    if known := LATIN_WORDS.get(word.lower()):
        return known
    parts = word.split("-")
    if len(parts) > 1 and all(part.lower() in LATIN_WORDS for part in parts):
        return " ".join(LATIN_WORDS[part.lower()] for part in parts)
    unknown.append(word)
    return " ".join(transliterate(part) for part in parts)


# --- Numbers ----------------------------------------------------------------------------------

_NUMBER = r"\d{1,3}(?:[  ]\d{3})+|\d+"
_GENITIVE_CUES = {"от", "до", "около", "свыше", "менее", "более", "без"}
# «с 10:00 до 21:00», «после 18:30»: the hour is in the genitive after these
_TIME_GENITIVE_CUES = _GENITIVE_CUES | {"с", "после", "перед"}

_TIME = re.compile(r"(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])")
_DATE = re.compile(
    r"(?<![\d.])(\d{2})\.(\d{2})(?:\.(\d{2,4}))?(?![\d.])|(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{2,4})(?![\d.])"
)
_THOUSANDS = re.compile(rf"(?P<n>{_NUMBER})\s*тыс(?:\.|яч\w*)?(?!\w)", re.I)
_MONEY = re.compile(rf"(?P<n>{_NUMBER})\s*(?:₽|руб\.|р\.)", re.I)
_PERCENT = re.compile(rf"(?P<n>{_NUMBER})\s*%")
_DECIMAL = re.compile(r"(?<![\d,.])(\d+)[,.](\d+)(?![\d,.])")
_BARE_NUMBER = re.compile(rf"(?<![\d.,])(?P<n>{_NUMBER})(?!\d)")


def _to_int(digits: str) -> int:
    return int(re.sub(r"\D", "", digits))


def _words_for(n: int, genitive: bool) -> str:
    if n > 999_999:
        return digits_words(str(n))  # too big to say as one number: digit by digit
    return cardinal_genitive(n) if genitive else cardinal(n)


def _after_cue(text: str, start: int, cues: set[str] | frozenset[str]) -> bool:
    before = re.findall(r"[а-яёА-ЯЁ]+", text[max(0, start - 12) : start])
    return bool(before) and before[-1].lower() in cues


def _after_genitive_cue(text: str, start: int) -> bool:
    return _after_cue(text, start, _GENITIVE_CUES)


def _spaced(text: str, start: int, end: int, words: str) -> str:
    """`words` with a space on a side where a letter touches it («РАВ4» -> «РАВ четыре»)."""
    left = " " if start > 0 and text[start - 1].isalpha() else ""
    right = " " if end < len(text) and text[end].isalpha() else ""
    return f"{left}{words}{right}"


def _sub(pattern: re.Pattern, text: str, make, changes: list[str], label: str) -> str:
    def replace(match: re.Match) -> str:
        replacement = make(match, text)
        if replacement is None:
            return match.group()
        if label not in changes:
            changes.append(label)
        return replacement

    return pattern.sub(replace, text)


def _make_time(match: re.Match, text: str) -> str | None:
    hour, minute = int(match[1]), int(match[2])
    if hour > 23 or minute > 59:
        return None
    if not _after_cue(text, match.start(), _TIME_GENITIVE_CUES):
        return time_words(time(hour, minute))
    hours = cardinal_genitive(hour)
    if minute == 0:
        return f"{hours} {genitive_noun(hour, 'часа', 'часов')}"
    minutes = f"ноля {cardinal_genitive(minute)}" if minute < 10 else cardinal_genitive(minute)
    return f"{hours} {minutes}"


def _make_date(match: re.Match, text: str) -> str | None:
    day, month = (match[1], match[2]) if match[1] else (match[4], match[5])
    day, month = int(day), int(month)
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{day_ordinal_genitive(day)} {MONTHS_RU_GENITIVE[month - 1]}"


def _make_thousands(match: re.Match, text: str) -> str:
    n = _to_int(match["n"]) * 1000
    return _words_for(n, _after_genitive_cue(text, match.start()))


def _make_money(match: re.Match, text: str) -> str:
    n = _to_int(match["n"])
    if _after_genitive_cue(text, match.start()):
        return f"{_words_for(n, True)} {genitive_noun(n, 'рубля', 'рублей')}"
    return f"{_words_for(n, False)} {plural_form(n, 'рубль', 'рубля', 'рублей')}"


def _make_percent(match: re.Match, text: str) -> str:
    n = _to_int(match["n"])
    if _after_genitive_cue(text, match.start()):
        return f"{_words_for(n, True)} {genitive_noun(n, 'процента', 'процентов')}"
    return f"{_words_for(n, False)} {plural_form(n, 'процент', 'процента', 'процентов')}"


def _make_decimal(match: re.Match, text: str) -> str:
    whole, fraction = _to_int(match[1]), match[2]
    return f"{_words_for(whole, False)} запятая {digits_words(fraction)}"


def _make_bare(match: re.Match, text: str) -> str:
    n = _to_int(match["n"])
    words = _words_for(n, _after_genitive_cue(text, match.start()))
    return _spaced(text, match.start(), match.end(), words)


# --- Markup and symbols -----------------------------------------------------------------------

_EMPHASIS = re.compile(r"[*_#`~^]")  # markdown: dropped without a trace
_BRACKETS = re.compile(r"[|\\<>\[\]{}]")  # dropped, but keep the words on both sides apart
_INVISIBLE = re.compile(r"[​-‏⁠﻿]")


def _strip_symbols(text: str) -> str:
    """Drop markdown characters, zero-width characters and emoji; keep letters, digits,
    punctuation and currency signs."""
    text = _INVISIBLE.sub("", _BRACKETS.sub(" ", _EMPHASIS.sub("", text)))
    return "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Cn"))


def normalize_for_speech(text: str) -> tuple[str, list[str]]:
    """The text as it should be spoken, and what was changed («time», «money», «latin»...)."""
    changes: list[str] = []
    text = text.replace("№", "номер ")
    stripped = _strip_symbols(text)
    if stripped != text:
        changes.append("markup")
    text = stripped

    text = _sub(_TIME, text, _make_time, changes, "time")
    text = _sub(_DATE, text, _make_date, changes, "date")
    text = _sub(_THOUSANDS, text, _make_thousands, changes, "thousands")
    text = _sub(_MONEY, text, _make_money, changes, "money")
    text = _sub(_PERCENT, text, _make_percent, changes, "percent")
    text = _sub(_DECIMAL, text, _make_decimal, changes, "decimal")

    unknown: list[str] = []

    def latin(match: re.Match) -> str:
        if "latin" not in changes:
            changes.append("latin")
        return _latin_replacement(match, unknown)

    text = _LATIN_WORD.sub(latin, text)
    if unknown:
        logger.warning("Latin words without a known Russian reading, transliterated: %s", unknown)
    text = _sub(_BARE_NUMBER, text, _make_bare, changes, "number")
    return re.sub(r"[ \t]+", " ", text).strip(), changes


# --- The stage before TTS ---------------------------------------------------------------------


@dataclass(frozen=True)
class SpokenText:
    """The result of preparing one sentence for the voice."""

    text: str | None  # what to say; None: say nothing
    changes: tuple[str, ...] = ()  # what normalization changed (empty: it was already speakable)
    violation: Violation | None = None  # the guard's finding, if any (may still be spoken)


@dataclass
class SpeechText:
    """Prepares every sentence for TTS. `guard` is the call's own SpeechGuard (the one the engine
    uses, so it knows whether anything was saved); None skips the guard."""

    guard: SpeechGuard | None = None
    code_violations: list[Violation] = field(default_factory=list)  # slips in code-built text

    def prepare(self, sentence: str, *, scripted: bool = False) -> SpokenText:
        text, changes = normalize_for_speech(sentence)
        if changes:
            # Latin car names in a code-built read-back are expected; anything else is news.
            expected = scripted and changes == ["latin"]
            (logger.info if expected else logger.warning)(
                "normalized %s sentence for speech (%s): %r -> %r",
                "code-built" if scripted else "model",
                ", ".join(changes),
                sentence,
                text,
            )
        if not text:
            return SpokenText(None, tuple(changes))

        violation = None
        if self.guard is not None:
            violation = self.guard.check(sentence) or self.guard.check(text)
        if violation is None:
            return SpokenText(text, tuple(changes))

        if not scripted:
            self.guard.blocked.append(violation)  # type: ignore[union-attr]
            logger.warning("speech guard blocked (%s) before TTS: %r", violation.rule, text)
            return SpokenText(None, tuple(changes), violation)

        self.code_violations.append(violation)
        if violation.rule in STRICT_RULES:
            logger.error("code-built sentence dropped (%s): %r", violation.rule, text)
            return SpokenText(None, tuple(changes), violation)
        logger.error(
            "code-built sentence trips the speech guard (%s), spoken anyway: %r",
            violation.rule,
            text,
        )
        return SpokenText(text, tuple(changes), violation)
