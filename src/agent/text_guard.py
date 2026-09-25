"""The speech guard: deterministic checks on every sentence the MODEL wrote, before it is spoken.

Code enforces guarantees the prompt cannot (see CLAUDE.md). Measured on real calls, the model
sometimes reads a full phone number aloud, says «записал» before anything is saved, says «заявка
принята» without ever calling confirm_booking, writes the CALLER's lines or a role label into its
own reply («userЗаписывай на тот, что я продиктовал…»), or slips characters from another script
into a Russian sentence («тридцати五千 рублей»). `SpeechGuard.check` finds those, with no LLM
call. What happens to a blocked sentence is decided by the caller: DialogueEngine drops it, and
only if the
whole reply would then be silent asks the model once more (with `correction_note`) or speaks the
neutral fallback for the rule. Text the code itself wrote (read-back, acceptance, greeting) is
not checked. The audio pipeline (step 2) reuses the same guard before TTS.

Rules, in order: foreign_script, role_leakage, phone_digits, acceptance_claim, written_down.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from agent.ru_words import longest_number_run

logger = logging.getLogger(__name__)

# A phone number read aloud is a long run of digits or number words; the code-built read-back
# says exactly four, which is allowed.
PHONE_RUN_LIMIT = 4
PHONE_GROUP_DIGITS = 7  # in a sentence about money, only a digit group this long is a phone

# Sentences about money legitimately contain long numbers («15 000 ₽»).
_MONEY_CUE = re.compile(r"рубл|\bруб\b|₽|тысяч|\bтыс\b", re.I)
_DIGIT_GROUP = re.compile(r"\d[\d\s\-()+]{5,}\d")

# Completed-action claims: "the request has been accepted / passed on". Future or conditional
# explanations («администратор свяжется с вами») are deliberately NOT matched: in real calls they
# were legitimate (measured: 2 of 3 sentences of that kind).
ACCEPTANCE_CLAIM = re.compile(
    r"заявк\w*\s+(?:уже\s+)?(?:принят|передан|отправлен|оформлен)\w*"
    r"|принят\w*\s+и\s+передан\w*|сообщение\s+передан\w*|ваша\s+просьба\s+принята"
    r"|запись\s+оформлена",
    re.I,
)
# Role leakage: the model writes a role label («user», «assistant», «system», «Клиент:», «Агент:»)
# or speaks as the caller. Latin role words count when they open the sentence, carry a colon, or
# are glued to Cyrillic («userЗаписывай»); inside other Latin words («ecosystem») they do not.
_LATIN_ROLES = r"(?:user|assistant|system)"
ROLE_LEAKAGE = re.compile(
    rf"^\s*{_LATIN_ROLES}\b"
    rf"|\b{_LATIN_ROLES}\s*[:：]"
    rf"|(?<=[А-Яа-яЁё]){_LATIN_ROLES}"
    rf"|{_LATIN_ROLES}(?=[А-Яа-яЁё])"
    r"|(?<![А-Яа-яЁёA-Za-z])(?:клиент|агент|администратор|ассистент|пользователь|система)\s*[:：]",
    re.I,
)
WRITTEN_DOWN = re.compile(r"\bзаписал[аи]?\b|\bзаписан[оа]?\b", re.I)

# What is said instead when a whole reply was blocked and the corrective round failed too.
# Gender-neutral, no digits, and they never blame the caller.
GUARD_FALLBACKS = {
    "role_leakage": "Давайте продолжим.",
    "phone_digits": "Хорошо, номер есть.",
    "acceptance_claim": "Давайте ещё раз проверим данные заявки.",
    "written_down": "Хорошо.",
    "foreign_script": "Простите, уточните, пожалуйста, ваш вопрос.",
}

# Hidden notes for the model (never spoken) that explain what was blocked, for the corrective
# round. {sentence} is the blocked text.
_CORRECTIONS = {
    "role_leakage": (
        "Твоя фраза «{sentence}» заблокирована: в ней служебное слово или реплика от имени "
        "клиента (user, assistant, system, «Клиент:», «Агент:»). Отвечай только своей репликой "
        "администратора, без ролей и без слов за клиента."
    ),
    "phone_digits": (
        "Твоя фраза «{sentence}» заблокирована: нельзя произносить цифры телефона. Продолжи "
        "разговор без них: задай следующий вопрос или вызови нужный инструмент."
    ),
    "acceptance_claim": (
        "Твоя фраза «{sentence}» заблокирована: заявка НЕ сохранена. Если клиент ясно "
        "подтвердил зачитанные данные, вызови confirm_booking; если данные ещё не "
        "подготовлены, вызови prepare_booking; иначе продолжай собирать данные. Не говори, что "
        "заявка принята или передана."
    ),
    "written_down": (
        "Твоя фраза «{sentence}» заблокирована: слова «записал», «записала», «записано» до "
        "сохранения заявки запрещены. Продолжи разговор нейтрально: «хорошо», «принято»."
    ),
    "foreign_script": (
        "Твоя фраза «{sentence}» заблокирована: в ней были символы не русского алфавита. "
        "Скажи то же самое только по-русски."
    ),
}


@dataclass(frozen=True)
class Violation:
    rule: str
    sentence: str
    detail: str = ""


def correction_note(violations: list["Violation"]) -> str:
    """The hidden note for the corrective round, one paragraph per distinct rule."""
    seen: dict[str, str] = {}
    for v in violations:
        seen.setdefault(v.rule, _CORRECTIONS[v.rule].format(sentence=v.sentence))
    return "Служебное сообщение (клиент его не слышит). " + " ".join(seen.values())


def fallback_sentence(violations: list["Violation"]) -> str:
    """The neutral phrase for the first blocked rule."""
    return GUARD_FALLBACKS[violations[0].rule]


@dataclass
class SpeechGuard:
    """Per-call guard state: has anything been saved yet? Feed it the model's sentences."""

    committed: bool = False
    blocked: list[Violation] = field(default_factory=list)

    def note_commit(self) -> None:
        """A tool saved a booking or a message: acceptance claims are true from now on."""
        self.committed = True

    def check(self, sentence: str) -> Violation | None:
        """The first rule the sentence breaks, or None. Does not record or log."""
        if chars := foreign_script_chars(sentence):
            return Violation("foreign_script", sentence, f"characters {sorted(set(chars))}")
        if ROLE_LEAKAGE.search(sentence):
            return Violation("role_leakage", sentence, "a role label or the caller's words")
        if _is_phone_dictation(sentence):
            return Violation("phone_digits", sentence, "digits or number words of a phone number")
        if self.committed:
            return None
        if not sentence.strip().endswith("?") and ACCEPTANCE_CLAIM.search(sentence):
            return Violation("acceptance_claim", sentence, "nothing was saved in this call")
        if WRITTEN_DOWN.search(sentence):
            return Violation("written_down", sentence, "nothing was saved in this call")
        return None

    def screen(self, sentence: str) -> Violation | None:
        """`check`, and remember and log a violation."""
        violation = self.check(sentence)
        if violation is not None:
            self.blocked.append(violation)
            logger.warning("speech guard blocked (%s): %r", violation.rule, sentence)
        return violation


def _is_phone_dictation(sentence: str) -> bool:
    if _MONEY_CUE.search(sentence):
        return any(
            sum(ch.isdigit() for ch in group) >= PHONE_GROUP_DIGITS
            for group in _DIGIT_GROUP.findall(sentence)
        )
    return longest_number_run(sentence) > PHONE_RUN_LIMIT


# Unicode names starting with these are flagged even when they are not letters (CJK
# punctuation, full-width forms): they show up together with the letters when a model
# switches script.
_SUSPICIOUS_NAME_PREFIXES = ("CJK", "IDEOGRAPHIC", "FULLWIDTH", "HALFWIDTH")


def _script(char: str) -> str:
    return unicodedata.name(char, "").split(" ", 1)[0]


def foreign_script_chars(text: str) -> list[str]:
    """Characters that do not belong in Russian speech: letters that are neither Cyrillic nor
    Latin (CJK, Arabic, Greek, ...), and CJK/full-width symbols. Digits, punctuation, spaces,
    «», —, №, ₽ and the like are fine. Returns them in order of appearance (with repeats)."""
    found = []
    for char in text:
        name = unicodedata.name(char, "")
        if char.isalpha():
            if _script(char) not in ("CYRILLIC", "LATIN"):
                found.append(char)
        elif name.startswith(_SUSPICIOUS_NAME_PREFIXES):
            found.append(char)
    return found
