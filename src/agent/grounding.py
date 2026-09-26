"""What the caller actually said during this call, to check data the model puts into tool calls.

The model can invent the caller's answers (role leakage: «userМеня зовут Дмитрий») and then call
a tool with a made-up name and phone. The speech guard cannot see that, so the tools ask this
module: did the caller really say this number / this name? Only the caller's own words count
(never the agent's), and the words are kept for the whole call.
"""

import re

from agent.ru_words import spoken_digit_runs

_WORD_RE = re.compile(r"[а-яёa-z]+")
MIN_NAME_PREFIX = 3


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("ё", "е"))


def _names_match(said: str, given: str) -> bool:
    """Same name in another case or shape: «Дмитрий» ~ «Дмитрию», «Ольга» ~ «Ольгой». They
    match if they share a prefix that is all of the shorter one except at most its last letter
    (at least three letters, or the whole word for a very short name). Diminutives («Дима» for
    «Дмитрий») do not match."""
    shortest = min(len(said), len(given))
    needed = min(max(MIN_NAME_PREFIX, shortest - 1), shortest)
    prefix = 0
    for a, b in zip(said, given, strict=False):
        if a != b:
            break
        prefix += 1
    return prefix >= needed


class CallerSpeech:
    """Everything the caller said in one call, as digits and as words."""

    def __init__(self) -> None:
        self._digits: list[str] = []  # the digits of each utterance
        self._words: set[str] = set()
        self._runs: list[str] = []  # runs of digits said in a row, and the pieces joined
        self._chain: str | None = None  # a run that touched the end of the last utterance

    def add(self, utterance: str) -> None:
        runs = spoken_digit_runs(utterance)
        if digits := "".join(run.digits for run in runs):
            self._digits.append(digits)
        self._words.update(_words(utterance))

        # A number dictated in pieces (the speech-to-text cuts at pauses) is a run that ends one
        # utterance and a run that starts the next. Both the pieces and the joined run are kept:
        # «в 14:00» followed by «8 916 123 45 67» must not swallow the number.
        continuing = self._chain is not None and bool(runs) and runs[0].at_start
        chain = None
        for index, run in enumerate(runs):
            self._runs.append(run.digits)
            if index == 0 and continuing:
                chain = f"{self._chain}{run.digits}"
                self._runs.append(chain)
            else:
                chain = run.digits
        self._chain = chain if runs and runs[-1].at_end else None

    def dictated_numbers(self) -> list[str]:
        """Runs of digits the caller said in a row, in the order they were said (a number
        dictated in pieces also appears joined). Not all of them are phone numbers."""
        return list(self._runs)

    def said_phone(self, phone: str) -> bool:
        """Do the digits of `phone` (any format, +7/8 prefix ignored) occur in what the caller
        said? Digits are compared inside one utterance and across all of them joined, because
        callers dictate a number in pieces. A `phone` without digits is not checked."""
        digits = re.sub(r"\D", "", phone)
        if len(digits) == 11 and digits[0] in "78":
            digits = digits[1:]
        if not digits:
            return True
        return any(digits in d for d in self._digits) or digits in "".join(self._digits)

    def said_name(self, name: str) -> bool:
        """Did the caller say any word of `name` (in any case form)?"""
        given = _words(name)
        if not given:
            return True
        return any(_names_match(said, word) for word in given for said in self._words)
