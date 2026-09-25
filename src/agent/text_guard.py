"""Checks for text that is about to be spoken by TTS.

Small models (Qwen in particular) sometimes slip characters from another script into a Russian
sentence («тридцати五千 рублей»); a TTS voice would read that out as garbage or in another
language. `foreign_script_chars` finds them; what to do about a hit (regenerate, drop the
sentence) is decided where the audio pipeline is built (step 2).
"""

import unicodedata

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
