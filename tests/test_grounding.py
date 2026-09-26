"""Offline tests for CallerSpeech: did the caller really say this number / this name?"""

import pytest

from agent.grounding import CallerSpeech


def heard(*utterances: str) -> CallerSpeech:
    speech = CallerSpeech()
    for utterance in utterances:
        speech.add(utterance)
    return speech


@pytest.mark.parametrize(
    "phone", ["+79161234567", "89161234567", "9161234567", "8 (916) 123-45-67"]
)
def test_a_dictated_number_is_found_whatever_the_format(phone):
    assert heard("Запишите на 8 916 123 45 67").said_phone(phone)


def test_a_number_dictated_in_words_is_found():
    speech = heard("восемь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь")
    assert speech.said_phone("+79161234567")


def test_a_number_dictated_in_pieces_across_utterances_is_found():
    speech = heard(
        "Мой номер девятьсот шестнадцать", "сто двадцать три", "сорок пять шестьдесят семь"
    )
    assert speech.said_phone("+79161234567")


def test_a_number_the_caller_never_said_is_not_found():
    speech = heard("Меня зовут Игорь", "Запишите на завтра на 14:00")
    assert not speech.said_phone("+79161234567")


def test_one_wrong_digit_is_not_found():
    assert not heard("8 916 123 45 67").said_phone("+79161234568")


def test_nothing_said_at_all_finds_nothing():
    assert not CallerSpeech().said_phone("+79161234567")


def test_a_phone_without_digits_is_not_checked():
    assert CallerSpeech().said_phone("как-нибудь")


@pytest.mark.parametrize(
    ("said", "given"),
    [
        ("Меня зовут Игорь", "Игорь"),
        ("меня зовут игорь", "Игорь"),
        ("Зовите меня Дмитрию", "Дмитрий"),  # another case form
        ("Ольгой зовут", "Ольга"),
        ("Артём", "Артем"),  # ё / е
        ("Игорь Петров", "Петров Игорь"),
        ("Я Анна", "Анна Сергеевна"),  # any word of the name will do
        ("Ян", "Ян"),
        ("Ян", "Яна"),
    ],
)
def test_the_name_was_said(said, given):
    assert heard(said).said_name(given)


@pytest.mark.parametrize(
    ("said", "given"),
    [
        ("Меня зовут Игорь", "Дмитрий"),
        ("Дима", "Дмитрий"),  # diminutives are not matched
        ("Привет", "Пётр"),
        ("Ян", "Яков"),
    ],
)
def test_the_name_was_not_said(said, given):
    assert not heard(said).said_name(given)


def test_a_name_without_letters_is_not_checked():
    assert CallerSpeech().said_name("-")
