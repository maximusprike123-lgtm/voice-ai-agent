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


# --- dictated_numbers ---------------------------------------------------------------------


def test_a_number_in_one_utterance_is_one_run():
    assert heard("Запишите на 8 903 555 12 34").dictated_numbers() == ["89035551234"]


def test_words_around_the_number_do_not_matter():
    speech = heard("Меня зовут Игорь, Камри 2015 года, номер 8 903 555 12 34, на 14:00")
    assert "89035551234" in speech.dictated_numbers()


def test_pieces_at_the_edges_of_neighbouring_utterances_are_also_kept_joined():
    numbers = heard("Мой номер 8 903", "555 12", "34 спасибо").dictated_numbers()

    assert "8903" in numbers and "55512" in numbers and "890355512" in numbers
    assert "8903555123" not in numbers  # the last piece is joined whole, not cut


def test_the_whole_number_appears_joined_after_the_last_piece():
    speech = heard("Мой номер 8 903", "555 12", "34")
    assert speech.dictated_numbers()[-1] == "89035551234"


def test_a_time_at_the_end_does_not_swallow_a_number_at_the_start_of_the_next_utterance():
    speech = heard("Завтра на 14:00", "8 903 555 12 34")
    numbers = speech.dictated_numbers()
    assert "89035551234" in numbers  # the piece on its own
    assert "140089035551234" in numbers  # and the (useless) joined run


def test_an_utterance_without_digits_breaks_the_chain():
    speech = heard("Мой номер 8 903", "да", "555 12 34")
    assert all(len(n) < 10 for n in speech.dictated_numbers())


def test_a_word_before_the_first_run_breaks_the_chain():
    speech = heard("Мой номер 8 903", "и ещё 555 12 34")
    assert all(len(n) < 10 for n in speech.dictated_numbers())


def test_nothing_dictated():
    assert heard("Здравствуйте").dictated_numbers() == []
