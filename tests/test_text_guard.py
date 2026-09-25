import pytest

from agent.text_guard import (
    GUARD_FALLBACKS,
    PHONE_RUN_LIMIT,
    SpeechGuard,
    Violation,
    correction_note,
    fallback_sentence,
    foreign_script_chars,
)


@pytest.mark.parametrize(
    "text",
    [
        "Полировка кузова стоит от пятнадцати до тридцати пяти тысяч рублей.",
        "Мы находимся в Москве, улица Примерная, дом один. Работаем с десяти до двадцати.",
        "Toyota Camry, BMW X5, PPF — оклейка «плёнкой» №1, 15 000 ₽ (примерно)!",
        "Ёлка, ёж; Ё.",
        "",
        "   \n\t",
        "2026-09-26, 14:30 - 20:00 / 100%",
    ],
)
def test_normal_russian_speech_has_no_foreign_characters(text):
    assert foreign_script_chars(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("от пятнадцати до тридцати五千 рублей", ["五", "千"]),
        ("Цена 三万 рублей", ["三", "万"]),
        ("Здравствуйте、чем помочь", ["、"]),  # CJK punctuation
        ("Мы работаем ежедневно，без выходных", ["，"]),  # full-width comma
        ("Привет مرحبا", list("مرحبا")),
        ("Привет こんにちは", list("こんにちは")),
        ("привет 안녕", list("안녕")),
        ("Сколько стоит α-полировка", ["α"]),
    ],
)
def test_characters_from_other_scripts_are_found_in_order(text, expected):
    assert foreign_script_chars(text) == expected


def test_repeats_are_reported_each_time():
    assert foreign_script_chars("五 и 五") == ["五", "五"]


def test_latin_and_cyrillic_mixed_are_fine():
    assert foreign_script_chars("Скажите model и марку: Kia Rio, Лада Веста") == []


# --- SpeechGuard: real violations and real legitimate sentences (from recorded calls) -------------

# Sentences the model actually wrote in the eval sweeps, and the rule that must stop each.
REAL_VIOLATIONS = [
    ("Хорошо, записал.", "written_down"),
    ("Хорошо, всё записал.", "written_down"),
    ("Записала номер.", "written_down"),
    ("Записал номер, который вы продиктовали.", "written_down"),
    ("Спасибо, записал.", "written_down"),
    (
        "Записал: восемь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь.",
        "phone_digits",
    ),
    ("Номер — 8 916 123 45 67.", "phone_digits"),
    ("Клиент уже назвал номер 8 916 123 45 67 — уточнять не нужно.", "phone_digits"),
    (
        "Небольшое уточнение: подскажите, пожалуйста, ваш номер ещё раз для заявки — или можно "
        "использовать тот, что вы назвали, восемь девятьсот шестнадцать сто двадцать три сорок "
        "пять шестьдесят семь.",
        "phone_digits",
    ),
    (
        "Оставляю заявку на полировку кузова, завтра, в четырнадцать ноль-ноль, на номер "
        "восемь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь.",
        "phone_digits",
    ),
    (
        "Уточните, пожалуйста, вас записать на номер, который вы продиктовали — восемь "
        "девятьсот шестнадцать, сто двадцать три, сорок пять, шестьдесят семь.",
        "phone_digits",
    ),
    ("Заявка принята, администратор перезвонит вам для уточнения деталей.", "acceptance_claim"),
    ("Ваша заявка передана администратору.", "acceptance_claim"),
    ("Сообщение передано администратору, он свяжется с вами.", "acceptance_claim"),
    ("Заявка уже оформлена.", "acceptance_claim"),
    ("Ваша просьба принята.", "acceptance_claim"),
    ("Оформлю заявку. 五千 рублей.", "foreign_script"),
]

# Sentences the model (or the code) legitimately says in real calls: none may be blocked.
REAL_LEGIT = [
    "Здравствуйте!",
    "Детейлинг-центр «Пример».",
    "Чем могу помочь?",
    "Как вас зовут?",
    "Подскажите, пожалуйста, ваше имя.",
    "Записать вас на номер, с которого вы звоните?",
    "Игорь, подскажите, пожалуйста, ваш телефон для связи — записать вас на номер, с которого вы звоните?",
    "Какая у вас марка и модель автомобиля?",
    "На какой день и время вам удобно записаться на полировку?",
    "Полировка кузова — от пятнадцати до тридцати пяти тысяч рублей, длительность от одного до двух дней.",
    "Детейлинг-мойка от трёх до пяти тысяч рублей.",
    "Керамика — от двадцати пяти тысяч рублей.",
    "Химчистка салона обойдётся от 10 000 до 18 000 ₽.",
    "Такой услуги в нашем списке нет — у нас есть оклейка защитной плёнкой, от двадцати тысяч рублей.",
    "Мы находимся в Москве, на улице Примерная, дом один.",
    "Въезд со двора, ворота с вывеской, для клиентов есть парковка у бокса.",
    "Мы работаем в субботу до восьми вечера.",
    "Работаем в будни с десяти до двадцати одного, в субботу с десяти до двадцати, в воскресенье выходной.",
    "Ближайшая суббота — это двадцать шестое сентября, работаем с десяти до двадцати.",
    "Суббота — это двадцать шестое сентября, время после обеда.",
    "В воскресенье мы не работаем.",
    "Может быть, подойдёт понедельник?",
    "Уточню: сегодня пятница, двадцать пятое сентября, значит ближайшая суббота — двадцать шестое.",
    "Проверьте, пожалуйста: Игорь, полировка кузова, автомобиль Тойота Камри, в субботу, двадцать шестого сентября, днём.",
    "Проверьте, пожалуйста: Игорь, керамическое покрытие, автомобиль Тойота Камри, в понедельник, двадцать восьмого сентября, в одиннадцать часов.",
    "Номер телефона заканчивается на четыре пять шесть семь.",
    "Всё верно?",
    "Извините, я ещё не отправил заявку.",
    "Точное время администратор уточнит при звонке.",
    "Могу ещё чем-то помочь?",
    "Я передам ваш вопрос администратору — он свяжется с вами.",
    "Хотите, я передам ему ваш вопрос?",
    "Если хотите, оставьте имя и телефон, и администратор вам перезвонит.",
    "Могу передать ваш вопрос администратору, он свяжется с вами и всё подробно расскажет.",
    "Уточню у администратора.",
    "Точную стоимость определит мастер после осмотра.",
    "К сожалению, точной информации о сроках гарантии у меня нет — этот вопрос лучше уточнить у администратора.",
    "Свободное время называет администратор, я его не вижу.",
    "Ваш номер телефона я не вижу, продиктуйте его, пожалуйста.",
    "Спасибо, Игорь.",
    "Хорошо.",
    "Принято.",
    "До свидания!",
    "Всего доброго.",
    "Приезжайте к десяти часам.",
    "Это займёт два-три дня.",
    "Полировка занимает от одного до двух дней.",
    "Оклейка от двадцати тысяч рублей, длительность от одного до четырёх дней.",
    "Мы принимаем наличные, банковские карты и переводы по счёту.",
    "Машину можно оставить на ночь в охраняемом тёплом боксе.",
    "Записать вас на полировку на субботу?",
    "Хотите, чтобы заявка была передана администратору?",
    "Заявка принята?",
    "Давайте оформим заявку.",
    "Оформляю заявку на полировку.",
    "Записываю вас на керамическое покрытие в понедельник, в одиннадцать часов.",
]


@pytest.mark.parametrize(("sentence", "rule"), REAL_VIOLATIONS)
def test_every_real_violation_from_the_sweeps_is_blocked_by_the_expected_rule(sentence, rule):
    violation = SpeechGuard().check(sentence)
    assert violation is not None and violation.rule == rule and violation.sentence == sentence


@pytest.mark.parametrize("sentence", REAL_LEGIT)
def test_no_legitimate_sentence_is_blocked(sentence):
    assert SpeechGuard().check(sentence) is None


def test_the_legit_corpus_stays_clean_after_a_save_too():
    guard = SpeechGuard()
    guard.note_commit()
    assert [s for s in REAL_LEGIT if guard.check(s)] == []


def test_acceptance_claims_and_written_down_are_allowed_once_something_was_saved():
    guard = SpeechGuard()
    guard.note_commit()

    assert guard.check("Заявка принята и передана администратору, он перезвонит.") is None
    assert guard.check("Сообщение передано администратору.") is None
    assert guard.check("Записал вас, ждите звонка.") is None


def test_phone_digits_and_foreign_script_are_blocked_even_after_a_save():
    guard = SpeechGuard()
    guard.note_commit()

    assert guard.check("Ваш номер 8 916 123 45 67.").rule == "phone_digits"
    assert guard.check("Заявка принята 五千").rule == "foreign_script"


def test_the_four_digit_read_back_is_exactly_at_the_allowed_limit():
    assert PHONE_RUN_LIMIT == 4
    assert SpeechGuard().check("Заканчивается на четыре пять шесть семь.") is None
    assert (
        SpeechGuard().check("Заканчивается на три четыре пять шесть семь.").rule == "phone_digits"
    )
    assert SpeechGuard().check("Заканчивается на 4567.") is None
    assert SpeechGuard().check("Заканчивается на 45678.").rule == "phone_digits"


@pytest.mark.parametrize(
    "sentence",
    [
        "Полировка стоит от 3 000 до 5 000 ₽.",
        "Это 15000 рублей.",
        "От двадцати пяти тысяч рублей.",
        "Дата: 26.09.2026, время 14:30.",
        "Приезжайте в две тысячи двадцать шестом году.",
    ],
)
def test_prices_dates_and_times_are_not_phone_numbers(sentence):
    assert SpeechGuard().check(sentence) is None


def test_a_long_digit_group_inside_a_money_sentence_is_still_a_phone():
    assert SpeechGuard().check("Стоит 500 рублей, звоните 89161234567.").rule == "phone_digits"


def test_a_question_about_acceptance_is_not_a_claim_but_the_word_written_down_still_is():
    guard = SpeechGuard()
    assert guard.check("Заявка принята?") is None
    assert guard.check("Записал?").rule == "written_down"


def test_rules_are_checked_in_the_documented_order():
    both = "Записал номер 8 916 123 45 67 五"
    assert SpeechGuard().check(both).rule == "foreign_script"
    assert SpeechGuard().check("Записал номер 8 916 123 45 67").rule == "phone_digits"


def test_screen_records_and_logs_but_check_does_not(caplog):
    guard = SpeechGuard()

    assert guard.check("Хорошо, записал.") is not None and guard.blocked == []
    with caplog.at_level("WARNING"):
        violation = guard.screen("Хорошо, записал.")

    assert guard.blocked == [violation]
    assert "speech guard blocked (written_down)" in caplog.text and "записал" in caplog.text
    assert guard.screen("Хорошо.") is None and len(guard.blocked) == 1


def test_a_new_guard_starts_with_nothing_saved():
    guard = SpeechGuard()
    assert guard.committed is False
    guard.note_commit()
    assert guard.committed is True
    assert SpeechGuard().committed is False  # per call, not global


# --- What is said or told to the model after a block ----------------------------------------------


def test_every_rule_has_a_fallback_and_a_correction():
    from agent.text_guard import _CORRECTIONS

    rules = {"foreign_script", "phone_digits", "acceptance_claim", "written_down"}
    assert set(GUARD_FALLBACKS) == rules == set(_CORRECTIONS)


def test_the_fallback_wording_is_the_agreed_one():
    assert GUARD_FALLBACKS == {
        "phone_digits": "Хорошо, номер есть.",
        "acceptance_claim": "Давайте ещё раз проверим данные заявки.",
        "written_down": "Хорошо.",
        "foreign_script": "Простите, уточните, пожалуйста, ваш вопрос.",
    }


def test_the_fallback_is_the_one_of_the_first_blocked_rule():
    violations = [Violation("phone_digits", "a"), Violation("written_down", "b")]
    assert fallback_sentence(violations) == "Хорошо, номер есть."


def test_the_correction_note_quotes_each_blocked_sentence_once_per_rule():
    note = correction_note(
        [
            Violation("acceptance_claim", "Заявка принята."),
            Violation("acceptance_claim", "Ваша заявка передана."),  # same rule: one paragraph
            Violation("phone_digits", "Номер 89161234567."),
        ]
    )

    assert note.startswith("Служебное сообщение (клиент его не слышит).")
    assert "«Заявка принята.»" in note and "«Номер 89161234567.»" in note
    assert "Ваша заявка передана" not in note
    assert "confirm_booking" in note and "prepare_booking" in note  # tells the model what to do
