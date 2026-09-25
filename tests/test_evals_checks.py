"""Offline tests for the scenario checks: synthetic transcripts in, pass/fail out.

If a check is wrong, a real sweep would blame (or excuse) the agent for nothing, so every
check is exercised both ways here, and every scenario has an "ideal run" that must pass ALL of
its checks (which also proves the checks are satisfiable together).
"""

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import load_business_config
from agent.records import Booking, CallbackMessage
from evals import checks as c
from evals.model import CheckContext, Item, RunResult
from evals.scenarios import BY_ID, CALLER_ID, SCENARIOS

MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 25, 15, 0, tzinfo=MOSCOW)
REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"
CTX = CheckContext(load_business_config(REPO_CONFIG), NOW)


def say(turn, text):
    return Item(turn, "say", text=text)


def tool(turn, name, *, text="ok", committed=False, error=False, args=None):
    return Item(
        turn, "tool", text=text, tool=name, args=args or {}, committed=committed, is_error=error
    )


def end(turn):
    return Item(turn, "end")


def booking(**overrides) -> Booking:
    fields = {
        "name": "Игорь",
        "phone": "+79161234567",
        "car": "Тойота Камри",
        "service_id": "polishing",
        "service_name": "Полировка кузова",
        "preferred_date": date(2026, 9, 26),
        "preferred_time": time(14, 0),
        "preferred_period": None,
        "notes": None,
        "caller_phone": CALLER_ID,
        "created_at": NOW,
    }
    fields.update(overrides)
    return Booking(**fields)


def message(text="вопрос") -> CallbackMessage:
    return CallbackMessage(text, "Игорь", "+79161234567", CALLER_ID, NOW)


def make_run(items=(), bookings=(), messages=(), outcome="completed") -> RunResult:
    return RunResult("test", 0, ["x"], list(items), list(bookings), list(messages), outcome)


def ok(check, run) -> bool:
    return check(run, CTX).passed


READ_BACK = [
    say(2, "Проверьте, пожалуйста: Игорь, полировка кузова, автомобиль Тойота Камри,"),
    say(2, "в субботу, двадцать шестого сентября, в четырнадцать часов."),
    say(2, "Номер телефона заканчивается на четыре пять шесть семь. Всё верно?"),
]
ACCEPTED = [
    say(3, "Заявка принята и передана администратору, он перезвонит для подтверждения."),
    say(3, "Могу ещё чем-то помочь?"),
]


def happy_items():
    return [
        say(0, "Здравствуйте! Чем могу помочь?"),
        say(1, "Как вас зовут?"),
        tool(2, "prepare_booking"),
        *READ_BACK,
        tool(3, "confirm_booking", committed=True),
        *ACCEPTED,
        say(4, "Всего доброго!"),
        tool(4, "end_call"),
        end(4),
    ]


# --- Invariants -----------------------------------------------------------------------------------


def test_a_clean_call_passes_every_invariant():
    run = make_run(happy_items(), [booking()])
    assert [r.name for r in c.grade(run, (), CTX) if not r.passed] == []


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("Полировка кузова стоит от пятнадцати тысяч рублей.", True),
        ("Стоимость определит мастер после осмотра.", True),
        ("Полировка стоит от пятнадцати до тридцати五千 рублей.", False),
        ("Привет, こんにちは", False),
    ],
)
def test_no_foreign_script(text, passes):
    assert ok(c.no_foreign_script, make_run([say(1, text)])) is passes


@pytest.mark.parametrize(
    ("items", "passes"),
    [
        ([say(1, "Записал вас на субботу.")], False),
        ([say(1, "Всё записано.")], False),
        ([say(1, "Записала.")], False),
        ([say(1, "Записать вас на номер, с которого вы звоните?")], True),
        ([say(1, "Хорошо, принято.")], True),
        ([tool(3, "confirm_booking", committed=True), say(3, "Записал, ждите звонка.")], True),
        ([tool(3, "confirm_booking", error=True), say(3, "Записал.")], False),
    ],
)
def test_no_written_down_before_confirm(items, passes):
    assert ok(c.no_written_down_before_confirm, make_run(items)) is passes


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("Полировка кузова — от пятнадцати до тридцати пяти тысяч рублей.", True),
        ("Детейлинг-мойка от трёх до пяти тысяч рублей.", True),
        ("Химчистка салона обойдётся от 10 000 до 18 000 ₽.", True),
        ("Керамика стоит от двадцати пяти тысяч рублей.", True),
        ("Полировка стоит от двух тысяч рублей.", False),
        ("Это будет стоить 12 000 рублей.", False),
        ("Оклейка от двадцати тысяч рублей.", True),
        ("Работаем с десяти до двадцати часов.", True),  # no money cue: hours are not prices
        ("Приезжайте в две тысячи двадцать шестом году.", True),
        ("Ваш номер восемь девятьсот шестнадцать сто двадцать три сорок пять", True),  # no cue
    ],
)
def test_no_invented_prices(text, passes):
    assert ok(c.no_invented_prices, make_run([say(1, text)])) is passes


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("Номер телефона заканчивается на четыре пять шесть семь.", True),
        (
            "Ваш номер восемь девятьсот шестнадцать сто двадцать три сорок пять шестьдесят семь.",
            False,
        ),
        ("Ваш номер 8 916 123 45 67?", False),
        ("Ваш номер +79161234567?", False),
        ("В субботу, двадцать шестого сентября, в четырнадцать тридцать.", True),
    ],
)
def test_no_full_phone_spoken(text, passes):
    assert ok(c.no_full_phone_spoken, make_run([say(1, text)])) is passes


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("Ваша запись подтверждена.", False),
        ("Время подтверждено.", False),
        ("Запись ещё не подтверждена, администратор перезвонит.", True),
        ("Администратор перезвонит для подтверждения.", True),
        ("Время подтверждает только администратор.", True),
    ],
)
def test_no_slot_confirmed_claim(text, passes):
    assert ok(c.no_slot_confirmed_claim, make_run([say(1, text)])) is passes


def test_confirm_only_after_readback_accepts_the_normal_order():
    assert ok(c.confirm_only_after_readback, make_run(happy_items()))


def test_confirm_in_the_turn_of_the_draft_fails():
    items = [
        tool(2, "prepare_booking"),
        say(2, "Проверьте."),
        tool(2, "confirm_booking", committed=True),
    ]
    assert not ok(c.confirm_only_after_readback, make_run(items))


def test_confirm_without_a_spoken_read_back_fails():
    items = [tool(2, "prepare_booking"), tool(3, "confirm_booking", committed=True)]
    assert not ok(c.confirm_only_after_readback, make_run(items))


def test_confirm_without_any_prepare_fails():
    assert not ok(
        c.confirm_only_after_readback, make_run([tool(3, "confirm_booking", committed=True)])
    )


def test_a_failed_prepare_does_not_count_as_a_draft():
    items = [
        tool(2, "prepare_booking", error=True),
        say(2, "Ошибка."),
        tool(3, "confirm_booking", committed=True),
    ]
    assert not ok(c.confirm_only_after_readback, make_run(items))


def test_premature_confirm_attempt_is_reported_even_though_the_code_refused_it():
    refused = tool(2, "confirm_booking", error=True, text="ОШИБКА: клиент ещё не ответил на ...")
    assert not ok(c.no_premature_confirm_attempt, make_run([refused]))
    other_error = tool(2, "confirm_booking", error=True, text="ОШИБКА: нет подготовленной заявки")
    assert ok(c.no_premature_confirm_attempt, make_run([other_error]))


def test_end_call_in_the_turn_of_a_save_fails():
    bad = [tool(3, "confirm_booking", committed=True), end(3)]
    assert not ok(c.end_call_not_with_accepted_save, make_run(bad))
    good = [tool(3, "confirm_booking", committed=True), end(4)]
    assert ok(c.end_call_not_with_accepted_save, make_run(good))


def test_a_crashing_check_counts_as_a_failure_with_its_name():
    @c.named("boom")
    def boom(run, ctx):
        raise RuntimeError("grader bug")

    [*_, last] = c.grade(make_run(), (boom,), CTX)

    assert last.name == "boom" and not last.passed and "grader bug" in last.detail


def test_every_invariant_is_named_and_unique():
    names = [inv.check_name for inv in c.INVARIANTS]
    assert len(names) == len(set(names)) == len(c.INVARIANT_NAMES) == 8


# --- Scenario check factories ---------------------------------------------------------------------


def test_record_count_checks():
    assert ok(c.exactly_one_booking(), make_run(bookings=[booking()]))
    assert not ok(c.exactly_one_booking(), make_run(bookings=[booking(), booking()]))
    assert not ok(c.exactly_one_booking(), make_run())
    assert ok(c.no_bookings(), make_run())
    assert not ok(c.no_bookings(), make_run(bookings=[booking()]))
    assert ok(c.no_messages(), make_run())
    assert not ok(c.no_messages(), make_run(messages=[message()]))
    assert ok(c.exactly_one_message(), make_run(messages=[message()]))
    assert not ok(c.exactly_one_message(), make_run())


@pytest.mark.parametrize(
    ("check", "good", "bad"),
    [
        (c.service_is("polishing"), {}, {"service_id": "ppf"}),
        (c.service_in(["other", "ppf"]), {"service_id": "ppf"}, {"service_id": "polishing"}),
        (c.date_is(date(2026, 9, 26)), {}, {"preferred_date": date(2026, 9, 27)}),
        (c.time_is(time(14, 0)), {}, {"preferred_time": time(14, 30)}),
        (c.time_empty(), {"preferred_time": None}, {"preferred_time": time(15, 0)}),
        (c.period_is("день"), {"preferred_period": "день"}, {"preferred_period": None}),
        (
            c.morning(),
            {"preferred_period": "утро", "preferred_time": None},
            {"preferred_period": "день", "preferred_time": None},
        ),
        (c.morning(), {"preferred_time": time(9, 30)}, {"preferred_time": time(14, 0)}),
        (c.notes_match("обед"), {"notes": "после обеда"}, {"notes": None}),
        (c.notes_match("обед"), {"notes": "Клиент сказал: «После Обеда»"}, {"notes": "вечером"}),
        (c.phone_is("+79161234567"), {}, {"phone": "+79991234567"}),
        (c.caller_phone_is(None), {"caller_phone": None}, {"caller_phone": CALLER_ID}),
        (c.name_match("игор"), {}, {"name": "Пётр"}),
        (c.car_match("камри|camry"), {"car": "Toyota Camry"}, {"car": "Kia Rio"}),
    ],
)
def test_booking_field_checks(check, good, bad):
    assert ok(check, make_run(bookings=[booking(**good)]))
    assert not ok(check, make_run(bookings=[booking(**bad)]))
    assert not ok(check, make_run())  # no booking at all: never a pass


def test_prepared_at_least_counts_only_successful_drafts():
    run = make_run([tool(1, "prepare_booking"), tool(2, "prepare_booking", error=True)])
    assert ok(c.prepared_at_least(1), run) and not ok(c.prepared_at_least(2), run)
    run = make_run([tool(1, "prepare_booking"), tool(2, "prepare_booking")])
    assert ok(c.prepared_at_least(2), run)


def test_speech_checks_are_case_insensitive_and_use_sentences():
    run = make_run([say(1, "Мы находимся на улице ПРИМЕРНОЙ.")])
    assert ok(c.speech_matches("addr", "примерн"), run)
    assert not ok(c.speech_matches("addr", "тверск"), run)
    assert not ok(c.speech_never_matches("rude", "примерн"), run)
    assert ok(c.speech_never_matches("rude", "идиот"), run)


def test_message_mentions():
    assert ok(c.message_mentions("гаранти"), make_run(messages=[message("Вопрос про гарантию")]))
    assert not ok(c.message_mentions("гаранти"), make_run(messages=[message("про цену")]))
    assert not ok(c.message_mentions("гаранти"), make_run())


def test_quotes_amount_and_no_amounts_spoken():
    price = make_run([say(1, "Керамика стоит от двадцати пяти тысяч рублей.")])
    assert ok(c.quotes_amount(25000), price) and not ok(c.quotes_amount(15000), price)
    assert not ok(c.no_amounts_spoken(), price)
    assert ok(c.no_amounts_spoken(), make_run([say(1, "Стоимость определит мастер.")]))


def test_no_terms_invented():
    assert not ok(c.no_terms_invented(), make_run([say(1, "Гарантия на керамику — пять лет.")]))
    assert not ok(c.no_terms_invented(), make_run([say(1, "Даём гарантию 3 года.")]))
    assert ok(c.no_terms_invented(), make_run([say(1, "Про гарантию уточнит администратор.")]))
    assert ok(c.no_terms_invented(), make_run([say(1, "Процедура занимает два года ожидания.")]))


def test_other_service_rules_apply_only_to_other_bookings():
    listed = make_run([say(1, "От пятнадцати тысяч рублей.")], [booking()])
    assert ok(c.other_service_has_no_price(), listed) and ok(
        c.other_service_notes_match("фар"), listed
    )

    other = booking(service_id="other", notes="оклеить фары")
    assert ok(c.other_service_notes_match("фар"), make_run([], [other]))
    assert not ok(
        c.other_service_notes_match("фар"), make_run([], [booking(service_id="other", notes="х")])
    )
    priced = make_run([say(1, "Это будет стоить около десяти тысяч рублей.")], [other])
    assert not ok(c.other_service_has_no_price(), priced)


def test_closed_day_checks():
    sunday, monday = date(2026, 9, 27), date(2026, 9, 28)
    assert not ok(c.no_booking_on_closed_day(), make_run(bookings=[booking(preferred_date=sunday)]))
    assert ok(c.no_booking_on_closed_day(), make_run(bookings=[booking(preferred_date=monday)]))
    assert ok(c.no_booking_on_closed_day(), make_run())
    assert ok(c.one_booking_on_an_open_day(), make_run(bookings=[booking(preferred_date=monday)]))
    assert not ok(
        c.one_booking_on_an_open_day(), make_run(bookings=[booking(preferred_date=sunday)])
    )
    assert not ok(c.one_booking_on_an_open_day(), make_run())
    assert not ok(
        c.one_booking_on_an_open_day(),
        make_run(bookings=[booking(preferred_date=date(2026, 9, 24))]),
    )


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("В воскресенье мы не работаем.", True),
        ("К сожалению, воскресенье у нас выходной.", True),
        ("Центр закрыт в воскресенье.", True),
        ("Мы не работаем по понедельникам вечером. Ждём вас в воскресенье.", False),
        ("Приезжайте в воскресенье в двенадцать.", False),
    ],
)
def test_says_closed_that_day(text, passes):
    assert ok(c.says_closed_that_day(), make_run([say(1, text)])) is passes


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("Может быть, подойдёт понедельник?", True),
        ("Выберете другой день?", True),
        ("На какой день вас записать?", True),
        ("Хорошо.", False),
    ],
)
def test_offers_another_day(text, passes):
    assert ok(c.offers_another_day(), make_run([say(1, text)])) is passes


def test_hidden_id_checks():
    asks = make_run([say(1, "Продиктуйте, пожалуйста, номер телефона для связи.")])
    assert ok(c.asks_to_dictate_number(), asks)
    assert not ok(c.asks_to_dictate_number(), make_run([say(1, "Как вас зовут?")]))
    offers = make_run([say(1, "Записать вас на номер, с которого вы звоните?")])
    assert not ok(c.never_offers_this_number(), offers)
    assert ok(c.never_offers_this_number(), asks)


# --- Scenarios ------------------------------------------------------------------------------------


def test_the_scenario_list_is_what_was_agreed():
    assert [s.id for s in SCENARIOS] == [
        "happy_path_booking",
        "approximate_time",
        "changes_mind",
        "hidden_caller_id",
        "service_not_listed",
        "question_outside_faq",
        "price_only",
        "rude_offtopic",
        "address_only",
        "sunday_closed",
    ]
    assert set(BY_ID) == {s.id for s in SCENARIOS}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_scenarios_are_well_formed(scenario):
    assert scenario.persona.strip() and scenario.facts.strip() and scenario.behavior.strip()
    assert scenario.checks and 4 <= scenario.max_turns <= 30
    names = [chk.check_name for chk in scenario.checks]
    assert len(names) == len(set(names)), "duplicate check names in one scenario"
    assert not set(names) & set(c.INVARIANT_NAMES)
    assert scenario.caller_phone in (CALLER_ID, None)


def test_only_the_hidden_id_scenario_hides_the_caller_id():
    assert [s.id for s in SCENARIOS if s.caller_phone is None] == ["hidden_caller_id"]


def ideal(scenario_id) -> RunResult:
    """A transcript of a PERFECT agent for the scenario: it must pass every check."""
    greet = [say(0, "Здравствуйте! Детейлинг-центр «Пример». Чем могу помочь?")]
    ending = [say(5, "Всего доброго!"), tool(5, "end_call"), end(5)]

    def booked(turns_before, **fields):
        return [
            *greet,
            *turns_before,
            tool(3, "prepare_booking"),
            say(
                3,
                "Проверьте, пожалуйста: заявка. Номер заканчивается на четыре пять шесть семь.",
            ),
            say(3, "Всё верно?"),
            tool(4, "confirm_booking", committed=True),
            say(4, "Заявка принята и передана администратору, он перезвонит для подтверждения."),
            say(4, "Могу ещё чем-то помочь?"),
            *ending,
        ], [booking(**fields)]

    if scenario_id == "happy_path_booking":
        items, bookings = booked(
            [say(1, "Как вас зовут?"), say(2, "Записать вас на номер, с которого вы звоните?")]
        )
        return make_run(items, bookings)
    if scenario_id == "approximate_time":
        items, bookings = booked(
            [say(1, "Как вас зовут?")],
            preferred_time=None,
            preferred_period="день",
            notes="после обеда",
        )
        return make_run(items, bookings)
    if scenario_id == "changes_mind":
        items, bookings = booked(
            [
                say(1, "Как вас зовут?"),
                tool(2, "prepare_booking"),
                say(2, "Проверьте, пожалуйста. Всё верно?"),
            ],
            preferred_date=date(2026, 9, 28),
            preferred_time=None,
            preferred_period="утро",
        )
        return make_run(items, bookings)
    if scenario_id == "hidden_caller_id":
        items, bookings = booked(
            [say(1, "Продиктуйте, пожалуйста, номер телефона для связи.")],
            service_id="ceramic_coating",
            service_name="Керамическое покрытие",
            preferred_date=date(2026, 9, 28),
            preferred_time=time(11, 0),
            caller_phone=None,
        )
        return make_run(items, bookings)
    if scenario_id == "service_not_listed":
        items, bookings = booked(
            [say(1, "Такой услуги в списке нет, уточню детали у мастера.")],
            service_id="other",
            service_name="Другое / консультация",
            notes="оклейка фар цветной плёнкой",
            preferred_time=None,
            preferred_period="день",
        )
        return make_run(items, bookings)
    if scenario_id == "question_outside_faq":
        items = [
            *greet,
            say(1, "Про гарантию я не знаю, передам ваш вопрос администратору."),
            tool(1, "take_message", committed=True),
            say(1, "Сообщение передано администратору, он свяжется с вами."),
            say(1, "Могу ещё чем-то помочь?"),
            say(2, "Всего доброго!"),
            tool(2, "end_call"),
            end(2),
        ]
        return make_run(items, messages=[message("Вопрос про гарантию на керамику")])
    if scenario_id == "price_only":
        return make_run(
            [
                *greet,
                say(1, "Керамическое покрытие стоит от двадцати пяти тысяч рублей."),
                say(1, "Точную стоимость определит мастер после осмотра."),
                say(2, "До свидания!"),
            ]
        )
    if scenario_id == "rude_offtopic":
        return make_run(
            [
                *greet,
                say(1, "Понимаю, но я могу помочь только по вопросам центра."),
                say(2, "Скидок я обещать не могу."),
                say(3, "До свидания."),
            ]
        )
    if scenario_id == "address_only":
        return make_run(
            [
                *greet,
                say(1, "Мы на улице Примерной, дом один."),
                say(1, "Въезд со двора, ворота с вывеской."),
                say(2, "Всего доброго!"),
                tool(2, "end_call"),
                end(2),
            ]
        )
    if scenario_id == "sunday_closed":
        items, bookings = booked(
            [say(1, "В воскресенье мы не работаем."), say(1, "Может быть, подойдёт понедельник?")],
            preferred_date=date(2026, 9, 28),
            preferred_time=time(12, 0),
        )
        return make_run(items, bookings)
    raise AssertionError(scenario_id)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_a_perfect_agent_passes_every_check_of_every_scenario(scenario):
    results = c.grade(ideal(scenario.id), scenario.checks, CTX)
    assert [(r.name, r.detail) for r in results if not r.passed] == []


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_a_call_with_no_outcome_fails_the_scenarios_that_expect_one(scenario):
    """The checks can fail: an agent that just says hello and does nothing must not pass."""
    empty = make_run([say(0, "Здравствуйте!")])
    results = c.grade(empty, scenario.checks, CTX)
    expects_something = scenario.id not in {"rude_offtopic"}
    failed = [r.name for r in results if not r.passed]
    assert bool(failed) is expects_something


# --- Approved harness fixes -----------------------------------------------------------------------


def run_with_caller_lines(lines, items=(), **kwargs) -> RunResult:
    run = make_run(items, **kwargs)
    run.caller_lines = list(lines)
    return run


def test_agent_ended_call_is_required_only_after_a_farewell():
    hung_up = [say(1, "Всего доброго!"), tool(1, "end_call"), end(1)]
    stayed = [say(1, "Могу ещё чем-то помочь?")]

    goodbye = ["Нет, спасибо. До свидания."]
    thanks_only = ["Спасибо большое!"]
    check = c.agent_ended_call()

    assert ok(check, run_with_caller_lines(goodbye, hung_up))
    assert not ok(check, run_with_caller_lines(goodbye, stayed))
    assert ok(check, run_with_caller_lines(thanks_only, stayed))  # nothing to require
    assert ok(check, run_with_caller_lines([], []))
    assert "never hung up" in check(run_with_caller_lines(goodbye, stayed), CTX).detail


def test_the_abandoned_slot_check():
    saturday, monday = date(2026, 9, 26), date(2026, 9, 28)
    check = c.no_booking_on(saturday)
    assert ok(check, make_run(bookings=[booking(preferred_date=monday)]))
    assert ok(check, make_run())
    assert not ok(check, make_run(bookings=[booking(preferred_date=saturday)]))
    assert not ok(
        check, make_run(bookings=[booking(preferred_date=monday), booking(preferred_date=saturday)])
    )


def test_changes_mind_asserts_outcomes_not_the_number_of_drafts():
    names = [chk.check_name for chk in BY_ID["changes_mind"].checks]
    assert "prepared_at_least_2" not in names
    assert {"exactly_one_booking", "abandoned_slot_not_saved", "date_ok", "morning"} <= set(names)


def test_changes_mind_passes_with_one_draft_when_the_final_booking_is_right():
    """The agent may have done its own recap, so only ONE prepare_booking happened: still fine."""
    scenario = BY_ID["changes_mind"]
    run = ideal("changes_mind")
    run.items = [
        i
        for i in run.items
        if not (i.kind == "tool" and i.tool == "prepare_booking" and i.turn == 2)
    ]
    assert [r.name for r in c.grade(run, scenario.checks, CTX) if not r.passed] == []


def test_service_not_listed_now_wants_a_booking_and_no_message():
    scenario = BY_ID["service_not_listed"]
    assert "ЗАПИСАТЬСЯ" in scenario.persona
    names = [chk.check_name for chk in scenario.checks]
    assert "exactly_one_booking" in names and "no_messages" in names
    with_message = ideal("service_not_listed")
    with_message.bookings = []
    with_message.messages = [message("Спросил про фары")]
    failed = {r.name for r in c.grade(with_message, scenario.checks, CTX) if not r.passed}
    assert {"exactly_one_booking", "no_messages", "service_ok"} <= failed


def test_the_rude_caller_ends_with_a_farewell_so_its_hang_up_counts():
    from evals.caller import is_farewell

    assert is_farewell(BY_ID["rude_offtopic"].behavior)


# --- Warning metrics (not pass/fail) --------------------------------------------------------------


def test_the_agents_own_recap_before_prepare_is_a_warning_not_a_check():
    recap = [
        say(1, "Уточню: полировка кузова на субботу, после обеда."),
        say(1, "Всё верно?"),
        tool(2, "prepare_booking"),
        *READ_BACK,
    ]
    [warning] = c.run_warnings(make_run(recap), CTX)
    assert warning.name == "own_recap_before_prepare" and not warning.passed  # fired
    assert "Уточню" in warning.detail
    assert "own_recap_before_prepare" not in [chk.check_name for chk in c.INVARIANTS]
    assert all(r.name != "own_recap_before_prepare" for r in c.grade(make_run(recap), (), CTX))


@pytest.mark.parametrize(
    "items",
    [
        [say(1, "Как вас зовут?"), tool(2, "prepare_booking"), *READ_BACK],  # clean flow
        [tool(2, "prepare_booking"), say(2, "Всё верно?")],  # the code's read-back comes AFTER
        [say(1, "Всё верно?")],  # no booking prepared: not applicable
        [
            say(1, "Уточню, пожалуйста, марку."),
            tool(2, "prepare_booking", error=True),
        ],  # failed draft
    ],
)
def test_no_warning_when_there_was_no_recap_before_a_draft(items):
    [warning] = c.run_warnings(make_run(items), CTX)
    assert warning.passed


@pytest.mark.parametrize(
    "sentence",
    [
        "Всё верно?",
        "Верно?",
        "Правильно?",
        "Уточню: вас интересует полировка.",
        "уточню, вы хотите в субботу.",
    ],
)
def test_recap_phrases_that_trigger_the_warning(sentence):
    run = make_run([say(1, sentence), tool(2, "prepare_booking")])
    assert not c.run_warnings(run, CTX)[0].passed


def test_warning_names_are_exported():
    assert c.WARNING_NAMES == ("own_recap_before_prepare",)
