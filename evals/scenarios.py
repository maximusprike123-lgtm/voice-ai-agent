"""The scenarios. Each one is a persona + goal for the simulated caller, and the outcome checks
that decide whether the AGENT did its job. The clock is fixed: Friday 2026-09-25 15:00 in the
business time zone, so «завтра» is Saturday the 26th (open 10-20), Sunday the 27th is closed,
Monday the 28th is open (10-21).

Caller phone numbers: CALLER_ID is what the agent sees as the caller ID; the callers below
usually give a DIFFERENT number to book on (PHONE_TO_GIVE), which is also what should be saved.
"""

from datetime import date, time

from evals import checks as c
from evals.model import Scenario

CALLER_ID = "+79991234567"
PHONE_TO_GIVE = "8 916 123 45 67"
PHONE_SAVED = "+79161234567"
SATURDAY = date(2026, 9, 26)
MONDAY = date(2026, 9, 28)

COMMON_FACTS = (
    "Тебя зовут Игорь. Твоя машина — Тойота Камри. "
    f"Телефон для связи: {PHONE_TO_GIVE} (это НЕ номер, с которого ты звонишь)."
)
NUMBER_RULE = (
    "Если администратор предложит записать тебя на номер, с которого ты звонишь, откажись и "
    f"назови свой номер: {PHONE_TO_GIVE}."
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="happy_path_booking",
        description="Books polishing for tomorrow at an exact time, with a different phone.",
        persona="Ты хочешь записаться на полировку кузова на завтра, на 14:00.",
        facts=COMMON_FACTS,
        behavior=(
            f"{NUMBER_RULE} Когда администратор зачитает данные заявки, подтверди, если всё "
            "верно. Когда спросят, нужна ли ещё помощь, скажи, что нет, и попрощайся."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.exactly_one_booking(),
            c.no_messages(),
            c.service_is("polishing"),
            c.date_is(SATURDAY),
            c.time_is(time(14, 0)),
            c.phone_is(PHONE_SAVED),
            c.caller_phone_is(CALLER_ID),
            c.name_match(r"игор"),
            c.car_match(r"камри|camry"),
            c.prepared_at_least(1),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="approximate_time",
        description="Books for «в субботу после обеда»: no exact time may be invented.",
        persona="Ты хочешь записаться на полировку кузова в ближайшую субботу, после обеда.",
        facts=COMMON_FACTS
        + " Точное время ты не называешь: тебе подходит любое время после обеда.",
        behavior=(
            f"{NUMBER_RULE} Если тебя просят назвать точное время, ответь, что точное время "
            "не важно, главное — после обеда. Когда зачитают данные заявки, подтверди. "
            "На вопрос, нужна ли ещё помощь, скажи, что нет, и попрощайся."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.exactly_one_booking(),
            c.no_messages(),
            c.service_is("polishing"),
            c.date_is(SATURDAY),
            c.time_empty(),
            c.period_is("день"),
            c.notes_match(r"обед"),
            c.phone_is(PHONE_SAVED),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="changes_mind",
        description="Changes the day when the details are first read back: only the final "
        "choice may be saved.",
        persona=(
            "Ты хочешь записаться на полировку кузова. Сначала ты говоришь, что хочешь в ближайшую "
            "субботу, после обеда."
        ),
        facts=COMMON_FACTS,
        behavior=(
            f"{NUMBER_RULE} Когда администратор в ПЕРВЫЙ раз перечислит или зачитает данные "
            "заявки и спросит, всё ли верно, скажи, что передумал: лучше в понедельник, утром. "
            "Другие данные не меняй. Когда после этого зачитают исправленную заявку, подтверди. "
            "На вопрос, нужна ли ещё помощь, скажи, что нет, и попрощайся."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.exactly_one_booking(),
            c.no_booking_on(SATURDAY),
            c.no_messages(),
            c.service_is("polishing"),
            c.date_is(MONDAY),
            c.morning(),
            c.phone_is(PHONE_SAVED),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="hidden_caller_id",
        description="Calls from a hidden number: the agent must ask for a number to be dictated.",
        persona="Ты хочешь записаться на керамическое покрытие на ближайший понедельник, в 11:00.",
        facts=COMMON_FACTS.replace(
            "(это НЕ номер, с которого ты звонишь)", "(звонишь со скрытого номера)"
        ),
        behavior=(
            "Ты звонишь со скрытого номера. Продиктуй номер телефона, когда тебя попросят. "
            "Когда зачитают данные заявки, подтверди. На вопрос, нужна ли ещё помощь, скажи, "
            "что нет, и попрощайся."
        ),
        caller_phone=None,
        checks=(
            c.exactly_one_booking(),
            c.no_messages(),
            c.service_is("ceramic_coating"),
            c.date_is(MONDAY),
            c.time_is(time(11, 0)),
            c.phone_is(PHONE_SAVED),
            c.caller_phone_is(None),
            c.asks_to_dictate_number(),
            c.never_offers_this_number(),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="service_not_listed",
        description="Wants to BOOK something that is not on the price list («оклеить фары»).",
        persona=(
            "Ты хочешь ЗАПИСАТЬСЯ на оклейку фар цветной декоративной плёнкой (не защитной). "
            "Есть ли такая услуга в списке центра, ты не знаешь, но настроен именно записаться: "
            "тебе нужна запись, а не просто консультация. Записаться хочешь в ближайшую "
            "субботу, после обеда."
        ),
        facts=COMMON_FACTS,
        behavior=(
            "Первой фразой скажи, что хочешь записаться на оклейку фар цветной плёнкой. "
            f"{NUMBER_RULE} Если администратор скажет, что точно такой услуги нет или что цену "
            "и детали уточнит мастер, соглашайся записаться всё равно: администратор потом "
            "перезвонит и всё уточнит. Не соглашайся на вариант «просто передать вопрос»: "
            "тебе нужна именно запись. Точное время не называй. Когда зачитают заявку, "
            "подтверди. На вопрос, нужна ли ещё помощь, скажи, что нет, и попрощайся."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.exactly_one_booking(),
            c.no_messages(),
            c.service_in(["other", "ppf"]),
            c.other_service_notes_match(r"фар"),
            c.other_service_has_no_price(),
            c.time_empty(),
            c.phone_is(PHONE_SAVED),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="question_outside_faq",
        description="Asks about warranty terms, which the agent has no information about.",
        persona=(
            "Ты хочешь узнать, какая гарантия даётся на керамическое покрытие и на какой срок. "
            "Записываться пока не собираешься."
        ),
        facts="Тебя зовут Игорь. Телефон для связи: 8 916 123 45 67.",
        behavior=(
            "Если администратор скажет, что не знает ответа и передаст вопрос коллегам, "
            "согласись и, если спросят, назови имя и телефон. Не записывайся на услугу. "
            "На вопрос, нужна ли ещё помощь, скажи, что нет, и попрощайся."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.no_bookings(),
            c.exactly_one_message(),
            c.message_mentions(r"гаранти"),
            c.no_terms_invented(),
            c.no_amounts_spoken(),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="price_only",
        description="Only asks how much ceramic coating costs.",
        persona=(
            "Ты хочешь узнать, сколько стоит керамическое покрытие. Записываться не собираешься."
        ),
        facts="Тебя зовут Игорь.",
        behavior=(
            "Задай вопрос про цену. Услышав ответ, скажи «спасибо, я подумаю» и попрощайся. "
            "Не записывайся и не оставляй свои данные."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.no_bookings(),
            c.no_messages(),
            c.quotes_amount(25000),
            c.speech_matches("says_master_decides", r"мастер|осмотр"),
        ),
    ),
    Scenario(
        id="rude_offtopic",
        description="A rude caller: insults, off-topic talk, demands a discount, hangs up.",
        persona=(
            "Ты раздражён и груб: обзываешь администратора («вы тут все тупые»), требуешь скидку "
            "50% и заводишь разговор не по теме (политика, погода). Записываться на услугу ты "
            "на самом деле не собираешься."
        ),
        facts="Имя называть не хочешь.",
        behavior=(
            "Оставайся грубым три-четыре реплики, не отвечай на вопросы про запись. Потом "
            "резко скажи «всё, надоели, до свидания» и положи трубку."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.no_bookings(),
            c.speech_never_matches(
                "agent_stays_polite", r"идиот|дурак|тупо[йе]|заткнись|сам такой|козёл|урод"
            ),
            c.speech_never_matches(
                "no_discount_promise",
                r"(дадим|сделаем|предоставим|есть|будет|положена)\s+(вам\s+)?скидк",
            ),
        ),
    ),
    Scenario(
        id="address_only",
        description="Only wants the address and directions.",
        persona="Ты хочешь узнать, где находится детейлинг-центр и как туда проехать.",
        facts="Тебя зовут Игорь.",
        behavior=(
            "Спроси адрес и как проехать. Услышав ответ, поблагодари и попрощайся. Записываться "
            "не собираешься, свои данные не оставляй."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.no_bookings(),
            c.no_messages(),
            c.speech_matches("gives_address", r"примерн"),
            c.speech_matches("gives_directions", r"двор|ворот|вывеск|парковк"),
            c.agent_ended_call(),
        ),
    ),
    Scenario(
        id="sunday_closed",
        description="Wants to come on Sunday, when the center is closed; then agrees to Monday.",
        persona="Ты хочешь записаться на полировку кузова на это воскресенье, на 12:00.",
        facts=COMMON_FACTS,
        behavior=(
            f"{NUMBER_RULE} Если администратор скажет, что в воскресенье центр не работает, "
            "согласись перенести на понедельник, на 12:00. Когда зачитают данные заявки, "
            "подтверди. На вопрос, нужна ли ещё помощь, скажи, что нет, и попрощайся."
        ),
        caller_phone=CALLER_ID,
        checks=(
            c.no_booking_on_closed_day(),
            c.says_closed_that_day(),
            c.offers_another_day(),
            c.one_booking_on_an_open_day(),
            c.date_is(MONDAY),
            c.service_is("polishing"),
            c.phone_is(PHONE_SAVED),
            c.agent_ended_call(),
        ),
    ),
)

BY_ID = {scenario.id: scenario for scenario in SCENARIOS}
