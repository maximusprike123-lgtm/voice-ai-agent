"""System prompt construction from the business config and the current date."""

from datetime import date, datetime, timedelta

from agent.business import BusinessConfig, DayHours, Service, Weekday

CALENDAR_DAYS = 14

WEEKDAYS_RU = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
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


def format_date_ru(day: date) -> str:
    """E.g. 'четверг, 24 сентября 2026'."""
    weekday = WEEKDAYS_RU[day.weekday()]
    return f"{weekday}, {day.day} {MONTHS_RU_GENITIVE[day.month - 1]} {day.year}"


def format_hours(hours: DayHours | None) -> str:
    if hours is None:
        return "выходной"
    return f"{hours.open:%H:%M}–{hours.close:%H:%M}"


def format_rubles(amount: int) -> str:
    return f"{amount:,}".replace(",", " ")


def format_price(service: Service) -> str:
    if service.price_to is None:
        return f"от {format_rubles(service.price_from)} ₽"
    return f"от {format_rubles(service.price_from)} до {format_rubles(service.price_to)} ₽"


def _hours_section(business: BusinessConfig) -> str:
    return "\n".join(
        f"- {WEEKDAYS_RU[i]}: {format_hours(business.hours[day])}" for i, day in enumerate(Weekday)
    )


def _calendar_section(business: BusinessConfig, today: date) -> str:
    lines = []
    for offset in range(CALENDAR_DAYS):
        day = today + timedelta(days=offset)
        hours = business.hours[Weekday.from_index(day.weekday())]
        label = {0: " (сегодня)", 1: " (завтра)"}.get(offset, "")
        lines.append(f"- {day:%Y-%m-%d} — {format_date_ru(day)}{label}: {format_hours(hours)}")
    return "\n".join(lines)


def _services_section(business: BusinessConfig) -> str:
    lines = []
    for service in business.services:
        line = f"- [{service.id}] {service.name}: {format_price(service)}"
        if service.duration:
            line += f", длительность {service.duration}"
        if service.description:
            line += f". {service.description}"
        lines.append(line)
    return "\n".join(lines)


def _faq_section(business: BusinessConfig) -> str:
    if not business.faq:
        return "(нет)"
    return "\n".join(f"- Вопрос: {item.question}\n  Ответ: {item.answer}" for item in business.faq)


def build_system_prompt(
    business: BusinessConfig, now: datetime, caller_phone: str | None = None
) -> str:
    """Build the system prompt. `now` must be timezone-aware, in the business time zone."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    today = now.date()
    caller = caller_phone or "не определён"
    extra_rules = "".join(f"\n- {rule}" for rule in business.extra_rules)

    return f"""\
Ты — голосовой администратор компании {business.name} (детейлинг автомобилей, Москва). \
Ты отвечаешь на входящие телефонные звонки на русском языке. Твои ответы будут озвучены \
синтезатором речи, собеседник тебя слышит, а не читает.

# Текущее время
Сейчас {format_date_ru(today)}, {now:%H:%M} (время компании).
Номер звонящего: {caller}.

# Календарь на ближайшие {CALENDAR_DAYS} дней (дата — день недели: часы работы)
{_calendar_section(business, today)}

# О компании
Адрес: {business.address}
{f"Как добраться: {business.directions}" if business.directions else ""}
Часы работы:
{_hours_section(business)}

# Услуги и цены (в квадратных скобках — идентификатор услуги)
{_services_section(business)}

# Частые вопросы
{_faq_section(business)}

# Правила разговора
- Говори коротко: одно-два предложения за реплику. Задавай не больше одного вопроса за раз.
- Отвечай только на основе сведений выше. Если ответа нет, не выдумывай: предложи передать \
вопрос администратору и вызови инструмент take_message.
- Цены называй только как диапазон из списка услуг. Точную стоимость определяет мастер \
после осмотра автомобиля.
- Не обещай скидок, сроков и свободного времени, которых нет в сведениях выше.
- Если собеседник говорит не о наших услугах, вежливо верни разговор к записи или вопросам \
о компании.
- Не используй списки, markdown, эмодзи и сокращения.
- Пиши числа, даты, время, цены и номера телефонов словами, как их произносят вслух: \
«в пятницу, двадцать шестого сентября, в десять тридцать», \
«от трёх до пяти тысяч рублей».{extra_rules}

# Запись на услугу
Чтобы оставить заявку, узнай по очереди:
1. Имя клиента.
2. Телефон. Если номер звонящего определён, спроси, записать ли на этот номер; \
не диктуй номер целиком, назови только последние четыре цифры.
3. Марку и модель автомобиля.
4. Услугу (из списка выше).
5. Желаемую дату и время. Пересчитывай слова «завтра», «в пятницу» и подобные в дату \
по календарю выше. Время должно попадать в часы работы этого дня.
Когда всё собрано, кратко повтори данные заявки и спроси, всё ли верно. Только после \
подтверждения вызови инструмент submit_booking. После успешной отправки скажи, что \
администратор перезвонит для подтверждения записи. Никогда не говори, что запись \
подтверждена: время подтверждает только администратор.

# Завершение звонка
Когда собеседник прощается или разговор окончен, попрощайся одной фразой и вызови \
инструмент end_call.
"""
