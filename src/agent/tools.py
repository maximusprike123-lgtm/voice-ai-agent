"""The agent's tools: prepare_booking, confirm_booking, take_message, end_call.

Principle: code enforces guarantees, the LLM handles conversation. So a booking is two steps.
`prepare_booking` validates, keeps a pending draft for this call and returns a read-back text
built by code, which the engine speaks verbatim (ToolOutcome.say). `confirm_booking` saves the
draft; the model calls it only after the caller says yes.

`ToolRegistry` implements the `ToolExecutor` protocol of DialogueEngine. Validation problems
are returned to the model as a result text starting with "ОШИБКА:" (never raised), so it can
ask the caller again. The tool schemas are static: nothing in them depends on the caller, the
time or the call, so they never break a backend's prompt-prefix cache.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from agent.business import OTHER_SERVICE_ID, BusinessConfig, Weekday
from agent.dialogue import ToolOutcome
from agent.llm import ToolCall, ToolSpec
from agent.prompt import format_hours
from agent.records import Booking, CallbackMessage, RecordSink
from agent.ru_words import WEEKDAYS_RU, date_on_phrase, digits_words, time_words

logger = logging.getLogger(__name__)

MAX_HORIZON_DAYS = 90
MAX_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 1000  # notes / message; keeps a future Telegram notification bounded
PHONE_DIGITS_READ_BACK = 4  # the read-back always says exactly this many trailing digits

ERROR_PREFIX = "ОШИБКА:"

# preferred_period value -> how it is said in the read-back.
PERIODS = {
    "утро": "утром",
    "день": "днём",
    "вечер": "вечером",
    "любое": "в любое время",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def normalize_phone(raw: str) -> str | None:
    """Return +7XXXXXXXXXX for any common Russian formatting, or None if it isn't a number."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits[0] in "78":
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "3456789":
        return None
    return f"+7{digits}"


def build_tool_specs(business: BusinessConfig) -> list[ToolSpec]:
    """Static schemas: they depend only on the business config, never on the call."""
    service_ids = [service.id for service in business.services] + [OTHER_SERVICE_ID]
    return [
        ToolSpec(
            name="prepare_booking",
            description=(
                "Подготовить заявку на запись и проверить данные. Вызывай, когда собраны все "
                "данные; при исправлении данных клиентом вызывай заново. Если вернулась ошибка, "
                "уточни у клиента и вызови снова. Если ошибки нет, система САМА зачитает клиенту "
                "текст для проверки: не пересказывай его и жди ответа клиента. Нужно передать "
                "хотя бы одно из полей preferred_time или preferred_period."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя клиента."},
                    "phone": {
                        "type": "string",
                        "description": "Телефон клиента для обратной связи, в любом формате.",
                    },
                    "car": {"type": "string", "description": "Марка и модель автомобиля."},
                    "service_id": {
                        "type": "string",
                        "enum": service_ids,
                        "description": (
                            "Идентификатор услуги из списка в квадратных скобках или "
                            f"{OTHER_SERVICE_ID}, если подходящей услуги нет."
                        ),
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Желаемая дата, формат YYYY-MM-DD.",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": (
                            "Желаемое время, формат HH:MM (24 часа). ТОЛЬКО если клиент назвал "
                            "точное время."
                        ),
                    },
                    "preferred_period": {
                        "type": "string",
                        "enum": list(PERIODS),
                        "description": (
                            "Примерный период дня, если точное время не названо: утро, день "
                            "(«после обеда»), вечер или любое («в любое время»)."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Слова клиента о времени как он их сказал («после обеда»), "
                            "пожелания; для услуги other обязательно: что именно нужно клиенту."
                        ),
                    },
                },
                "required": ["name", "phone", "car", "service_id", "preferred_date"],
            },
        ),
        ToolSpec(
            name="confirm_booking",
            description=(
                "Отправить подготовленную заявку администратору. Вызывай ТОЛЬКО после того, как "
                "клиент ясно сказал «да» на зачитанный текст проверки. Без аргументов."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        ToolSpec(
            name="take_message",
            description=(
                "Передать сообщение или вопрос администратору, когда ответа нет в сведениях "
                "или клиент просит с ним связаться."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Суть вопроса или просьбы."},
                    "name": {"type": "string", "description": "Имя клиента, если назвал."},
                    "phone": {"type": "string", "description": "Телефон, если назвал."},
                },
                "required": ["message"],
            },
        ),
        ToolSpec(
            name="end_call",
            description=(
                "Завершить звонок. Вызывай после прощания. После принятой заявки сначала "
                "спроси, нужна ли помощь ещё, и дождись ответа клиента: в том же ответе, где "
                "заявка принята, звонок завершить нельзя."
            ),
            parameters={"type": "object", "properties": {}},
        ),
    ]


@dataclass(frozen=True)
class _Fields:
    """A fully validated and normalized booking request."""

    name: str
    phone: str
    car: str
    service_id: str
    service_name: str
    preferred_date: date
    preferred_time: time | None
    preferred_period: str | None
    notes: str | None


class ToolRegistry:
    """Executes tool calls for ONE phone call (it remembers the draft and what was submitted)."""

    def __init__(
        self,
        business: BusinessConfig,
        sink: RecordSink,
        clock: Callable[[], datetime],
        caller_phone: str | None = None,
    ) -> None:
        self._business = business
        self._sink = sink
        self._clock = clock
        self._caller_phone = caller_phone
        self._draft: dict[str, Any] | None = None  # raw arguments of the pending booking
        self._submitted: set[tuple] = set()
        self._turn = 0  # counts caller utterances (see begin_turn)
        self._confirmed_in_turn: int | None = None  # turn of the last accepted booking
        self.specs = build_tool_specs(business)
        self._handlers = {
            "prepare_booking": self._prepare_booking,
            "confirm_booking": self._confirm_booking,
            "take_message": self._take_message,
            "end_call": self._end_call,
        }

    def begin_turn(self) -> None:
        self._turn += 1

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    async def execute(self, call: ToolCall) -> ToolOutcome:
        handler = self._handlers.get(call.name)
        if handler is None:
            names = ", ".join(self._handlers)
            return _error(f"неизвестный инструмент {call.name!r}. Доступны: {names}.")
        try:
            args = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError:
            return _error("аргументы вызова — некорректный JSON. Вызови инструмент ещё раз.")
        if not isinstance(args, dict):
            return _error("аргументы вызова должны быть JSON-объектом.")
        return await handler(args)

    # --- prepare_booking / confirm_booking ---------------------------------------------------

    async def _prepare_booking(self, args: dict[str, Any]) -> ToolOutcome:
        # A failed re-prepare must not leave the previous draft confirmable: the caller has
        # just changed something, so the old draft is stale either way.
        self._draft = None
        fields, problems = _validate_booking(args, self._now(), self._business)
        if fields is None:
            return _error(
                "заявка НЕ подготовлена. Исправь и вызови снова:\n- " + "\n- ".join(problems)
            )

        self._draft = args  # replaces any earlier draft
        return ToolOutcome(
            result=(
                "Заявка подготовлена, клиенту уже зачитан текст для проверки. Ничего не "
                "пересказывай, жди ответа клиента. Если клиент ясно сказал «да», вызови "
                "confirm_booking; если хочет что-то изменить, вызови prepare_booking заново."
            ),
            say=_read_back(fields),
        )

    async def _confirm_booking(self, args: dict[str, Any]) -> ToolOutcome:
        if self._draft is None:
            return _error(
                "нет подготовленной заявки. Сначала вызови prepare_booking; confirm_booking "
                "только после того, как клиент согласился с зачитанным текстом."
            )
        now = self._now()
        # Re-validate: the call may have run past closing time or midnight since the draft.
        fields, problems = _validate_booking(self._draft, now, self._business)
        if fields is None:
            self._draft = None
            return _error(
                "заявка больше не действительна, отправлять её нельзя:\n- "
                + "\n- ".join(problems)
                + "\nУточни данные у клиента и вызови prepare_booking заново."
            )

        key = (
            fields.name,
            fields.phone,
            fields.car,
            fields.service_id,
            fields.preferred_date,
            fields.preferred_time,
            fields.preferred_period,
            fields.notes,
        )
        if key in self._submitted:
            self._draft = None
            self._confirmed_in_turn = self._turn
            return ToolOutcome("Эта заявка уже принята раньше. Повторно отправлять не нужно.")

        booking = Booking(
            name=fields.name,
            phone=fields.phone,
            car=fields.car,
            service_id=fields.service_id,
            service_name=fields.service_name,
            preferred_date=fields.preferred_date,
            preferred_time=fields.preferred_time,
            preferred_period=fields.preferred_period,
            notes=fields.notes,
            caller_phone=self._caller_phone,
            created_at=now,
        )
        try:
            await self._sink.add_booking(booking)
        except Exception:
            logger.exception("could not store booking")
            # The draft stays, so confirm_booking can be retried.
            return _error(
                "заявку сохранить не удалось из-за технической проблемы. Извинись, скажи, что "
                "записать сейчас не получилось, и попроси клиента перезвонить позже."
            )
        self._draft = None
        self._submitted.add(key)
        self._confirmed_in_turn = self._turn

        result = (
            "Заявка принята и передана администратору. Скажи клиенту, что администратор "
            "перезвонит для подтверждения записи. Не говори, что время подтверждено. "
            "Затем спроси, нужна ли помощь ещё, и дождись ответа: пока клиент не ответил, "
            "end_call вызывать нельзя."
        )
        if fields.service_id == OTHER_SERVICE_ID:
            result += " Цену не называй: её определит мастер после осмотра."
        if fields.preferred_time is None:
            result += " Точное время не указано: администратор уточнит его при звонке."
        return ToolOutcome(result)

    # --- take_message --------------------------------------------------------------------

    async def _take_message(self, args: dict[str, Any]) -> ToolOutcome:
        problems: list[str] = []
        message = _text(args, "message", problems, required=True)
        name = _text(args, "name", problems)
        raw_phone = _text(args, "phone", problems)
        if problems or not message:
            return _error(
                "сообщение НЕ передано. Исправь и вызови снова:\n- " + "\n- ".join(problems)
            )

        phone = normalize_phone(raw_phone) if raw_phone else None
        if raw_phone and phone is None:
            # Don't lose the message over a bad number: keep what the caller said.
            message += f" (телефон, названный клиентом: {raw_phone})"
        record = CallbackMessage(
            message=message[:MAX_TEXT_LENGTH],
            name=name,
            phone=phone,
            caller_phone=self._caller_phone,
            created_at=self._now(),
        )
        try:
            await self._sink.add_message(record)
        except Exception:
            logger.exception("could not store message")
            return _error(
                "сообщение сохранить не удалось из-за технической проблемы. Извинись и "
                "попроси клиента перезвонить позже."
            )
        return ToolOutcome(
            "Сообщение передано администратору. Скажи клиенту, что администратор свяжется с ним."
        )

    # --- end_call ------------------------------------------------------------------------

    async def _end_call(self, args: dict[str, Any]) -> ToolOutcome:
        if self._confirmed_in_turn == self._turn:
            # The booking was accepted in this very turn: the caller has not heard that yet,
            # let alone said goodbye. Hanging up now would cut them off.
            return _error(
                "звонок завершить нельзя: клиент ещё не слышал, что заявка принята. Скажи, что "
                "администратор перезвонит, спроси, нужна ли помощь ещё, и дождись ответа "
                "клиента. Завершай звонок, только когда клиент попрощался или ничего больше "
                "не нужно."
            )
        return ToolOutcome("Звонок завершается.", ends_call=True)


# --- Validation ---------------------------------------------------------------------------------


def _validate_booking(
    args: dict[str, Any], now: datetime, business: BusinessConfig
) -> tuple[_Fields | None, list[str]]:
    """Validate and normalize booking arguments. Returns (fields, []) or (None, problems)."""
    problems: list[str] = []

    name = _text(args, "name", problems, required=True)
    if name and len(name) > MAX_NAME_LENGTH:
        problems.append(f"name: слишком длинное (максимум {MAX_NAME_LENGTH} символов).")
    car = _text(args, "car", problems, required=True)
    notes = _text(args, "notes", problems)
    if notes:
        notes = notes[:MAX_TEXT_LENGTH]

    phone = None
    raw_phone = _text(args, "phone", problems, required=True)
    if raw_phone:
        phone = normalize_phone(raw_phone)
        if phone is None:
            problems.append(
                f"phone: {raw_phone!r} не похож на российский номер (нужно 10 цифр после "
                "+7). Переспроси номер у клиента."
            )

    service_id = _text(args, "service_id", problems, required=True)
    service_name = None
    if service_id == OTHER_SERVICE_ID:
        service_name = "Другое / консультация"
        if not notes:
            problems.append(
                f"notes: для service_id={OTHER_SERVICE_ID} обязательно опиши, что нужно клиенту."
            )
    elif service_id:
        service = business.service_by_id(service_id)
        if service is None:
            valid = ", ".join([s.id for s in business.services] + [OTHER_SERVICE_ID])
            problems.append(f"service_id: неизвестная услуга {service_id!r}. Допустимо: {valid}.")
        else:
            service_name = service.name

    preferred_date = _parse_date(args, now, business, problems)

    raw_time = _text(args, "preferred_time", problems)
    preferred_time = _parse_time(raw_time, preferred_date, now, business, problems)
    period = _text(args, "preferred_period", problems)
    if period is not None and period not in PERIODS:
        problems.append(f"preferred_period: {period!r} — допустимо только: {', '.join(PERIODS)}.")
        period = None
    if raw_time is None and period is None and not _mentions(problems, "preferred_period"):
        problems.append(
            "нужно указать время: preferred_time, если клиент назвал точное время, или "
            f"preferred_period ({', '.join(PERIODS)}), если только примерный период. Не "
            "придумывай точное время."
        )

    if problems:
        return None, problems
    assert name and car and phone and service_id and service_name and preferred_date
    return (
        _Fields(
            name=name,
            phone=phone,
            car=car,
            service_id=service_id,
            service_name=service_name,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            preferred_period=period,
            notes=notes,
        ),
        [],
    )


def _mentions(problems: list[str], field: str) -> bool:
    return any(p.startswith(f"{field}:") for p in problems)


def _read_back(fields: _Fields) -> str:
    """The text the caller hears before confirming. Built by code, numbers in words."""
    if fields.service_id == OTHER_SERVICE_ID:
        service = f"другая услуга, вы описали так: {fields.notes}"
    else:
        service = _lower_first(fields.service_name)

    if fields.preferred_time is not None:
        when = f"в {time_words(fields.preferred_time)}"
    else:
        assert fields.preferred_period is not None
        when = PERIODS[fields.preferred_period]

    last_digits = digits_words(fields.phone[-PHONE_DIGITS_READ_BACK:])
    return (
        f"Проверьте, пожалуйста: {fields.name}, {service}, автомобиль {fields.car}, "
        f"{date_on_phrase(fields.preferred_date)}, {when}. "
        f"Номер телефона заканчивается на {last_digits}. "
        "Всё верно?"
    )


def _lower_first(text: str) -> str:
    """'Полировка кузова' -> 'полировка кузова', but leave 'PPF' and similar alone."""
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


# --- helpers ---------------------------------------------------------------------------------


def _error(text: str) -> ToolOutcome:
    return ToolOutcome(f"{ERROR_PREFIX} {text}")


def _text(
    args: dict[str, Any], key: str, problems: list[str], required: bool = False
) -> str | None:
    """A trimmed non-empty string, or None. Numbers are accepted and stringified (phones)."""
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, str | int | float | type(None)):
        problems.append(f"{key}: должно быть строкой.")
        return None
    text = "" if value is None else str(value).strip()
    if not text:
        if required:
            problems.append(f"{key}: обязательное поле, уточни у клиента.")
        return None
    return text


def _parse_date(
    args: dict[str, Any], now: datetime, business: BusinessConfig, problems: list[str]
) -> date | None:
    raw = _text(args, "preferred_date", problems, required=True)
    if raw is None:
        return None
    try:
        if not _DATE_RE.match(raw):
            raise ValueError
        day = date.fromisoformat(raw)
    except ValueError:
        problems.append(f"preferred_date: {raw!r} — нужна дата в формате YYYY-MM-DD.")
        return None

    today = now.date()
    weekday = WEEKDAYS_RU[day.weekday()]
    if day < today:
        problems.append(f"preferred_date: {raw} уже прошла (сегодня {today:%Y-%m-%d}).")
        return None
    if day > today + timedelta(days=MAX_HORIZON_DAYS):
        problems.append(
            f"preferred_date: {raw} слишком далеко, записываем не дальше чем на "
            f"{MAX_HORIZON_DAYS} дней вперёд. Уточни дату (возможно, ошибка в годе)."
        )
        return None
    if business.hours[Weekday.from_index(day.weekday())] is None:
        problems.append(f"preferred_date: {raw} ({weekday}) — в этот день мы не работаем.")
        return None
    return day


def _parse_time(
    raw: str | None,
    day: date | None,
    now: datetime,
    business: BusinessConfig,
    problems: list[str],
) -> time | None:
    if raw is None:
        return None
    match = _TIME_RE.match(raw)
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        problems.append(
            f"preferred_time: {raw!r} — нужно время HH:MM. Если клиент назвал лишь примерный "
            "период, не заполняй это поле, а передай preferred_period."
        )
        return None
    value = time(int(match[1]), int(match[2]))

    if day is None:
        return value  # the date problem is already reported (or the day is closed)
    hours = business.hours[Weekday.from_index(day.weekday())]
    if hours is not None and not (hours.open <= value < hours.close):
        problems.append(
            f"preferred_time: {value:%H:%M} вне часов работы в этот день "
            f"({WEEKDAYS_RU[day.weekday()]}: {format_hours(hours)})."
        )
        return None
    if day == now.date() and value <= now.time().replace(tzinfo=None):
        problems.append(f"preferred_time: {value:%H:%M} сегодня уже прошло (сейчас {now:%H:%M}).")
        return None
    return value
