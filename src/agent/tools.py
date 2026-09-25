"""The agent's tools: submit_booking, take_message, end_call.

`ToolRegistry` implements the `ToolExecutor` protocol of DialogueEngine. Validation problems
are returned to the model as a result text starting with "ОШИБКА:" (never raised), so it can
ask the caller again. The tool schemas are static: nothing in them depends on the caller, the
time or the call, so they never break a backend's prompt-prefix cache.
"""

import json
import logging
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Any

from agent.business import OTHER_SERVICE_ID, BusinessConfig, Weekday
from agent.dialogue import ToolOutcome
from agent.llm import ToolCall, ToolSpec
from agent.prompt import WEEKDAYS_RU, format_hours
from agent.records import Booking, CallbackMessage, RecordSink

logger = logging.getLogger(__name__)

MAX_HORIZON_DAYS = 90
MAX_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 1000  # notes / message; keeps a future Telegram notification bounded

ERROR_PREFIX = "ОШИБКА:"

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
            name="submit_booking",
            description=(
                "Отправить заявку на запись администратору. Вызывай только после того, как "
                "клиент подтвердил все данные заявки. Если вернулась ошибка, заявка НЕ "
                "отправлена: уточни у клиента и вызови снова."
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
                            "Желаемое время, формат HH:MM (24 часа). Только если клиент назвал "
                            "точное время. Если назван лишь примерный период, не заполняй."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Слова клиента, которые не попали в другие поля: примерный период "
                            "(«после обеда»), пожелания; для услуги other обязательно: что "
                            "именно нужно клиенту."
                        ),
                    },
                },
                "required": ["name", "phone", "car", "service_id", "preferred_date"],
            },
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
            description="Завершить звонок. Вызывай после прощания.",
            parameters={"type": "object", "properties": {}},
        ),
    ]


class _ArgumentError(Exception):
    """The model's arguments are unusable; the message goes back to the model."""


class ToolRegistry:
    """Executes tool calls for ONE phone call (it remembers what was already submitted)."""

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
        self._submitted: set[tuple] = set()
        self.specs = build_tool_specs(business)
        self._handlers = {
            "submit_booking": self._submit_booking,
            "take_message": self._take_message,
            "end_call": self._end_call,
        }

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    async def execute(self, call: ToolCall) -> ToolOutcome:
        handler = self._handlers.get(call.name)
        if handler is None:
            return _error(
                f"неизвестный инструмент {call.name!r}. Доступны: {', '.join(self._handlers)}."
            )
        try:
            args = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError:
            return _error("аргументы вызова — некорректный JSON. Вызови инструмент ещё раз.")
        if not isinstance(args, dict):
            return _error("аргументы вызова должны быть JSON-объектом.")
        return await handler(args)

    # --- submit_booking -----------------------------------------------------------------

    async def _submit_booking(self, args: dict[str, Any]) -> ToolOutcome:
        now = self._now()
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
                    f"notes: для service_id={OTHER_SERVICE_ID} обязательно опиши, что нужно "
                    "клиенту."
                )
        elif service_id:
            service = self._business.service_by_id(service_id)
            if service is None:
                valid = ", ".join([s.id for s in self._business.services] + [OTHER_SERVICE_ID])
                problems.append(
                    f"service_id: неизвестная услуга {service_id!r}. Допустимо: {valid}."
                )
            else:
                service_name = service.name

        preferred_date = _parse_date(args, now, self._business, problems)
        preferred_time = _parse_time(args, preferred_date, now, self._business, problems)

        if problems:
            return _error(
                "заявка НЕ отправлена. Исправь и вызови снова:\n- " + "\n- ".join(problems)
            )

        assert name and car and phone and service_id and service_name and preferred_date
        key = (name, phone, car, service_id, preferred_date, preferred_time, notes)
        if key in self._submitted:
            return ToolOutcome("Эта заявка уже принята раньше. Повторно отправлять не нужно.")

        booking = Booking(
            name=name,
            phone=phone,
            car=car,
            service_id=service_id,
            service_name=service_name,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            notes=notes,
            caller_phone=self._caller_phone,
            created_at=now,
        )
        try:
            await self._sink.add_booking(booking)
        except Exception:
            logger.exception("could not store booking")
            return _error(
                "заявку сохранить не удалось из-за технической проблемы. Извинись, скажи, что "
                "записать сейчас не получилось, и попроси клиента перезвонить позже."
            )
        self._submitted.add(key)

        result = (
            "Заявка принята и передана администратору. Скажи клиенту, что администратор "
            "перезвонит для подтверждения записи. Не говори, что время подтверждено. "
            "Цену не называй: её определит мастер."
        )
        if preferred_time is None:
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
        return ToolOutcome("Звонок завершается.", ends_call=True)


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
    args: dict[str, Any],
    day: date | None,
    now: datetime,
    business: BusinessConfig,
    problems: list[str],
) -> time | None:
    raw = _text(args, "preferred_time", problems)
    if raw is None:
        return None
    match = _TIME_RE.match(raw)
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        problems.append(
            f"preferred_time: {raw!r} — нужно время HH:MM. Если клиент назвал лишь примерный "
            "период, не заполняй это поле, а запиши его слова в notes."
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
