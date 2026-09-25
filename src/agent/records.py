"""What the agent produces: bookings and callback messages, and where they are sent.

`RecordSink` is the seam between the tools (step 1.6) and persistence/notification: step 1.7
adds a SQLite sink, step 1.8 wraps it with the Telegram notifier. Tools only see the protocol.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Protocol


@dataclass(frozen=True)
class Booking:
    name: str
    phone: str  # normalized: +7XXXXXXXXXX
    car: str
    service_id: str  # a business.yaml service id, or "other"
    service_name: str
    preferred_date: date
    preferred_time: time | None  # exact time, only if the caller named one
    preferred_period: str | None  # "утро" | "день" | "вечер" | "любое"
    notes: str | None
    caller_phone: str | None  # the number the call came from, if known
    created_at: datetime


@dataclass(frozen=True)
class CallbackMessage:
    message: str
    name: str | None
    phone: str | None  # normalized, if the caller gave a valid one
    caller_phone: str | None
    created_at: datetime


class RecordSink(Protocol):
    async def add_booking(self, booking: Booking) -> None:
        """Persist/forward a booking. Raise on failure so the tool can tell the model."""
        ...

    async def add_message(self, message: CallbackMessage) -> None: ...


@dataclass
class InMemorySink:
    """Keeps everything in lists. For tests, the CLI and live checks until step 1.7."""

    bookings: list[Booking] = field(default_factory=list)
    messages: list[CallbackMessage] = field(default_factory=list)

    async def add_booking(self, booking: Booking) -> None:
        self.bookings.append(booking)

    async def add_message(self, message: CallbackMessage) -> None:
        self.messages.append(message)
