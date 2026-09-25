"""Business configuration (config/business.yaml): schema and loader."""

from datetime import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class BusinessConfigError(Exception):
    """Raised when business.yaml is missing, unparsable or invalid."""


class Weekday(StrEnum):
    MON = "mon"
    TUE = "tue"
    WED = "wed"
    THU = "thu"
    FRI = "fri"
    SAT = "sat"
    SUN = "sun"

    @classmethod
    def from_index(cls, index: int) -> "Weekday":
        """Map datetime.weekday() (Monday == 0) to a Weekday."""
        return list(cls)[index]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DayHours(_Model):
    open: time
    close: time

    @field_validator("open", "close", mode="before")
    @classmethod
    def _require_string(cls, value: Any) -> Any:
        # Unquoted 10:00 is parsed by YAML 1.1 as the integer 600, which pydantic
        # would silently read as 00:10. Require quoted "HH:MM" strings instead.
        if not isinstance(value, str | time):
            raise ValueError('time must be a quoted string like "10:00"')
        return value

    @model_validator(mode="after")
    def _check_order(self) -> "DayHours":
        if self.open >= self.close:
            raise ValueError("open must be earlier than close")
        return self


class Service(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    description: str | None = None
    price_from: int = Field(gt=0, description="Rubles")
    price_to: int | None = Field(default=None, gt=0, description="Rubles; None means 'from'")
    duration: str | None = Field(default=None, description='Free text, e.g. "3–4 часа"')

    @model_validator(mode="after")
    def _check_prices(self) -> "Service":
        if self.price_to is not None and self.price_to < self.price_from:
            raise ValueError("price_to must be >= price_from")
        return self


class FaqItem(_Model):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class BusinessConfig(_Model):
    name: str = Field(min_length=1)
    address: str = Field(min_length=1)
    directions: str | None = None
    greeting: str = Field(min_length=1)
    hours: dict[Weekday, DayHours | None] = Field(description="null means closed")
    services: list[Service] = Field(min_length=1)
    faq: list[FaqItem] = Field(default_factory=list)
    extra_rules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "BusinessConfig":
        missing = [day.value for day in Weekday if day not in self.hours]
        if missing:
            raise ValueError(f"hours: missing weekdays {missing} (use null for closed days)")
        if all(hours is None for hours in self.hours.values()):
            raise ValueError("hours: the business must be open at least one day")
        ids = [service.id for service in self.services]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"services: duplicate ids {duplicates}")
        return self

    def service_by_id(self, service_id: str) -> Service | None:
        return next((s for s in self.services if s.id == service_id), None)


def load_business_config(path: Path) -> BusinessConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BusinessConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise BusinessConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise BusinessConfigError(f"{path}: top level must be a mapping")

    try:
        return BusinessConfig.model_validate(raw)
    except ValidationError as exc:
        raise BusinessConfigError(f"invalid business config in {path}:\n{exc}") from exc
