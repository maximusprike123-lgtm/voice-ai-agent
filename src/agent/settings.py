"""Runtime settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM: generic OpenAI-compatible Chat Completions API.
    llm_base_url: str = Field(min_length=1)
    llm_api_key: SecretStr = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    # Optional passthrough for the "reasoning_effort" chat-completions field (part of the
    # OpenAI request schema, also honored by Ollama and others for hybrid-thinking models).
    # None means: omit the field entirely, so plain non-reasoning backends are unaffected.
    # "none" disables thinking; low/medium/high enable it at that effort level.
    llm_reasoning_effort: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    # Optional JSON object merged into every chat-completions request body (see LLMClient).
    # E.g. LLM_EXTRA_BODY={"provider": {"sort": "latency"}} for OpenRouter provider routing.
    llm_extra_body: dict[str, Any] | None = None
    # Per-turn stall limits used by DialogueEngine (not by the HTTP client): how long to wait
    # for the first token, and, once text is flowing, for each further chunk. A silence at the
    # start is retried once, so the caller waits at most about 2 x the first-event timeout.
    llm_first_event_timeout_seconds: float = Field(default=4.0, gt=0)
    llm_event_timeout_seconds: float = Field(default=8.0, gt=0)
    # Kill switch for the speech guard (agent.text_guard): on by default. Turning it off lets
    # the model's sentences through unchecked; it exists for debugging, not for production.
    speech_guard: bool = True

    telegram_bot_token: SecretStr = Field(min_length=1)
    telegram_chat_id: str = Field(min_length=1)

    timezone: str = "Europe/Moscow"
    business_config_path: Path = Path("config/business.yaml")
    db_path: Path = Path("data/agent.db")
    log_level: str = "INFO"

    @field_validator("llm_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("llm_reasoning_effort")
    @classmethod
    def _check_reasoning_effort(cls, value: str | None) -> str | None:
        allowed = {"none", "low", "medium", "high"}
        if value is not None and value not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)} or unset, got {value!r}")
        return value

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown time zone: {value!r}") from exc
        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache
def get_settings() -> Settings:
    return Settings()
