"""Runtime settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
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
    # The agent's gender: picks the TTS voice (step 2) and how the model speaks about itself in
    # the prompt («я понял» / «я поняла»). Phrases written in code stay gender-neutral.
    agent_gender: Literal["male", "female"] = "male"

    # --- Speech output (TTS): a primary and a local fallback, both behind TTSClient -------------
    tts_provider: Literal["elevenlabs", "silero"] = "elevenlabs"
    # The provider used when the primary fails or is not configured; "none" = no fallback.
    tts_fallback_provider: Literal["elevenlabs", "silero", "none"] = "silero"
    # A primary that gives no first audio within this time loses the sentence to the fallback,
    # and then sits out `tts_primary_pause_seconds` before it is tried again.
    tts_first_chunk_timeout_seconds: float = Field(default=1.5, gt=0)
    tts_primary_pause_seconds: float = Field(default=60.0, ge=0)
    # ElevenLabs: without a key the provider reports "key: not set" and the fallback speaks.
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_base_url: str = "https://api.elevenlabs.io"  # a regional or proxy endpoint
    elevenlabs_voice_male: str | None = None
    elevenlabs_voice_female: str | None = None
    elevenlabs_tts_model: str = "eleven_flash_v2_5"
    # Silero (local): model v5_ru has 5 voices; v5_cis_base is the MIT-licensed base model.
    silero_model: str = "v5_ru"
    silero_cache_dir: Path = Path("data/silero")  # downloaded models (git-ignored)
    silero_speaker_male: str = "aidar"
    silero_speaker_female: str = "xenia"

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

    @field_validator("elevenlabs_api_key", "elevenlabs_voice_male", "elevenlabs_voice_female")
    @classmethod
    def _blank_is_unset(cls, value):
        """`KEY=` in .env means "not set", not an empty key."""
        if value is None:
            return None
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        return value if raw.strip() else None

    @field_validator("elevenlabs_base_url")
    @classmethod
    def _check_elevenlabs_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def elevenlabs_voice_id(self) -> str | None:
        """The ElevenLabs voice for the agent's gender, if one is configured."""
        if self.agent_gender == "male":
            return self.elevenlabs_voice_male
        return self.elevenlabs_voice_female

    @property
    def silero_speaker(self) -> str:
        """The Silero speaker for the agent's gender."""
        if self.agent_gender == "male":
            return self.silero_speaker_male
        return self.silero_speaker_female


@lru_cache
def get_settings() -> Settings:
    return Settings()
