import pytest
from pydantic import ValidationError

from agent.settings import Settings

REQUIRED = {
    "LLM_BASE_URL": "https://llm.example.com/v1/",
    "LLM_API_KEY": "sk-test-secret",
    "LLM_MODEL": "test-model",
    "TELEGRAM_BOT_TOKEN": "123:abc",
    "TELEGRAM_CHAT_ID": "42",
}


@pytest.fixture
def env(monkeypatch):
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def test_loads_from_env(env):
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://llm.example.com/v1"
    assert settings.llm_model == "test-model"
    assert settings.timezone == "Europe/Moscow"
    assert settings.tz.key == "Europe/Moscow"


def test_secrets_are_not_printed(env):
    settings = Settings(_env_file=None)
    assert "sk-test-secret" not in repr(settings)
    assert settings.llm_api_key.get_secret_value() == "sk-test-secret"


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_missing_required_value_fails(env, missing):
    env.delenv(missing)
    with pytest.raises(ValidationError, match=missing.lower()):
        Settings(_env_file=None)


def test_empty_value_fails(env):
    env.setenv("LLM_MODEL", "")
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(_env_file=None)


def test_invalid_base_url_fails(env):
    env.setenv("LLM_BASE_URL", "llm.example.com")
    with pytest.raises(ValidationError, match="http"):
        Settings(_env_file=None)


def test_invalid_timezone_fails(env):
    env.setenv("TIMEZONE", "Mars/Olympus")
    with pytest.raises(ValidationError, match="time zone"):
        Settings(_env_file=None)
