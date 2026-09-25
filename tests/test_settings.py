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


def test_llm_stall_timeouts_default_to_4s_first_event_and_8s_between_events(env):
    settings = Settings(_env_file=None)
    assert settings.llm_first_event_timeout_seconds == 4.0
    assert settings.llm_event_timeout_seconds == 8.0


def test_llm_stall_timeouts_are_configurable(env):
    env.setenv("LLM_FIRST_EVENT_TIMEOUT_SECONDS", "2.5")
    env.setenv("LLM_EVENT_TIMEOUT_SECONDS", "6")
    settings = Settings(_env_file=None)
    assert (settings.llm_first_event_timeout_seconds, settings.llm_event_timeout_seconds) == (
        2.5,
        6,
    )


@pytest.mark.parametrize("name", ["LLM_FIRST_EVENT_TIMEOUT_SECONDS", "LLM_EVENT_TIMEOUT_SECONDS"])
@pytest.mark.parametrize("bad", ["0", "-1"])
def test_llm_stall_timeouts_must_be_positive(env, name, bad):
    env.setenv(name, bad)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_llm_extra_body_defaults_to_none(env):
    assert Settings(_env_file=None).llm_extra_body is None


def test_llm_extra_body_is_parsed_from_a_json_object(env):
    env.setenv("LLM_EXTRA_BODY", '{"provider": {"sort": "latency"}}')
    assert Settings(_env_file=None).llm_extra_body == {"provider": {"sort": "latency"}}


def test_llm_extra_body_rejects_invalid_json(env):
    env.setenv("LLM_EXTRA_BODY", "{not json")
    with pytest.raises(Exception, match="LLM_EXTRA_BODY|llm_extra_body"):
        Settings(_env_file=None)


def test_the_speech_guard_is_on_by_default_and_has_a_kill_switch(env):
    assert Settings(_env_file=None).speech_guard is True
    env.setenv("SPEECH_GUARD", "false")
    assert Settings(_env_file=None).speech_guard is False
