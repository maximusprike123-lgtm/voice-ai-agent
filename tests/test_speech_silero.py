"""Offline tests for SileroTTS with a fake model, the model listing, and the factory/settings."""

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from agent.settings import Settings
from agent.speech import silero_tts
from agent.speech.elevenlabs_tts import ElevenLabsTTS
from agent.speech.factory import build_tts, build_voice
from agent.speech.silero_tts import V5_RU_SPEAKERS, SileroTTS
from agent.speech.tts import FailoverTTS, TtsError


class FakeModel:
    """Stands in for a Silero model: `speakers`, and apply_tts returning float samples."""

    speakers = list(V5_RU_SPEAKERS)

    def __init__(self, samples=None, *, accepts_options=True, fail=None, delay=0.0):
        self._samples = samples if samples is not None else np.full(8000, 0.5, dtype=np.float32)
        self.calls = []
        self.running = 0
        self.max_running = 0
        self._fail = fail
        self._delay = delay
        if accepts_options:
            self.apply_tts = self._with_options
        else:
            self.apply_tts = self._plain

    def _work(self, kwargs):
        import time

        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.calls.append(kwargs)
        time.sleep(self._delay)
        self.running -= 1
        if self._fail:
            raise self._fail
        return self._samples

    def _with_options(
        self,
        text,
        speaker,
        sample_rate=24000,
        put_accent=False,
        put_yo=False,
        put_stress_homo=False,
        put_yo_homo=False,
    ):
        return self._work(
            dict(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo,
            )
        )

    def _plain(self, text, speaker, sample_rate=24000):
        return self._work(dict(text=text, speaker=speaker, sample_rate=sample_rate))


def voice(model, **kwargs):
    loads = []

    def loader(model_id, cache_dir):
        loads.append((model_id, cache_dir))
        return model

    return SileroTTS(loader=loader, **kwargs), loads


async def speak(tts, text="Здравствуйте."):
    return [chunk async for chunk in tts.synthesize(text)]


# --- SileroTTS -----------------------------------------------------------------------------------


async def test_a_sentence_comes_out_as_100_ms_chunks_of_8_khz_pcm():
    tts, _ = voice(FakeModel(), model_id="v5_ru", speaker="aidar")  # one second of samples

    chunks = await speak(tts)

    assert [len(c) for c in chunks] == [1600] * 10  # 100 ms = 800 samples = 1600 bytes
    samples = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert len(samples) == 8000 and set(samples.tolist()) == {16384}  # 0.5 of full scale


async def test_the_model_is_asked_for_8_khz_the_chosen_speaker_and_stress_marking():
    model = FakeModel()
    tts, _ = voice(model, speaker="xenia")
    await tts.warm_up()
    model.calls.clear()  # (the warm-up primed the model with a sentence)

    await speak(tts, "Как вас зовут?")

    assert model.calls == [
        dict(text="Как вас зовут?", speaker="xenia", sample_rate=8000, put_accent=True, put_yo=True)
    ]


async def test_flags_the_model_does_not_take_are_not_sent():
    model = FakeModel(accepts_options=False)
    tts, _ = voice(model)
    await tts.warm_up()
    model.calls.clear()
    await speak(tts)
    assert model.calls == [dict(text="Здравствуйте.", speaker="aidar", sample_rate=8000)]


async def test_other_sample_rates_and_a_bad_one():
    tts, _ = voice(FakeModel(np.zeros(24000, dtype=np.float32)), sample_rate=24000)
    assert len(await speak(tts)) == 10  # 100 ms at 24 kHz
    with pytest.raises(ValueError, match="8000"):
        SileroTTS(sample_rate=16000)


async def test_loud_samples_are_clipped_not_wrapped():
    tts, _ = voice(FakeModel(np.array([2.0, -2.0, 1.0, -1.0], dtype=np.float32)))
    samples = np.frombuffer(b"".join(await speak(tts)), dtype="<i2")
    assert samples.tolist() == [32767, -32768, 32767, -32767]


async def test_a_torch_tensor_like_result_is_accepted():
    class Tensor:
        def numpy(self):
            return np.full(160, 0.25, dtype=np.float32)

    tts, _ = voice(FakeModel(Tensor()))
    assert len(b"".join(await speak(tts))) == 320


async def test_the_model_loads_once_even_for_concurrent_sentences():
    model = FakeModel(delay=0.02)
    tts, loads = voice(model)

    await asyncio.gather(tts.warm_up(), speak(tts, "Раз."), speak(tts, "Два."))

    assert len(loads) == 1


async def test_sentences_are_rendered_one_at_a_time():
    model = FakeModel(delay=0.03)
    tts, _ = voice(model)
    await tts.warm_up()
    model.calls.clear()

    await asyncio.gather(*(speak(tts, f"Фраза {i}.") for i in range(4)))

    assert model.max_running == 1 and len(model.calls) == 4


async def test_warm_up_primes_the_model_once_so_the_first_sentence_is_not_slow():
    model = FakeModel()
    tts, _ = voice(model)

    await tts.warm_up()
    await tts.warm_up()

    assert [c["text"] for c in model.calls] == ["Здравствуйте."]  # one priming render, not two


async def test_a_priming_failure_does_not_fail_the_warm_up(caplog):
    tts, _ = voice(FakeModel(fail=ValueError("cold")))
    await tts.warm_up()  # no exception
    assert "priming" in caplog.text


async def test_an_unknown_speaker_is_reported_at_load_time():
    tts, _ = voice(FakeModel(), speaker="nobody")
    with pytest.raises(TtsError, match="no speaker 'nobody'"):
        await tts.warm_up()


async def test_text_the_model_cannot_read_becomes_a_tts_error_naming_it():
    tts, _ = voice(FakeModel(fail=ValueError("unknown symbol")))
    with pytest.raises(TtsError, match="Здравствуйте"):
        await speak(tts)


async def test_no_audio_is_an_error():
    tts, _ = voice(FakeModel(np.zeros(0, dtype=np.float32)))
    with pytest.raises(TtsError, match="no audio"):
        await speak(tts)


async def test_a_failed_load_is_a_tts_error_and_can_be_retried():
    attempts = []

    def loader(model_id, cache_dir):
        attempts.append(model_id)
        if len(attempts) == 1:
            raise TtsError("download failed")
        return FakeModel()

    tts = SileroTTS(loader=loader)
    with pytest.raises(TtsError, match="download failed"):
        await tts.warm_up()
    assert await speak(tts)  # the second attempt loads
    assert len(attempts) == 2


def test_silero_is_always_configured_and_lists_its_speakers_once_loaded():
    tts, _ = voice(FakeModel())
    assert tts.unconfigured_reason is None and tts.speakers == []


async def test_the_speakers_of_the_loaded_model_are_listed():
    tts, _ = voice(FakeModel())
    await tts.warm_up()
    assert tts.speakers == list(V5_RU_SPEAKERS)


async def test_the_cache_directory_reaches_the_loader():
    tts, loads = voice(FakeModel(), model_id="v5_cis_base", cache_dir=Path("/tmp/somewhere"))
    await tts.warm_up()
    assert loads == [("v5_cis_base", Path("/tmp/somewhere"))]


# --- The model listing ---------------------------------------------------------------------------


def test_the_model_url_is_read_from_the_downloaded_listing(tmp_path):
    (tmp_path / "models.yml").write_text(
        "tts_models:\n  ru:\n    v5_ru:\n      latest:\n        package: https://models.example/v5_ru.pt\n",
        encoding="utf-8",
    )
    assert silero_tts._model_url("v5_ru", tmp_path) == "https://models.example/v5_ru.pt"
    with pytest.raises(TtsError, match="no Russian TTS model 'v9_ru'"):
        silero_tts._model_url("v9_ru", tmp_path)


def test_a_missing_listing_is_downloaded_into_the_cache(tmp_path, monkeypatch):
    fetched = []

    def fake_download(url, target):
        fetched.append(url)
        target.write_text(
            "tts_models:\n  ru:\n    v5_ru:\n      latest:\n        package: https://m/v5_ru.pt\n"
        )

    monkeypatch.setattr(silero_tts, "_download", fake_download)
    assert silero_tts._model_url("v5_ru", tmp_path) == "https://m/v5_ru.pt"
    assert fetched == [silero_tts.MODELS_URL] and (tmp_path / "models.yml").exists()


@pytest.mark.skipif(
    not (os.environ.get("SILERO_LIVE") and Path("data/silero/v5_ru.pt").exists()),
    reason="set SILERO_LIVE=1 and download data/silero/v5_ru.pt to run the real model",
)
async def test_the_real_model_speaks_russian():
    tts = SileroTTS("v5_ru", "aidar")
    await tts.warm_up()
    samples = np.frombuffer(
        b"".join(await speak(tts, "Здравствуйте, чем могу помочь?")), dtype="<i2"
    )
    assert len(samples) > 8000 and np.abs(samples).max() > 1000
    assert set(V5_RU_SPEAKERS) <= set(tts.speakers)


# --- Settings and the factory --------------------------------------------------------------------

REQUIRED = {
    "LLM_BASE_URL": "https://llm.example.com/v1",
    "LLM_API_KEY": "sk-test",
    "LLM_MODEL": "m",
    "TELEGRAM_BOT_TOKEN": "1:a",
    "TELEGRAM_CHAT_ID": "1",
}


@pytest.fixture
def settings_env(monkeypatch):
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    for name in list(os.environ):
        if name.startswith(("TTS_", "ELEVENLABS_", "SILERO_", "AGENT_")):
            monkeypatch.delenv(name)
    return monkeypatch


def make_settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_the_defaults_are_elevenlabs_with_a_silero_fallback_and_no_key(settings_env):
    s = make_settings()
    assert (s.tts_provider, s.tts_fallback_provider) == ("elevenlabs", "silero")
    assert s.elevenlabs_api_key is None and s.elevenlabs_voice_id is None
    assert (s.tts_first_chunk_timeout_seconds, s.tts_primary_pause_seconds) == (1.5, 60.0)
    assert s.silero_model == "v5_ru" and s.silero_speaker == "aidar"


def test_blank_values_in_env_mean_not_set(settings_env):
    settings_env.setenv("ELEVENLABS_API_KEY", "")
    settings_env.setenv("ELEVENLABS_VOICE_MALE", "  ")
    s = make_settings()
    assert s.elevenlabs_api_key is None and s.elevenlabs_voice_male is None


def test_the_gender_picks_the_voice_and_the_speaker(settings_env):
    male = make_settings(elevenlabs_voice_male="m1", elevenlabs_voice_female="f1")
    female = make_settings(
        agent_gender="female", elevenlabs_voice_male="m1", elevenlabs_voice_female="f1"
    )
    assert (male.elevenlabs_voice_id, male.silero_speaker) == ("m1", "aidar")
    assert (female.elevenlabs_voice_id, female.silero_speaker) == ("f1", "xenia")


def test_the_key_stays_secret_in_the_settings(settings_env):
    s = make_settings(elevenlabs_api_key="xi-very-secret")
    assert "xi-very-secret" not in repr(s) and "xi-very-secret" not in str(s)


def test_the_base_url_is_validated_and_trimmed(settings_env):
    assert make_settings(elevenlabs_base_url="https://proxy.example/").elevenlabs_base_url == (
        "https://proxy.example"
    )
    with pytest.raises(ValueError):
        make_settings(elevenlabs_base_url="proxy.example")


def test_the_default_voice_is_a_failover_with_an_unconfigured_primary(settings_env):
    tts = build_tts(make_settings())

    assert isinstance(tts, FailoverTTS)
    assert tts.unconfigured_reason is None  # the Silero fallback can speak
    assert (
        isinstance(tts._primary, ElevenLabsTTS)
        and tts._primary.unconfigured_reason == "key: not set"
    )
    assert isinstance(tts._fallback, SileroTTS) and tts._fallback.speaker == "aidar"


def test_the_factory_hands_the_settings_to_the_voices(settings_env):
    s = make_settings(
        elevenlabs_api_key="k",
        elevenlabs_voice_male="voice-m",
        tts_first_chunk_timeout_seconds=0.9,
        tts_primary_pause_seconds=12,
        silero_model="v5_cis_base",
    )
    tts = build_tts(s)

    assert tts._primary.unconfigured_reason is None
    assert tts._fallback.model_id == "v5_cis_base"
    assert (tts._first_chunk_timeout, tts._pause_seconds) == (0.9, 12)


@pytest.mark.parametrize(
    ("provider", "fallback", "expected"),
    [
        ("silero", "silero", SileroTTS),
        ("silero", "none", SileroTTS),
        ("elevenlabs", "none", ElevenLabsTTS),
        ("elevenlabs", "elevenlabs", ElevenLabsTTS),
    ],
)
def test_no_failover_when_there_is_nothing_to_fail_over_to(
    settings_env, provider, fallback, expected
):
    tts = build_tts(make_settings(tts_provider=provider, tts_fallback_provider=fallback))
    assert isinstance(tts, expected)


def test_silero_can_be_the_primary_with_elevenlabs_behind_it(settings_env):
    tts = build_tts(make_settings(tts_provider="silero", tts_fallback_provider="elevenlabs"))
    assert isinstance(tts, FailoverTTS) and isinstance(tts._primary, SileroTTS)


def test_an_unknown_provider_name_is_rejected(settings_env):
    with pytest.raises(ValueError):
        build_voice("festival", make_settings())
    with pytest.raises(ValueError):
        make_settings(tts_provider="festival")
