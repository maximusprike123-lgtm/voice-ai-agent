"""Builds the call's TTS from the settings: the chosen primary, and the fallback behind it."""

from agent.settings import Settings
from agent.speech.elevenlabs_tts import ElevenLabsTTS
from agent.speech.silero_tts import SileroTTS
from agent.speech.tts import FailoverTTS, TTSClient


def build_voice(provider: str, settings: Settings) -> TTSClient:
    """One voice by name, for the settings' gender."""
    if provider == "elevenlabs":
        return ElevenLabsTTS(
            settings.elevenlabs_api_key,
            settings.elevenlabs_voice_id,
            model=settings.elevenlabs_tts_model,
            base_url=settings.elevenlabs_base_url,
        )
    if provider == "silero":
        return SileroTTS(
            settings.silero_model, settings.silero_speaker, cache_dir=settings.silero_cache_dir
        )
    raise ValueError(f"unknown TTS provider {provider!r}")


def build_tts(settings: Settings) -> TTSClient:
    """The voice of a call: the primary alone, or a FailoverTTS with the fallback behind it. If the
    fallback is the same provider as the primary (or "none") there is nothing to fail over to."""
    primary = build_voice(settings.tts_provider, settings)
    fallback_name = settings.tts_fallback_provider
    if fallback_name in ("none", settings.tts_provider):
        return primary
    return FailoverTTS(
        primary,
        build_voice(fallback_name, settings),
        first_chunk_timeout=settings.tts_first_chunk_timeout_seconds,
        pause_seconds=settings.tts_primary_pause_seconds,
    )
