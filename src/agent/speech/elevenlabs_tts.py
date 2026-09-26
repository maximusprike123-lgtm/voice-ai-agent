"""ElevenLabs TTS over the HTTP streaming endpoint, and the list of voices.

Our sentences arrive whole, so the plain streaming endpoint fits: the first audio comes as fast as
with a WebSocket, the connection is kept alive between sentences, and the end of the sentence is
the end of the response (a WebSocket that carries several sentences has no such marker). Audio is
requested as `pcm_8000` (16-bit, 8 kHz, mono), which is the telephone format already.

The key lives in a SecretStr, goes out only in the `xi-api-key` header (never in a URL), and is
kept out of errors and logs. Without a key or a voice the client says so («key: not set») and the
FailoverTTS uses the fallback. Note: ElevenLabs states it blocks Russia, so requests may need a
regional or proxy endpoint (base_url) or may simply fail; that is what the fallback is for.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
from pydantic import SecretStr

from agent.speech.tts import TtsError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.elevenlabs.io"
DEFAULT_MODEL = "eleven_flash_v2_5"
OUTPUT_FORMAT = "pcm_8000"
ERROR_TEXT_LIMIT = 200


def _secret(value: SecretStr | str | None) -> str | None:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return raw.strip() if raw and raw.strip() else None


def _error_detail(response: httpx.Response) -> str:
    """A short, key-free description of an error response."""
    try:
        body = response.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status") or detail
        text = str(detail)
    except ValueError:
        text = response.text
    return " ".join(text.split())[:ERROR_TEXT_LIMIT]


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(
        self,
        api_key: SecretStr | str | None,
        voice_id: str | None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        language: str = "ru",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._key = _secret(api_key)
        self._voice_id = (voice_id or "").strip() or None
        self._model = model
        self._language = language
        self._voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": speed,
        }
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout or httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=5.0),
        )

    def __repr__(self) -> str:  # never shows the key
        return f"ElevenLabsTTS(model={self._model!r}, key={'set' if self._key else 'not set'})"

    @property
    def has_key(self) -> bool:
        return self._key is not None

    @property
    def has_voice(self) -> bool:
        return self._voice_id is not None

    @property
    def unconfigured_reason(self) -> str | None:
        if self._key is None:
            return "key: not set"
        if self._voice_id is None:
            return "voice: not set"
        return None

    async def warm_up(self) -> None:
        """Open the connection (TLS included) with a cheap authenticated request, so the first
        sentence does not pay for it."""
        if self.unconfigured_reason:
            return
        try:
            response = await self._client.get("/v1/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise TtsError(f"ElevenLabs is unreachable ({type(exc).__name__})") from None
        if response.status_code in (401, 403):
            raise TtsError(f"ElevenLabs refused the key (HTTP {response.status_code})")

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._key or ""}

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if reason := self.unconfigured_reason:
            raise TtsError(f"ElevenLabs is not configured ({reason})")
        body = {
            "text": text,
            "model_id": self._model,
            "language_code": self._language,
            "voice_settings": self._voice_settings,
            "apply_text_normalization": "off",  # the text is already words (agent.speech.text)
        }
        try:
            async with self._client.stream(
                "POST",
                f"/v1/text-to-speech/{self._voice_id}/stream",
                params={"output_format": OUTPUT_FORMAT},
                json=body,
                headers=self._headers(),
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise TtsError(
                        f"ElevenLabs HTTP {response.status_code}: {_error_detail(response)}"
                    )
                pending = b""
                async for data in response.aiter_bytes():
                    pending += data
                    even = len(pending) - len(pending) % 2  # 16-bit samples: whole pairs only
                    if even:
                        chunk, pending = pending[:even], pending[even:]
                        yield chunk
        except httpx.TimeoutException:
            raise TtsError("ElevenLabs timed out") from None
        except httpx.HTTPError as exc:
            raise TtsError(f"ElevenLabs request failed ({type(exc).__name__})") from None

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass(frozen=True)
class VoiceInfo:
    voice_id: str
    name: str
    category: str = ""
    gender: str = ""
    accent: str = ""
    age: str = ""
    use_case: str = ""
    description: str = ""
    languages: tuple[str, ...] = ()
    preview_url: str = ""
    labels: dict[str, str] = field(default_factory=dict)


def _voice_info(raw: dict) -> VoiceInfo:
    labels = {str(k): str(v) for k, v in (raw.get("labels") or {}).items()}
    languages = tuple(
        dict.fromkeys(
            v["language"] for v in raw.get("verified_languages") or [] if v.get("language")
        )
    )
    return VoiceInfo(
        voice_id=raw.get("voice_id", ""),
        name=raw.get("name", ""),
        category=raw.get("category") or "",
        gender=labels.get("gender", ""),
        accent=labels.get("accent", ""),
        age=labels.get("age", ""),
        use_case=labels.get("use_case") or labels.get("usecase", ""),
        description=(raw.get("description") or "")[:120],
        languages=languages,
        preview_url=raw.get("preview_url") or "",
        labels=labels,
    )


async def list_voices(
    api_key: SecretStr | str | None,
    *,
    base_url: str = DEFAULT_BASE_URL,
    search: str | None = None,
    voice_type: str | None = None,
    language: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> list[VoiceInfo]:
    """Voices of the account (and, with `voice_type="community"`, the shared library), newest
    pages first, filtered by `language` (an ISO code such as «ru») when the voice lists its
    languages. Listing costs no characters. Raises TtsError with no key or on a bad reply."""
    key = _secret(api_key)
    if key is None:
        raise TtsError("key: not set")
    own = client is None
    http = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=10.0)
    voices: list[VoiceInfo] = []
    token: str | None = None
    try:
        while len(voices) < limit:
            params: dict[str, str | int] = {"page_size": min(100, limit)}
            if search:
                params["search"] = search
            if voice_type:
                params["voice_type"] = voice_type
            if token:
                params["next_page_token"] = token
            try:
                response = await http.get("/v2/voices", params=params, headers={"xi-api-key": key})
            except httpx.HTTPError as exc:
                raise TtsError(f"ElevenLabs is unreachable ({type(exc).__name__})") from None
            if response.status_code != 200:
                raise TtsError(f"ElevenLabs HTTP {response.status_code}: {_error_detail(response)}")
            page = response.json()
            voices += [_voice_info(v) for v in page.get("voices", [])]
            token = page.get("next_page_token")
            if not page.get("has_more") or not token:
                break
    finally:
        if own:
            await http.aclose()
    if language:
        voices = [v for v in voices if not v.languages or language in v.languages]
    return voices[:limit]
