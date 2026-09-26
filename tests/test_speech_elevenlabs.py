"""Offline tests for the ElevenLabs client: requests, streaming, errors, a secret key."""

import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from agent.speech.elevenlabs_tts import ElevenLabsTTS, list_voices
from agent.speech.tts import TtsError

KEY = "xi-secret-key-0123456789"
VOICE = "voice_abc"
BASE = "https://eleven.test"


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *parts):
        self._parts = parts

    async def __aiter__(self):
        for part in self._parts:
            yield part


def make(handler, *, key=KEY, voice=VOICE, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
    return ElevenLabsTTS(key, voice, client=client, **kwargs)


async def speak(tts, text="Здравствуйте."):
    return [chunk async for chunk in tts.synthesize(text)]


# --- Configuration -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "voice", "reason"),
    [
        (None, VOICE, "key: not set"),
        ("", VOICE, "key: not set"),
        ("   ", VOICE, "key: not set"),
        (SecretStr(""), VOICE, "key: not set"),
        (KEY, None, "voice: not set"),
        (KEY, "  ", "voice: not set"),
        (KEY, VOICE, None),
        (SecretStr(KEY), VOICE, None),
    ],
)
def test_the_client_says_what_is_missing(key, voice, reason):
    assert make(lambda r: httpx.Response(200), key=key, voice=voice).unconfigured_reason == reason


async def test_an_unconfigured_client_makes_no_request_and_names_the_problem():
    calls = []
    tts = make(lambda r: calls.append(r) or httpx.Response(200), key=None)

    with pytest.raises(TtsError, match="key: not set"):
        await speak(tts)
    await tts.warm_up()  # quiet no-op

    assert calls == []


# --- The request ---------------------------------------------------------------------------------


async def test_the_request_asks_for_8_khz_pcm_and_carries_the_key_only_in_a_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Chunks(b"\x01\x00" * 100))

    tts = make(handler, model="eleven_flash_v2_5", speed=1.1)
    await speak(tts, "Как вас зовут?")

    [request] = seen
    assert request.method == "POST"
    assert request.url.path == f"/v1/text-to-speech/{VOICE}/stream"
    assert dict(request.url.params) == {"output_format": "pcm_8000"}
    assert request.headers["xi-api-key"] == KEY
    body = json.loads(request.content)
    assert body["text"] == "Как вас зовут?"
    assert body["model_id"] == "eleven_flash_v2_5" and body["language_code"] == "ru"
    assert body["apply_text_normalization"] == "off"
    assert body["voice_settings"]["speed"] == 1.1
    assert KEY not in str(request.url) and KEY not in request.content.decode()


async def test_the_audio_comes_back_in_whole_samples_whatever_the_network_chunking():
    audio = bytes(range(200))
    parts = [audio[:3], audio[3:4], audio[4:101], audio[101:]]  # odd sizes, split mid-sample
    tts = make(lambda r: httpx.Response(200, stream=Chunks(*parts)))

    chunks = await speak(tts)

    assert all(len(c) % 2 == 0 and c for c in chunks)
    assert b"".join(chunks) == audio


async def test_a_trailing_half_sample_is_dropped():
    tts = make(lambda r: httpx.Response(200, stream=Chunks(b"\x01\x02\x03")))
    assert b"".join(await speak(tts)) == b"\x01\x02"


# --- Errors --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            401,
            {"detail": {"status": "invalid_api_key", "message": "Invalid API key"}},
            "Invalid API key",
        ),
        (403, {"detail": {"message": "Access from this country is not allowed"}}, "country"),
        (429, {"detail": {"status": "too_many_concurrent_requests"}}, "too_many_concurrent"),
        (422, {"detail": [{"msg": "bad"}]}, "bad"),
        (500, {}, "HTTP 500"),
    ],
)
async def test_http_errors_become_tts_errors_with_the_reason(status, body, expected):
    tts = make(lambda r: httpx.Response(status, json=body))

    with pytest.raises(TtsError) as caught:
        await speak(tts)

    assert f"HTTP {status}" in str(caught.value) and expected in str(caught.value)


async def test_an_error_page_that_is_not_json_is_shortened():
    tts = make(lambda r: httpx.Response(502, text="<html>" + "x" * 5000))
    with pytest.raises(TtsError) as caught:
        await speak(tts)
    assert len(str(caught.value)) < 300


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("slow"), "timed out"),
        (httpx.ConnectTimeout("slow"), "timed out"),
        (httpx.ConnectError("no route"), "ConnectError"),
    ],
)
async def test_network_failures_become_tts_errors(error, expected):
    def handler(request):
        raise error

    with pytest.raises(TtsError, match=expected):
        await speak(make(handler))


async def test_the_key_never_appears_in_errors_logs_or_the_repr(caplog):
    def handler(request):
        return httpx.Response(401, json={"detail": {"message": "bad key"}})

    tts = make(handler)
    with caplog.at_level(logging.DEBUG), pytest.raises(TtsError) as caught:
        await speak(tts)

    assert KEY not in str(caught.value) and KEY not in caplog.text
    assert KEY not in repr(tts) and "key=set" in repr(tts)
    assert "key=not set" in repr(make(handler, key=None))


# --- Warm-up -------------------------------------------------------------------------------------


async def test_warm_up_opens_the_connection_with_an_authenticated_request():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[])

    await make(handler).warm_up()

    assert [(r.method, r.url.path, r.headers["xi-api-key"]) for r in seen] == [
        ("GET", "/v1/models", KEY)
    ]


@pytest.mark.parametrize("status", [401, 403])
async def test_warm_up_reports_a_refused_key(status):
    with pytest.raises(TtsError, match=f"HTTP {status}"):
        await make(lambda r: httpx.Response(status)).warm_up()


async def test_warm_up_reports_an_unreachable_server_and_ignores_other_statuses():
    def down(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(TtsError, match="unreachable"):
        await make(down).warm_up()
    await make(lambda r: httpx.Response(500)).warm_up()  # the first real sentence will tell


# --- The list of voices --------------------------------------------------------------------------

VOICES_PAGE_1 = {
    "voices": [
        {
            "voice_id": "v1",
            "name": "Антон",
            "category": "professional",
            "labels": {
                "gender": "male",
                "accent": "russian",
                "age": "middle_aged",
                "use_case": "conversational",
            },
            "description": "A calm voice",
            "verified_languages": [
                {"language": "ru", "model_id": "eleven_multilingual_v2"},
                {"language": "ru", "model_id": "eleven_flash_v2_5"},
                {"language": "en", "model_id": "eleven_flash_v2_5"},
            ],
            "preview_url": "https://x/p1.mp3",
        },
        {
            "voice_id": "v2",
            "name": "Rachel",
            "labels": {"gender": "female"},
            "verified_languages": [{"language": "en"}],
        },
    ],
    "has_more": True,
    "next_page_token": "tok2",
}
VOICES_PAGE_2 = {
    "voices": [{"voice_id": "v3", "name": "Марина", "labels": {"gender": "female"}}],
    "has_more": False,
}


def voices_handler(seen):
    def handler(request):
        seen.append(request)
        page = (
            VOICES_PAGE_2 if request.url.params.get("next_page_token") == "tok2" else VOICES_PAGE_1
        )
        return httpx.Response(200, json=page)

    return handler


async def test_voices_are_listed_across_pages_with_their_labels():
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(voices_handler(seen)), base_url=BASE)

    voices = await list_voices(KEY, client=client, search="russian")

    assert [v.voice_id for v in voices] == ["v1", "v2", "v3"]
    anton = voices[0]
    assert (anton.name, anton.gender, anton.accent, anton.use_case) == (
        "Антон",
        "male",
        "russian",
        "conversational",
    )
    assert anton.languages == ("ru", "en") and anton.preview_url == "https://x/p1.mp3"
    assert [r.url.path for r in seen] == ["/v2/voices", "/v2/voices"]
    assert seen[0].url.params["search"] == "russian" and seen[0].headers["xi-api-key"] == KEY
    assert seen[1].url.params["next_page_token"] == "tok2"
    assert KEY not in str(seen[0].url)


async def test_voices_can_be_filtered_by_language_keeping_those_that_do_not_say():
    client = httpx.AsyncClient(transport=httpx.MockTransport(voices_handler([])), base_url=BASE)
    voices = await list_voices(KEY, client=client, language="ru")
    assert [v.voice_id for v in voices] == ["v1", "v3"]  # v2 lists only English


async def test_the_number_of_voices_is_limited():
    client = httpx.AsyncClient(transport=httpx.MockTransport(voices_handler([])), base_url=BASE)
    assert len(await list_voices(KEY, client=client, limit=2)) == 2


async def test_listing_without_a_key_says_so_and_makes_no_request():
    with pytest.raises(TtsError, match="key: not set"):
        await list_voices(None)


async def test_a_refused_listing_is_reported():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(401, json={"detail": {"message": "no"}})
        ),
        base_url=BASE,
    )
    with pytest.raises(TtsError, match="HTTP 401") as caught:
        await list_voices(KEY, client=client)
    assert KEY not in str(caught.value)
