"""Check the TTS voices: which are configured, how fast they start, and samples to listen to.

    .venv/bin/python scripts/check_tts.py check    # each voice and the failover, per sentence
    .venv/bin/python scripts/check_tts.py samples  # Silero voices as wav files, to listen to
    .venv/bin/python scripts/check_tts.py voices --search russian --language ru
                                                   # the ElevenLabs voices of the account (free)

Uses the settings in .env. A key is never printed: only "key: set" or "key: not set". `check`
calls ElevenLabs only if a key and a voice are configured (three short sentences, about 400
characters).
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.settings import Settings, get_settings  # noqa: E402
from agent.speech import audio  # noqa: E402
from agent.speech.elevenlabs_tts import list_voices  # noqa: E402
from agent.speech.factory import build_tts, build_voice  # noqa: E402
from agent.speech.silero_tts import SileroTTS  # noqa: E402
from agent.speech.text import SpeechText  # noqa: E402
from agent.speech.tts import TtsError  # noqa: E402
from agent.text_guard import SpeechGuard  # noqa: E402

OUT_CHECK = Path("data/tts_check")
OUT_SAMPLES = Path("data/tts_samples")

# What the agent really says: the greeting, the read-back, a price answer (all in words).
SENTENCES = [
    "Здравствуйте! Детейлинг-центр «Пример». Чем могу помочь?",
    "Проверьте, пожалуйста: Игорь, полировка кузова, автомобиль Тойота Камри, в субботу, "
    "двадцать шестого сентября, в четырнадцать часов. Номер телефона заканчивается на "
    "четыре пять шесть семь. Всё верно?",
    "Полировка кузова стоит от пятнадцати до тридцати пяти тысяч рублей. Точную стоимость "
    "определит мастер после осмотра.",
]
# v5_ru: 5 voices. v5_cis_base (MIT licence): 60 voices named <language>_<name>; only ru_* speak
# Russian as their own language, the others are accents.
SILERO_SAMPLE_MODELS = ("v5_ru", "v5_cis_base")
RUSSIAN_PREFIX = "ru_"


def spoken(sentence: str) -> str:
    result = SpeechText(SpeechGuard(committed=True)).prepare(sentence, scripted=True)
    return result.text or sentence


async def timed(tts, text: str) -> tuple[float, float, bytes]:
    """(seconds to the first chunk, seconds in total, all audio) for one sentence."""
    started = time.perf_counter()
    first = None
    parts = []
    async for chunk in tts.synthesize(text):
        if first is None:
            first = time.perf_counter() - started
        parts.append(chunk)
    return first or 0.0, time.perf_counter() - started, b"".join(parts)


def describe(voice) -> str:
    if voice.name == "elevenlabs":  # never the key itself
        key = "set" if voice.has_key else "not set"
        return f"key: {key}, voice: {'set' if voice.has_voice else 'not set'}"
    return "configured" if voice.unconfigured_reason is None else voice.unconfigured_reason


async def check(settings: Settings) -> int:
    print(f"gender: {settings.agent_gender}")
    print(f"primary: {settings.tts_provider}, fallback: {settings.tts_fallback_provider}\n")
    OUT_CHECK.mkdir(parents=True, exist_ok=True)

    for name in dict.fromkeys([settings.tts_provider, settings.tts_fallback_provider]):
        if name == "none":
            continue
        voice = build_voice(name, settings)
        print(f"== {name}: {describe(voice)}")
        if voice.unconfigured_reason:
            print("   skipped\n")
            await voice.aclose()
            continue
        started = time.perf_counter()
        try:
            await voice.warm_up()
        except TtsError as exc:
            print(f"   warm-up failed: {exc}\n")
            await voice.aclose()
            continue
        print(f"   warm-up {time.perf_counter() - started:.2f}s")
        await report(voice, name)
        await voice.aclose()

    print("== failover (what a call would use)")
    tts = build_tts(settings)
    await tts.warm_up()
    await report(tts, "failover")
    if hasattr(tts, "stats"):
        print(f"   stats: {tts.stats}")
    await tts.aclose()
    print(f"\nwav files: {OUT_CHECK.resolve()}")
    return 0


async def report(tts, label: str) -> None:
    for index, sentence in enumerate(SENTENCES, 1):
        text = spoken(sentence)
        try:
            first, total, data = await timed(tts, text)
        except TtsError as exc:
            print(f"   [{index}] failed: {exc}")
            continue
        seconds = len(data) / 2 / audio.TELEPHONY_RATE
        who = f" by {tts.last_provider}" if hasattr(tts, "last_provider") else ""
        print(
            f"   [{index}] first audio {first * 1000:6.0f} ms, total {total * 1000:6.0f} ms, "
            f"{seconds:5.1f} s of speech{who}, {len(text)} chars"
        )
        audio.write_wav(OUT_CHECK / f"{label}_{index}.wav", audio.from_pcm16_bytes(data))
    print()


async def samples(settings: Settings) -> int:
    """Every Silero voice reads the same three sentences: through the phone line as the agent
    would sound to a caller (`_phone.wav`), and at 24 kHz to judge the voice itself (`_24k.wav`)."""
    text = " ".join(spoken(s) for s in SENTENCES)
    written = []
    for model_id in SILERO_SAMPLE_MODELS:
        tts = SileroTTS(model_id, "-", cache_dir=settings.silero_cache_dir)
        print(f"== {model_id}: loading (first time downloads 90-150 MB)")
        try:
            model = await asyncio.to_thread(tts._loader, model_id, tts._cache_dir)
        except TtsError as exc:
            print(f"   skipped: {exc}")
            continue
        speakers = list(getattr(model, "speakers", []))
        if model_id != "v5_ru":
            speakers = [sp for sp in speakers if sp.startswith(RUSSIAN_PREFIX)]
        print(f"   speakers ({len(speakers)}): {', '.join(speakers)}")
        for speaker in speakers:
            voice = SileroTTS(
                model_id, speaker, cache_dir=settings.silero_cache_dir, loader=lambda *_, m=model: m
            )
            out = OUT_SAMPLES / model_id
            out.mkdir(parents=True, exist_ok=True)
            try:
                wide = audio.from_pcm16_bytes(await voice.render(text, 24000))
                narrow = audio.from_pcm16_bytes(await voice.render(text, 8000))
            except TtsError as exc:
                print(f"   {speaker}: {exc}")
                continue
            phone = audio.through_phone_line(narrow, audio.TELEPHONY_RATE, "alaw")
            audio.write_wav(out / f"{speaker}_24k.wav", wide, 24000)
            audio.write_wav(out / f"{speaker}_phone.wav", phone, audio.TELEPHONY_RATE)
            written.append(out / f"{speaker}_phone.wav")
            print(f"   {speaker}: {len(narrow) / 2 / 8000:.1f} s")
    index = OUT_SAMPLES / "INDEX.txt"
    lines = [
        "Silero voices reading the agent's three sentences (~12 s each).",
        "*_phone.wav: as a caller hears it (8 kHz, telephone band, A-law).",
        "*_24k.wav: the voice itself, wideband.",
        "Listen on a Mac:  afplay <file>",
        "v5_ru = CC BY-NC (internal tests only); v5_cis_base = MIT.",
        "Genders are NOT documented: judge by ear.",
        "",
        *(str(path.resolve()) for path in written),
    ]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(written)} voices; the list of files: {index.resolve()}")
    return 0


async def voices(settings: Settings, args: argparse.Namespace) -> int:
    print("key: set" if settings.elevenlabs_api_key else "key: not set")
    if not settings.elevenlabs_api_key:
        return 0
    try:
        found = await list_voices(
            settings.elevenlabs_api_key,
            base_url=settings.elevenlabs_base_url,
            search=args.search,
            voice_type=args.type,
            language=args.language,
            limit=args.limit,
        )
    except TtsError as exc:
        print(f"could not list voices: {exc}")
        return 1
    print(f"{len(found)} voices\n")
    for v in found:
        langs = ",".join(v.languages) or "?"
        print(f"{v.voice_id}  {v.name:<24} {v.gender or '?':<7} {v.accent or '-':<12} [{langs}]")
        if v.description:
            print(f"    {v.description}")
    print("\nPut the chosen id into ELEVENLABS_VOICE_MALE / ELEVENLABS_VOICE_FEMALE in .env")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the TTS voices.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("samples")
    v = sub.add_parser("voices")
    v.add_argument("--search", help="text to search in names, descriptions and labels")
    v.add_argument("--language", help="keep voices verified for this language code, e.g. ru")
    v.add_argument("--type", help="personal, community, default, workspace...")
    v.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "check":
        return asyncio.run(check(settings))
    if args.command == "samples":
        return asyncio.run(samples(settings))
    return asyncio.run(voices(settings, args))


if __name__ == "__main__":
    sys.exit(main())
