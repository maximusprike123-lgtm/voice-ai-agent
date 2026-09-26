"""Silero v5 TTS: a local Russian voice, the fallback of the agent.

Runs on CPU (about 40 seconds of audio per second on one core, so a sentence takes ~0.1 s) and
renders straight to 8 kHz, so there is no resampling. It is not a streaming model: the first
chunk appears when the whole sentence is ready, and the sentence is then handed out in short
chunks. The model loads once per process (a few seconds; `warm_up`) and one sentence is rendered at
a time (a lock), which is plenty for a pilot.

Licence: the standard v5 models (`v5_ru`) are CC BY-NC, so internal tests only; `v5_cis_base` is
MIT. torch is imported only when a model is loaded, so importing this module is cheap.
"""

import asyncio
import inspect
import logging
import warnings
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np

from agent.speech.audio import TELEPHONY_RATE, pcm16_bytes
from agent.speech.tts import TtsError

logger = logging.getLogger(__name__)

CHUNK_MS = 100
SAMPLE_RATES = (8000, 24000, 48000)
PRIMING_TEXT = "Здравствуйте."
V5_RU_SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")
# What apply_tts is asked for, when the model takes the flag (the v5_ru ones do).
_OPTIONS = {"put_accent": True, "put_yo": True, "put_stress_homo": True, "put_yo_homo": True}


MODELS_URL = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
DEFAULT_CACHE_DIR = Path("data/silero")  # git-ignored; the models are 90-150 MB each


def _download(url: str, target: Path) -> None:
    """Download `url` to `target` (through a .part file, so a broken download is never taken for
    a model)."""
    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    logger.info("downloading %s (first use only)", url)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            with part.open("wb") as out:
                for data in response.iter_bytes(1 << 20):
                    out.write(data)
    except httpx.HTTPError as exc:
        raise TtsError(f"could not download {url}: {type(exc).__name__}") from None
    part.replace(target)


def _model_url(model_id: str, cache_dir: Path) -> str:
    import yaml

    listing = cache_dir / "models.yml"
    if not listing.exists():
        _download(MODELS_URL, listing)
    try:
        return yaml.safe_load(listing.read_text(encoding="utf-8"))["tts_models"]["ru"][model_id][
            "latest"
        ]["package"]
    except (KeyError, TypeError) as exc:
        raise TtsError(f"Silero has no Russian TTS model {model_id!r}") from exc


def load_silero_model(model_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Any:
    """Load a Silero TTS model (v4/v5 package format) from `cache_dir`, downloading it on first
    use. Needs torch. (The `silero` pip package would write into the working directory and into
    site-packages, so the loading is done here.)"""
    try:
        from torch import package
    except ImportError as exc:
        raise TtsError("torch is not installed (pip install torch)") from exc
    path = cache_dir / f"{model_id}.pt"
    if not path.exists():
        _download(_model_url(model_id, cache_dir), path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
            model = package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    except Exception as exc:
        raise TtsError(f"could not load {path}: {exc}") from exc
    model.to("cpu")
    return model


class SileroTTS:
    name = "silero"

    def __init__(
        self,
        model_id: str = "v5_ru",
        speaker: str = "aidar",
        *,
        sample_rate: int = TELEPHONY_RATE,
        chunk_ms: int = CHUNK_MS,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        loader: Callable[[str, Path], Any] = load_silero_model,
    ) -> None:
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"Silero renders at {SAMPLE_RATES} Hz")
        self.model_id = model_id
        self.speaker = speaker
        self._sample_rate = sample_rate
        self._chunk_bytes = sample_rate * chunk_ms // 1000 * 2
        self._cache_dir = cache_dir
        self._loader = loader
        self._model: Any = None
        self._options: dict[str, bool] = {}
        self._lock = asyncio.Lock()

    @property
    def unconfigured_reason(self) -> str | None:
        return None

    @property
    def speakers(self) -> list[str]:
        """The voices of the loaded model (loads it if needed is up to the caller: warm_up)."""
        return list(getattr(self._model, "speakers", []))

    async def warm_up(self) -> None:
        """Load the model and prime it with a short sentence: the first real synthesis is several
        times slower than the rest (kernels, caches), and a caller must not pay for that."""
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load)
                try:
                    await asyncio.to_thread(self._render, PRIMING_TEXT, self._sample_rate)
                except TtsError:
                    logger.warning("Silero priming sentence failed; the first sentence may be slow")

    def _load(self) -> None:
        model = self._loader(self.model_id, self._cache_dir)
        speakers = list(getattr(model, "speakers", []))
        if speakers and self.speaker not in speakers:
            raise TtsError(f"Silero model {self.model_id!r} has no speaker {self.speaker!r}")
        try:
            accepted = set(inspect.signature(model.apply_tts).parameters)
        except (TypeError, ValueError):
            accepted = set()
        self._options = {k: v for k, v in _OPTIONS.items() if k in accepted}
        self._model = model
        logger.info("Silero %s loaded, speaker %s", self.model_id, self.speaker)

    def _render(self, text: str, sample_rate: int) -> bytes:
        try:
            audio = self._model.apply_tts(
                text=text, speaker=self.speaker, sample_rate=sample_rate, **self._options
            )
        except Exception as exc:  # the model rejects text it cannot read
            raise TtsError(f"Silero could not read {text!r}: {exc}") from exc
        samples = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        return pcm16_bytes(np.asarray(samples, dtype=np.float64) * 32767)

    async def render(self, text: str, sample_rate: int | None = None) -> bytes:
        """The whole sentence as 16-bit PCM at `sample_rate` (default: this voice's rate)."""
        rate = sample_rate or self._sample_rate
        if rate not in SAMPLE_RATES:
            raise ValueError(f"Silero renders at {SAMPLE_RATES} Hz")
        await self.warm_up()
        async with self._lock:
            return await asyncio.to_thread(self._render, text, rate)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        data = await self.render(text)
        if not data:
            raise TtsError("Silero returned no audio")
        for start in range(0, len(data), self._chunk_bytes):
            yield data[start : start + self._chunk_bytes]

    async def aclose(self) -> None:
        self._model = None
