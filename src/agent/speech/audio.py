"""Telephone audio: 8 kHz mono 16-bit PCM, WAV files, a simulated phone line, real-time pacing.

The agent's native audio is what Asterisk's AudioSocket carries: 8 kHz, 16-bit, mono, in 20 ms
frames. Anything else (a microphone at 48 kHz, an m4a voice memo, a TTS voice at 24 kHz) is
converted at the edge. `TelephoneChannel` makes audio sound like a call (8 kHz, the 300-3400 Hz
band, G.711), so recordings and the microphone can stand in for the phone.
"""

import asyncio
import subprocess
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from agent.speech import g711
from agent.speech.g711 import Codec
from agent.speech.resample import BandPass, Resampler, resample

TELEPHONY_RATE = 8000
FRAME_MS = 20
FRAME_SAMPLES = TELEPHONY_RATE * FRAME_MS // 1000  # 160
FRAME_BYTES = FRAME_SAMPLES * 2  # 320


class AudioError(Exception):
    """An audio file or stream that cannot be used."""


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Float or int samples to int16, rounded and clipped (no wrap-around on overload)."""
    return np.clip(np.rint(np.asarray(samples, dtype=np.float64)), -32768, 32767).astype(np.int16)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return to_pcm16(samples).tobytes()


def from_pcm16_bytes(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise AudioError("16-bit PCM must have an even number of bytes")
    return np.frombuffer(data, dtype="<i2")


# --- Files ---------------------------------------------------------------------------------------


def write_wav(path: Path | str, samples: np.ndarray, rate: int = TELEPHONY_RATE) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm16_bytes(samples))


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """(int16 mono samples, sample rate) of a 16-bit PCM WAV; stereo is averaged to mono."""
    try:
        with wave.open(str(path), "rb") as source:
            channels, width, rate = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
            frames = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"{path}: not a PCM WAV file ({exc})") from exc
    if width != 2:
        raise AudioError(f"{path}: only 16-bit WAV is supported, got {8 * width}-bit")
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = to_pcm16(samples.reshape(-1, channels).mean(axis=1))
    return samples, rate


def _decode_with_ffmpeg(path: Path, rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-",
    ]
    try:
        done = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise AudioError("ffmpeg is needed to read this file format but was not found") from exc
    if done.returncode != 0:
        raise AudioError(
            f"{path}: ffmpeg could not read it: {done.stderr.decode(errors='replace')[:200]}"
        )
    return from_pcm16_bytes(done.stdout)


def load_audio(path: Path | str, rate: int = TELEPHONY_RATE) -> np.ndarray:
    """Any audio file as int16 mono at `rate`. WAV is read directly, other formats (m4a, mp3,
    ogg...) go through ffmpeg."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"{path}: no such file")
    if path.suffix.lower() in (".wav", ".wave"):
        samples, file_rate = read_wav(path)
        return to_pcm16(resample(samples, file_rate, rate))
    return _decode_with_ffmpeg(path, rate)


# --- Framing and pacing --------------------------------------------------------------------------


def frames(pcm: np.ndarray | bytes, frame_samples: int = FRAME_SAMPLES) -> Iterator[bytes]:
    """Cut 16-bit PCM into frames of `frame_samples` (20 ms at 8 kHz); the last is zero-padded."""
    data = pcm if isinstance(pcm, bytes) else pcm16_bytes(pcm)
    size = frame_samples * 2
    for start in range(0, len(data), size):
        chunk = data[start : start + size]
        yield chunk + bytes(size - len(chunk))


async def paced(
    items: Iterable[bytes],
    interval: float = FRAME_MS / 1000,
    *,
    speed: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
) -> AsyncIterator[bytes]:
    """Yield the items one per `interval` seconds of real time, like a live line (`speed` 2.0 =
    twice as fast). Each item is scheduled from the start, so delays do not add up."""
    start = now()
    for index, item in enumerate(items):
        due = start + index * interval / speed
        if (wait := due - now()) > 0:
            await sleep(wait)
        yield item


# --- The phone line ------------------------------------------------------------------------------


class TelephoneChannel:
    """Makes audio sound like it came over a call: to 8 kHz, the telephone band, G.711 squeeze.

    A stream converter (feed chunks of any size at `in_rate`; every chunk gives int16 at 8 kHz),
    delayed by `delay` samples because of the band-pass filter."""

    def __init__(self, in_rate: int, codec: Codec = "alaw", *, band_limit: bool = True) -> None:
        self._codec = codec
        self._resampler = Resampler(in_rate, TELEPHONY_RATE)
        self._band = BandPass(TELEPHONY_RATE) if band_limit else None

    @property
    def delay(self) -> int:
        return self._band.delay if self._band else 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        return self._finish(self._resampler.process(chunk))

    def flush(self) -> np.ndarray:
        """The end of the signal, including the last `delay` samples held back by the filter."""
        return self._finish(np.concatenate([self._resampler.flush(), np.zeros(self.delay)]))

    def _finish(self, samples: np.ndarray) -> np.ndarray:
        if self._band is not None:
            samples = self._band.process(samples)
        return g711.roundtrip(to_pcm16(samples), self._codec)


def through_phone_line(samples: np.ndarray, in_rate: int, codec: Codec = "alaw") -> np.ndarray:
    """A whole recording as it would sound after a call: int16 at 8 kHz, same length and timing
    as the original (the filter delay is removed)."""
    channel = TelephoneChannel(in_rate, codec)
    out = np.concatenate([channel.process(samples), channel.flush()])
    return out[channel.delay :]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """`speech` plus `noise` (repeated or cut to the same length) at the given signal-to-noise
    ratio in dB, by RMS. Both at the same sample rate; returns float64, not clipped."""
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if len(noise) == 0:
        raise AudioError("the noise recording is empty")
    noise = np.resize(noise, len(speech))
    speech_rms = float(np.sqrt(np.mean(speech**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if noise_rms == 0 or speech_rms == 0:
        return speech
    gain = speech_rms / (noise_rms * 10 ** (snr_db / 20))
    return speech + noise * gain
