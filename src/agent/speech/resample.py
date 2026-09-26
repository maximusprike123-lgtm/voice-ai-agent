"""Sample-rate conversion and band-limiting for streams, in numpy.

`Resampler` converts between any two rates in chunks of any size: it keeps the tail of the input
between calls, so cutting a signal into chunks gives the same output as converting it whole (a
test checks that). It is a windowed-sinc interpolator with the cutoff placed below the OUTPUT
Nyquist when downsampling, so 48 kHz -> 8 kHz does not alias.
"""

import math

import numpy as np

LOBES = 8  # half-width of the kernel, in lobes of the (scaled) sinc
CUTOFF = 0.95  # of the output Nyquist frequency


def _kernel(u: np.ndarray, half_width: float, cutoff: float) -> np.ndarray:
    """Blackman-windowed sinc, `cutoff` relative to the input Nyquist frequency."""
    x = np.abs(u) / half_width
    window = np.where(x < 1.0, 0.42 + 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2 * np.pi * x), 0.0)
    return cutoff * np.sinc(cutoff * u) * window


class Resampler:
    def __init__(self, in_rate: int, out_rate: int) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._in_rate, self._out_rate = in_rate, out_rate
        self._step = in_rate / out_rate  # input samples per output sample
        self._cutoff = min(1.0, 1.0 / self._step) * CUTOFF
        self._half = math.ceil(LOBES * max(1.0, self._step))  # in input samples
        self._offsets = np.arange(-self._half + 1, self._half + 1)
        self._buffer = np.zeros(self._half)  # input from `_start` on; zeros before the signal
        self._start = -self._half
        self._produced = 0  # output samples emitted so far
        self._consumed = 0  # real input samples received so far

    @property
    def is_identity(self) -> bool:
        return self._in_rate == self._out_rate

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Convert the next chunk of samples (any int or float array); returns float64."""
        samples = np.asarray(chunk, dtype=np.float64)
        if self.is_identity:
            return samples
        self._consumed += len(samples)
        self._buffer = np.concatenate([self._buffer, samples])
        return self._emit(self._start + len(self._buffer))

    def flush(self) -> np.ndarray:
        """The last output samples (the signal is treated as followed by silence)."""
        if self.is_identity:
            return np.zeros(0)
        wanted = math.floor(self._consumed * self._out_rate / self._in_rate) - self._produced
        self._buffer = np.concatenate([self._buffer, np.zeros(self._half)])
        out = self._emit(self._start + len(self._buffer))
        return out[: max(0, wanted)]

    def _emit(self, buffer_end: int) -> np.ndarray:
        """Output every sample whose whole kernel lies inside the buffer."""
        first = self._produced
        last_centre = buffer_end - 1 - self._half  # the latest input sample usable as a centre
        # the highest n whose position n * in / out still has its centre at or before that sample
        highest = -(-(last_centre + 1) * self._out_rate // self._in_rate) - 1
        count = highest - first + 1
        if count <= 0:
            return np.zeros(0)
        n = first + np.arange(count)
        positions = n * self._in_rate / self._out_rate
        centres = np.floor(positions).astype(np.int64)
        taps = centres[:, None] + self._offsets[None, :]
        weights = _kernel(positions[:, None] - taps, self._half, self._cutoff)
        weights /= weights.sum(axis=1, keepdims=True)
        out = (weights * self._buffer[taps - self._start]).sum(axis=1)

        self._produced += count
        next_centre = math.floor(self._produced * self._in_rate / self._out_rate)
        drop = next_centre - self._half + 1 - self._start
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._start += drop
        return out


def resample(samples: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Convert a whole signal at once; returns float64."""
    resampler = Resampler(in_rate, out_rate)
    return np.concatenate([resampler.process(samples), resampler.flush()])


class BandPass:
    """FIR band-pass for a stream, the telephone band by default (300-3400 Hz at 8 kHz).

    Linear phase, so it delays the signal by (taps - 1) / 2 samples, and chunks of any size give
    the same output as the whole signal."""

    def __init__(
        self, rate: int = 8000, low_hz: float = 300.0, high_hz: float = 3400.0, taps: int = 101
    ) -> None:
        if taps % 2 == 0:
            raise ValueError("taps must be odd")
        n = np.arange(taps) - (taps - 1) / 2
        window = np.blackman(taps)
        low_pass = lambda hz: (2 * hz / rate) * np.sinc(2 * hz / rate * n)  # noqa: E731
        self._h = (low_pass(high_hz) - low_pass(low_hz)) * window
        self._tail = np.zeros(taps - 1)

    @property
    def delay(self) -> int:
        return (len(self._h) - 1) // 2

    def process(self, chunk: np.ndarray) -> np.ndarray:
        samples = np.concatenate([self._tail, np.asarray(chunk, dtype=np.float64)])
        self._tail = samples[len(samples) - (len(self._h) - 1) :]
        return np.convolve(samples, self._h, mode="valid")
