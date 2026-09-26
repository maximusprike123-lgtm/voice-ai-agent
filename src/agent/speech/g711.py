"""G.711 companding: A-law (used on most European lines) and mu-law, on numpy arrays.

The tables are built from the reference algorithm once and used as lookups, so encoding a frame
is a single array index. Decoding matches ffmpeg's `pcm_alaw` / `pcm_mulaw` exactly. Encoding
differs from ffmpeg only for samples that sit exactly between two code levels (about 1.5% of the
16-bit range): ffmpeg rounds to the nearest level, the reference algorithm used here truncates, so
the two can pick neighbouring codewords. Both are valid G.711 (tests check this).
"""

import functools
from typing import Literal

import numpy as np

Codec = Literal["alaw", "ulaw"]
CODECS: tuple[Codec, ...] = ("alaw", "ulaw")

_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635
_ALAW_SEGMENT_ENDS = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def _ulaw_encode_sample(sample: int) -> int:
    if sample < 0:
        sample, mask = -sample, 0x7F
    else:
        mask = 0xFF
    sample = min(sample, _ULAW_CLIP) + _ULAW_BIAS
    exponent = max(0, (sample >> 7).bit_length() - 1)  # 0..7
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ((exponent << 4) | mantissa) ^ mask


def _ulaw_decode_byte(byte: int) -> int:
    byte = ~byte & 0xFF
    magnitude = (((byte & 0x0F) << 3) + _ULAW_BIAS) << ((byte & 0x70) >> 4)
    return _ULAW_BIAS - magnitude if byte & 0x80 else magnitude - _ULAW_BIAS


def _alaw_encode_sample(sample: int) -> int:
    sample >>= 3  # 13 bits
    if sample >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        sample = -sample - 1
    segment = next((i for i, end in enumerate(_ALAW_SEGMENT_ENDS) if sample <= end), 8)
    if segment >= 8:
        return 0x7F ^ mask
    value = segment << 4
    value |= (sample >> 1) & 0x0F if segment < 2 else (sample >> segment) & 0x0F
    return value ^ mask


def _alaw_decode_byte(byte: int) -> int:
    byte ^= 0x55
    magnitude = (byte & 0x0F) << 4
    segment = (byte & 0x70) >> 4
    if segment == 0:
        magnitude += 8
    else:
        magnitude += 0x108
        if segment > 1:
            magnitude <<= segment - 1
    return magnitude if byte & 0x80 else -magnitude


_ENCODERS = {"alaw": _alaw_encode_sample, "ulaw": _ulaw_encode_sample}
_DECODERS = {"alaw": _alaw_decode_byte, "ulaw": _ulaw_decode_byte}


@functools.cache
def _encode_table(codec: Codec) -> np.ndarray:
    encode = _ENCODERS[codec]
    return np.array([encode(value) for value in range(-32768, 32768)], dtype=np.uint8)


@functools.cache
def _decode_table(codec: Codec) -> np.ndarray:
    decode = _DECODERS[codec]
    return np.array([decode(byte) for byte in range(256)], dtype=np.int16)


def _check(codec: str) -> None:
    if codec not in CODECS:
        raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")


def encode(pcm: np.ndarray, codec: Codec) -> bytes:
    """16-bit PCM samples to G.711 bytes, one byte per sample."""
    _check(codec)
    samples = np.asarray(pcm, dtype=np.int16)
    return _encode_table(codec)[samples.astype(np.int32) + 32768].tobytes()


def decode(data: bytes, codec: Codec) -> np.ndarray:
    """G.711 bytes to 16-bit PCM samples."""
    _check(codec)
    return _decode_table(codec)[np.frombuffer(data, dtype=np.uint8)]


def roundtrip(pcm: np.ndarray, codec: Codec) -> np.ndarray:
    """What comes out of the far end of a G.711 line: the samples, companded and expanded."""
    return decode(encode(pcm, codec), codec)
