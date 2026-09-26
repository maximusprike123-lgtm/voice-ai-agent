"""Offline tests for the audio helpers: G.711, resampling, the phone line, WAV, pacing."""

import shutil
import subprocess

import numpy as np
import pytest

from agent.speech import audio, g711
from agent.speech.resample import BandPass, Resampler, resample

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def tone(hz, rate, seconds=1.0, amplitude=10000):
    return np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate) * amplitude


def rms(samples):
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


# --- G.711 ------------------------------------------------------------------------------------


def ffmpeg_encode(samples, codec):
    name = {"alaw": "alaw", "ulaw": "mulaw"}[codec]
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "8000", "-ac", "1", "-i", "-"]
        + ["-f", name, "-acodec", f"pcm_{name}", "-"],
        input=samples.astype("<i2").tobytes(),
        capture_output=True,
        check=True,
    )
    return done.stdout


def ffmpeg_decode(data, codec):
    name = {"alaw": "alaw", "ulaw": "mulaw"}[codec]
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", name, "-ar", "8000", "-ac", "1", "-i", "-"]
        + ["-f", "s16le", "-"],
        input=data,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(done.stdout, dtype="<i2")


def level_index(code, codec):
    """(sign, position among the code levels by size) of a G.711 byte."""
    value = (code ^ 0x55) if codec == "alaw" else (~code & 0xFF)
    return value >> 7, value & 0x7F


@needs_ffmpeg
@pytest.mark.parametrize("codec", g711.CODECS)
def test_encoding_agrees_with_ffmpeg_up_to_the_neighbouring_level(codec):
    every_sample = np.arange(-32768, 32768, dtype=np.int16)
    ours = np.frombuffer(g711.encode(every_sample, codec), dtype=np.uint8)
    theirs = np.frombuffer(ffmpeg_encode(every_sample, codec), dtype=np.uint8)

    different = np.nonzero(ours != theirs)[0]
    assert len(different) < 0.02 * len(every_sample)  # only samples on a boundary between levels
    for i in different:
        (sign_a, level_a), (sign_b, level_b) = (
            level_index(int(ours[i]), codec),
            level_index(int(theirs[i]), codec),
        )
        assert sign_a == sign_b and abs(level_a - level_b) == 1, int(every_sample[i])


@needs_ffmpeg
@pytest.mark.parametrize("codec", g711.CODECS)
def test_decoding_matches_ffmpeg_for_every_byte(codec):
    every_byte = bytes(range(256))
    assert np.array_equal(g711.decode(every_byte, codec), ffmpeg_decode(every_byte, codec))


@pytest.mark.parametrize("codec", g711.CODECS)
def test_one_byte_per_sample_and_int16_back(codec):
    samples = np.array([0, 1, -1, 1000, -1000, 32767, -32768], dtype=np.int16)
    data = g711.encode(samples, codec)
    assert len(data) == len(samples)
    assert g711.decode(data, codec).dtype == np.int16


@pytest.mark.parametrize("codec", g711.CODECS)
def test_companding_error_is_small_relative_to_the_signal(codec):
    signal = tone(440, 8000, amplitude=12000).astype(np.int16)
    error = signal.astype(np.float64) - g711.roundtrip(signal, codec)
    assert 20 * np.log10(rms(signal) / rms(error)) > 30  # dB


@pytest.mark.parametrize("codec", g711.CODECS)
def test_a_second_pass_through_the_codec_changes_nothing(codec):
    once = g711.roundtrip(tone(300, 8000).astype(np.int16), codec)
    assert np.array_equal(g711.roundtrip(once, codec), once)


def test_known_codewords():
    assert g711.decode(bytes([0xFF, 0x7F]), "ulaw").tolist() == [0, 0]  # silence
    assert g711.decode(bytes([0x00, 0x80]), "ulaw").tolist() == [-32124, 32124]  # the extremes
    assert g711.decode(bytes([0xD5, 0x55]), "alaw").tolist() == [8, -8]  # the smallest steps


def test_an_unknown_codec_is_rejected():
    with pytest.raises(ValueError, match="codec"):
        g711.encode(np.zeros(4, dtype=np.int16), "opus")
    with pytest.raises(ValueError, match="codec"):
        g711.decode(b"\x00", "opus")


# --- Resampling -------------------------------------------------------------------------------

RATE_PAIRS = [
    (48000, 8000),
    (44100, 8000),
    (24000, 8000),
    (22050, 8000),
    (16000, 8000),
    (8000, 16000),
    (8000, 24000),
    (8000, 48000),
]


@pytest.mark.parametrize(("in_rate", "out_rate"), RATE_PAIRS)
@pytest.mark.parametrize("length", [5, 1000, 4801])
def test_a_whole_signal_gives_exactly_the_expected_length(in_rate, out_rate, length):
    out = resample(tone(300, in_rate)[:length], in_rate, out_rate)
    assert len(out) == length * out_rate // in_rate


@pytest.mark.parametrize(("in_rate", "out_rate"), RATE_PAIRS)
@pytest.mark.parametrize("chunk", [1, 37, 333, 4096])
def test_cutting_the_stream_into_chunks_changes_nothing(in_rate, out_rate, chunk):
    signal = tone(700, in_rate, 0.25) + tone(2100, in_rate, 0.25, 3000)
    whole = resample(signal, in_rate, out_rate)

    resampler = Resampler(in_rate, out_rate)
    parts = [resampler.process(signal[i : i + chunk]) for i in range(0, len(signal), chunk)]
    streamed = np.concatenate([*parts, resampler.flush()])

    assert len(streamed) == len(whole)
    assert np.allclose(streamed, whole, atol=1e-9)


@pytest.mark.parametrize(("in_rate", "out_rate"), RATE_PAIRS)
def test_a_tone_inside_the_band_keeps_its_level(in_rate, out_rate):
    out = resample(tone(1000, in_rate), in_rate, out_rate)
    assert rms(out[100:-100]) == pytest.approx(rms(tone(1000, in_rate)), rel=0.01)


@pytest.mark.parametrize("in_rate", [16000, 24000, 48000])
def test_a_tone_above_the_output_band_is_removed_not_folded_down(in_rate):
    out = resample(tone(6000, in_rate), in_rate, 8000)  # 6 kHz would alias to 2 kHz
    assert rms(out[100:-100]) < 0.01 * rms(tone(6000, in_rate))


def test_the_same_rate_passes_the_signal_through():
    signal = tone(500, 8000, 0.1)
    resampler = Resampler(8000, 8000)
    assert np.array_equal(resampler.process(signal), signal)
    assert len(resampler.flush()) == 0


def test_an_empty_chunk_is_fine_and_invalid_rates_are_rejected():
    resampler = Resampler(48000, 8000)
    assert len(resampler.process(np.zeros(0))) == 0
    with pytest.raises(ValueError):
        Resampler(0, 8000)


# --- Band-pass --------------------------------------------------------------------------------


def test_the_telephone_band_passes_speech_frequencies_and_cuts_the_rest():
    band = BandPass()
    for hz, low, high in [(1000, 0.95, 1.05), (50, 0, 0.05), (3900, 0, 0.3)]:
        out = BandPass().process(tone(hz, 8000, 1.0))[500:-500]
        assert low <= rms(out) / rms(tone(hz, 8000)) <= high, hz
    assert band.delay == 50


def test_the_band_pass_gives_the_same_output_whatever_the_chunking():
    signal = tone(900, 8000, 0.5) + tone(200, 8000, 0.5, 4000)
    whole = BandPass().process(signal)
    band = BandPass()
    chunked = np.concatenate(
        [band.process(signal[i : i + 160]) for i in range(0, len(signal), 160)]
    )
    assert np.allclose(chunked, whole)


def test_the_band_pass_needs_an_odd_number_of_taps():
    with pytest.raises(ValueError):
        BandPass(taps=100)


# --- PCM helpers and files --------------------------------------------------------------------


def test_to_pcm16_rounds_and_clips_instead_of_wrapping():
    out = audio.to_pcm16(np.array([0.4, 0.6, -0.6, 40000.0, -40000.0]))
    assert out.tolist() == [0, 1, -1, 32767, -32768]


def test_pcm_bytes_roundtrip_and_odd_lengths_are_rejected():
    samples = np.array([0, 1, -2, 32767, -32768], dtype=np.int16)
    assert np.array_equal(audio.from_pcm16_bytes(audio.pcm16_bytes(samples)), samples)
    with pytest.raises(audio.AudioError):
        audio.from_pcm16_bytes(b"\x00\x01\x02")


def test_a_wav_file_roundtrips(tmp_path):
    samples = tone(500, 8000, 0.2).astype(np.int16)
    audio.write_wav(tmp_path / "a.wav", samples)
    back, rate = audio.read_wav(tmp_path / "a.wav")
    assert rate == 8000 and np.array_equal(back, samples)


def test_stereo_is_averaged_to_mono(tmp_path):
    import wave

    left, right = np.full(100, 1000, dtype="<i2"), np.full(100, 3000, dtype="<i2")
    with wave.open(str(tmp_path / "s.wav"), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.column_stack([left, right]).tobytes())
    samples, rate = audio.read_wav(tmp_path / "s.wav")
    assert rate == 16000 and set(samples.tolist()) == {2000}


def test_unsupported_wav_and_missing_files_are_reported(tmp_path):
    import wave

    with wave.open(str(tmp_path / "eight.wav"), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(1)
        out.setframerate(8000)
        out.writeframes(b"\x80" * 10)
    with pytest.raises(audio.AudioError, match="16-bit"):
        audio.read_wav(tmp_path / "eight.wav")
    (tmp_path / "not.wav").write_bytes(b"this is not audio")
    with pytest.raises(audio.AudioError, match="not a PCM WAV"):
        audio.read_wav(tmp_path / "not.wav")
    with pytest.raises(audio.AudioError, match="no such file"):
        audio.load_audio(tmp_path / "missing.wav")


def test_load_audio_converts_a_wav_to_telephone_rate(tmp_path):
    audio.write_wav(tmp_path / "wide.wav", tone(1000, 16000, 0.5).astype(np.int16), 16000)
    out = audio.load_audio(tmp_path / "wide.wav")
    assert out.dtype == np.int16 and len(out) == 4000
    assert rms(out[50:-50]) == pytest.approx(10000 / np.sqrt(2), rel=0.02)


@needs_ffmpeg
def test_load_audio_reads_other_formats_through_ffmpeg(tmp_path):
    audio.write_wav(tmp_path / "wide.wav", tone(1000, 16000, 0.5).astype(np.int16), 16000)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(tmp_path / "wide.wav"), str(tmp_path / "wide.flac")],
        check=True,
    )
    out = audio.load_audio(tmp_path / "wide.flac")
    assert abs(len(out) - 4000) <= 8
    assert rms(out[50:-50]) == pytest.approx(10000 / np.sqrt(2), rel=0.02)


# --- Frames and pacing ------------------------------------------------------------------------


def test_frames_are_twenty_milliseconds_and_the_last_is_padded():
    assert audio.FRAME_SAMPLES == 160 and audio.FRAME_BYTES == 320
    pieces = list(audio.frames(np.ones(400, dtype=np.int16)))  # 160 + 160 + 80 samples
    assert [len(p) for p in pieces] == [320, 320, 320]
    assert pieces[0] == pieces[1] == np.ones(160, dtype="<i2").tobytes()
    assert pieces[2] == np.ones(80, dtype="<i2").tobytes() + bytes(160)  # 80 samples, then zeros


def test_frames_accept_bytes_and_an_empty_signal():
    assert list(audio.frames(bytes(320))) == [bytes(320)]
    assert list(audio.frames(np.zeros(0, dtype=np.int16))) == []


async def test_pacing_schedules_every_frame_from_the_start_without_drift():
    clock = {"now": 100.0}
    slept = []

    async def sleep(seconds):
        slept.append(round(seconds, 6))
        clock["now"] += seconds + 0.003  # every sleep overshoots by 3 ms

    out = [
        f
        async for f in audio.paced(
            [b"a", b"b", b"c", b"d"], 0.02, sleep=sleep, now=lambda: clock["now"]
        )
    ]

    assert out == [b"a", b"b", b"c", b"d"]
    assert slept == [0.02, 0.017, 0.017]  # after the first, each wait absorbs the overshoot
    assert clock["now"] == pytest.approx(100.0 + 3 * 0.02 + 0.003)  # the last wait ended 3 ms late


async def test_pacing_can_run_faster_than_real_time():
    clock = {"now": 0.0}
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        clock["now"] += seconds

    [
        f
        async for f in audio.paced(
            [b"a", b"b", b"c"], 0.02, speed=2.0, sleep=sleep, now=lambda: clock["now"]
        )
    ]
    assert slept == pytest.approx([0.01, 0.01])


# --- The phone line ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", g711.CODECS)
def test_a_recording_keeps_its_length_and_level_through_the_phone_line(codec):
    out = audio.through_phone_line(tone(1000, 48000, 1.0), 48000, codec)
    assert out.dtype == np.int16 and len(out) == 8000
    assert rms(out[200:-200]) == pytest.approx(10000 / np.sqrt(2), rel=0.03)


def test_the_phone_line_cuts_what_a_phone_does_not_carry():
    low_hum = audio.through_phone_line(tone(50, 16000), 16000)
    high_hiss = audio.through_phone_line(tone(3950, 16000), 16000)
    assert rms(low_hum[200:-200]) < 0.05 * rms(tone(50, 16000))
    assert rms(high_hiss[200:-200]) < 0.3 * rms(tone(3950, 16000))


def test_what_comes_out_is_valid_g711():
    out = audio.through_phone_line(tone(700, 48000, 0.3), 48000, "alaw")
    assert np.array_equal(g711.roundtrip(out, "alaw"), out)


def test_the_channel_works_on_a_stream_in_any_chunks():
    signal = tone(800, 48000, 0.4) + tone(1900, 48000, 0.4, 2500)
    whole = audio.through_phone_line(signal, 48000)

    channel = audio.TelephoneChannel(48000)
    parts = [channel.process(signal[i : i + 960]) for i in range(0, len(signal), 960)]
    streamed = np.concatenate([*parts, channel.flush()])[channel.delay :]

    assert np.array_equal(streamed, whole)
    assert channel.delay == 50


def test_the_band_limit_can_be_switched_off():
    channel = audio.TelephoneChannel(8000, "ulaw", band_limit=False)
    assert channel.delay == 0
    out = channel.process(tone(50, 8000, 0.5))
    assert rms(out) > 0.9 * rms(tone(50, 8000, 0.5))


# --- Noise ------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr", [0, 5, 10, 20])
def test_noise_is_mixed_at_the_requested_signal_to_noise_ratio(snr):
    speech = tone(500, 8000, 1.0)
    noise = np.random.default_rng(1).normal(size=3000)  # shorter than the speech: it is repeated
    mixed = audio.mix_at_snr(speech, noise, snr)
    added = mixed - speech
    assert 20 * np.log10(rms(speech) / rms(added)) == pytest.approx(snr, abs=0.01)
    assert len(mixed) == len(speech)


def test_mixing_needs_a_noise_recording():
    with pytest.raises(audio.AudioError):
        audio.mix_at_snr(np.ones(10), np.zeros(0), 10)
