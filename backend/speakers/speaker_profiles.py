"""Per-speaker statistics (speech time, pitch, likely gender) from diarized turns.

The pitch estimate uses a normalised autocorrelation over voiced frames. It is
only used to prefer a matching-register synthetic voice; it never changes the
speaker labels themselves.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME = 640  # 40 ms
HOP = 320  # 20 ms
MIN_F0, MAX_F0 = 65.0, 400.0
MAX_ANALYSIS_SECONDS = 90.0


def _load(audio: Path) -> np.ndarray:
    with wave.open(str(audio), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            2,
        ):
            raise ValueError("Speaker analysis needs a 16 kHz mono PCM WAV")
        return (
            np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )


def median_pitch(samples: np.ndarray) -> tuple[float | None, float]:
    """Return (median F0 in Hz or None, voiced fraction)."""
    if samples.size < FRAME * 4:
        return None, 0.0
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    if not energy.size or energy.max() <= 1e-5:
        return None, 0.0
    loud = energy > max(0.02 * energy.max(), 0.01)
    frames = frames[loud]
    if not len(frames):
        return None, 0.0
    frames = frames * np.hanning(FRAME)[None, :]
    spectrum = np.fft.rfft(frames, n=2 * FRAME, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :FRAME]
    zero = autocorr[:, :1]
    zero[zero <= 0] = 1e-9
    autocorr = autocorr / zero
    low, high = int(SAMPLE_RATE / MAX_F0), int(SAMPLE_RATE / MIN_F0)
    window = autocorr[:, low : high + 1]
    peaks = window.argmax(axis=1)
    strength = window[np.arange(len(window)), peaks]
    voiced = strength > 0.55
    if voiced.sum() < 8:
        return None, float(voiced.mean()) if len(voiced) else 0.0
    lags = peaks[voiced] + low
    f0 = SAMPLE_RATE / lags
    return float(np.median(f0)), float(voiced.mean())


def classify(f0: float | None) -> tuple[str, float]:
    """(gender, confidence) from a median pitch in Hz; None gives ("unknown", 0.0).

    Below 150 Hz is male and above 175 Hz female, more confidently the further out; the
    overlapping band in between is split at 163 Hz with a low confidence of 0.35.
    """
    if f0 is None:
        return "unknown", 0.0
    if f0 < 150:
        return "male", min(1.0, (150 - f0) / 40 + 0.5)
    if f0 > 175:
        return "female", min(1.0, (f0 - 175) / 40 + 0.5)
    return "female" if f0 >= 163 else "male", 0.35


def speaker_profiles(audio: Path, turns) -> dict[str, dict]:
    """Summarise every speaker from its diarized turns."""
    samples = _load(audio)
    by_speaker: dict[str, list] = {}
    for turn in turns:
        by_speaker.setdefault(turn.speaker, []).append(turn)
    profiles = {}
    for speaker, items in by_speaker.items():
        items = sorted(items, key=lambda t: t.end - t.start, reverse=True)
        pieces, collected = [], 0.0
        for turn in items:
            start, end = int(turn.start * SAMPLE_RATE), int(turn.end * SAMPLE_RATE)
            clip = samples[max(0, start) : max(0, end)]
            if clip.size:
                pieces.append(clip)
                collected += clip.size / SAMPLE_RATE
            if collected >= MAX_ANALYSIS_SECONDS:
                break
        f0, voiced = median_pitch(np.concatenate(pieces)) if pieces else (None, 0.0)
        gender, confidence = classify(f0)
        profiles[speaker] = {
            "speech_seconds": round(sum(t.end - t.start for t in items), 2),
            "turns": len(items),
            "first_seen": round(min(t.start for t in items), 2),
            "pitch_hz": round(f0, 1) if f0 else None,
            "voiced_fraction": round(voiced, 2),
            "gender": gender,
            "gender_confidence": round(confidence, 2),
        }
    return profiles
