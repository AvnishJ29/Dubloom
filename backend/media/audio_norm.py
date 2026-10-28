"""Level every synthesised clip with the user's own TTS normaliser (backend/vendor/normalize_volume.py, the script
behind audio_generations_peak_normalized): a speech-gated AGC lifts the clip's speech to one fixed RMS target so
every line lands at the same level and nothing fades within a line, then a look-ahead peak limiter caps it at a
fixed headroom. Applied to every clip the voice engine returns and again to every trimmed clip at mix time, so
cached clips from older jobs are levelled too. Idempotent within rounding."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from backend.config import settings
from backend.vendor.normalize_volume import compute_agc_gain_db, peak_limit, undb

logger = logging.getLogger(__name__)

AGC_WINDOW = 4.0  # seconds of speech the level is measured over (the script's default)
AGC_SMOOTH = 1.0  # seconds of gain smoothing, so a line never pumps


def level_clip(
    path: Path, target_rms: float | None = None, peak_dbfs: float | None = None
) -> dict | None:
    """Rewrite the 16-bit PCM WAV at `path` levelled. Returns {'gain_db', 'peak_dbfs'} or None if untouched."""
    target_rms = settings.tts_target_rms if target_rms is None else target_rms
    peak_dbfs = settings.tts_peak_dbfs if peak_dbfs is None else peak_dbfs
    try:
        with wave.open(str(path), "rb") as reader:
            params = reader.getparams()
            if params.sampwidth != 2:
                return None
            raw = np.frombuffer(reader.readframes(params.nframes), dtype="<i2")
    except (wave.Error, EOFError, OSError):
        return None  # missing, or not a PCM WAV (another engine's format): left as is
    if raw.size == 0:
        return None
    y = raw.astype(np.float64) / 32768.0
    if params.nchannels > 1:
        y = y.reshape(-1, params.nchannels).mean(axis=1)
    before = float(np.sqrt(np.mean(y**2))) or 1e-9
    curve = compute_agc_gain_db(y, params.framerate, target_rms, AGC_WINDOW, None, None, AGC_SMOOTH)
    if curve is not None:  # None: no speech found, only the limiter applies
        y = y * undb(curve[0])
    y = peak_limit(y, peak_dbfs, params.framerate)
    out = np.clip(np.round(y * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setparams(params._replace(nchannels=1))
        writer.writeframes(out.tobytes())
    after = float(np.sqrt(np.mean(y**2))) or 1e-9
    return {"gain_db": round(20 * float(np.log10(after / before)), 2), "peak_dbfs": peak_dbfs}
