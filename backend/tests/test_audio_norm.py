"""Every synthesised clip leaves at the same loudness and peak, whatever level the engine returned it at."""

import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.media.audio_norm import level_clip


def write_tone(path: Path, amplitude: float, seconds: float = 1.5, rate: int = 24000):
    t = np.arange(int(seconds * rate)) / rate
    y = (amplitude * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(np.round(y * 32767), -32768, 32767).astype("<i2").tobytes())


def stats(path: Path):
    with wave.open(str(path), "rb") as r:
        y = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return 20 * np.log10(np.abs(y).max()), 20 * np.log10(np.sqrt(np.mean(y**2)))


class LevelClipTest(unittest.TestCase):
    def test_quiet_and_loud_clips_end_up_alike(self):
        with TemporaryDirectory() as temp:
            quiet, loud = Path(temp) / "q.wav", Path(temp) / "l.wav"
            write_tone(quiet, 0.05)
            write_tone(loud, 0.9)
            self.assertIsNotNone(level_clip(quiet))
            self.assertIsNotNone(level_clip(loud))
            (pq, rq), (pl, rl) = stats(quiet), stats(loud)
            self.assertAlmostEqual(rq, rl, delta=0.3)  # same speech level within rounding
            self.assertLessEqual(max(pq, pl), -1.0 + 0.05)  # never above the limiter's ceiling
            self.assertAlmostEqual(level_clip(loud)["gain_db"], 0.0, delta=0.1)  # idempotent

    def test_fading_line_is_held_level(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "f.wav"
            rate = 24000
            t = np.arange(int(12 * rate)) / rate
            fade = np.linspace(1.0, 0.25, t.size)  # speech that falls flat towards the end
            y = (
                0.5 * fade * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            ).astype(np.float32)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(np.clip(np.round(y * 32767), -32768, 32767).astype("<i2").tobytes())
            level_clip(path)
            with wave.open(str(path), "rb") as r:
                out = (
                    np.frombuffer(r.readframes(r.getnframes()), dtype="<i2").astype(np.float32)
                    / 32768
                )
            head, tail = out[: out.size // 10], out[-out.size // 10 :]
            drift = 20 * np.log10(np.sqrt(np.mean(tail**2)) / np.sqrt(np.mean(head**2)))
            self.assertGreater(drift, -3.0)  # 12 dB fade in, under 3 dB out

    def test_silence_is_left_alone(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "s.wav"
            write_tone(path, 0.0)
            level_clip(path)
            self.assertLess(stats(path)[1], -80)


if __name__ == "__main__":
    unittest.main()
