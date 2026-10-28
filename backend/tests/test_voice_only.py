"""A voice-only dub muxes the mix onto the untouched picture in seconds and reports no lip sync."""

import json
import subprocess
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.rendering.lip_models import checkpoint_for, LIP_SYNC_MODELS
from backend.rendering.lip_sync import voice_only


class VoiceOnlyTest(unittest.TestCase):
    def test_model_registry_knows_none(self):
        self.assertIn("none", LIP_SYNC_MODELS)
        self.assertEqual(checkpoint_for("none")[0], "none")

    def test_mux_keeps_every_frame_and_reports_no_sync(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            enc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    "64x48",
                    "-r",
                    "25",
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(root / "source.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
            for _ in range(25):
                enc.stdin.write(np.full((48, 64, 3), 90, np.uint8).tobytes())
            enc.stdin.close()
            enc.wait()
            with wave.open(str(root / "mix.wav"), "wb") as w:
                tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 32767).astype(
                    "<i2"
                )
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(tone.tobytes())
            final = voice_only(root / "source.mp4", root / "mix.wav", root)
            self.assertTrue(final.is_file())
            report = json.loads((root / "lip_sync_report.json").read_text())
            self.assertEqual(report["engine"], "none")
            self.assertEqual(report["frames"], 25)
            self.assertIsNone(report["lines_lip_synced"])
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(final),
                ],
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("video", probe)
            self.assertIn("audio", probe)


if __name__ == "__main__":
    unittest.main()
