"""Large redrawn faces get their mouth region sharpened; small ones are left to the model."""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.rendering.mouth_enhance import enhance_lips, mouth_box


class MouthEnhanceTest(unittest.TestCase):
    def test_small_faces_are_skipped(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(
                root / "t.npz",
                bbox=np.array([[10, 10, 100, 120]] * 5, np.float32),
                track=np.array([0] * 5),
                lmk=np.zeros((5, 106, 2), np.float32),
            )
            report = enhance_lips(root / "lips.mp4", root / "t.npz", root / "out.mp4")
            self.assertFalse(report["applied"])

    def test_large_faces_are_sharpened_and_the_video_keeps_its_shape(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=640x480:rate=25:duration=0.4",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(root / "lips.mp4"),
                ],
                check=True,
            )
            n = 10
            np.savez(
                root / "t.npz",
                bbox=np.array([[100, 60, 420, 440]] * n, np.float32),
                track=np.array([0] * n),
                lmk=np.zeros((n, 106, 2), np.float32),
            )
            report = enhance_lips(root / "lips.mp4", root / "t.npz", root / "out.mp4", "cpu")
            self.assertTrue(report["applied"])
            self.assertEqual(report["frames"], n)
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames",
                    "-of",
                    "csv=p=0",
                    str(root / "out.mp4"),
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(out, "640,480,10")
            self.assertEqual(
                mouth_box(np.array([100, 60, 420, 440]), 640, 480), (68, 231, 452, 480)
            )


if __name__ == "__main__":
    unittest.main()
