"""A short is a vertical cut of the dub with the dubbed lines burned in, framed on the face when known."""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.media.shorts import face_crop_x, make_short, write_subtitles


class ShortsTest(unittest.TestCase):
    def test_subtitles_are_shifted_to_the_short(self):
        with TemporaryDirectory() as temp:
            srt = Path(temp) / "s.srt"
            segments = [
                {"start": 1, "end": 3, "translation": "पहली line"},
                {"start": 10, "end": 12, "dub_start": 10.5, "dub_end": 12.2, "translation": "दूसरी"},
                {"start": 30, "end": 31, "translation": "बाहर"},
            ]
            self.assertEqual(write_subtitles(segments, 2.0, 15.0, srt), 2)
            text = srt.read_text(encoding="utf-8")
            self.assertIn(
                "00:00:00,000 --> 00:00:01,000\nपहली line", text
            )  # clipped at the short's start
            self.assertIn(
                "00:00:08,500 --> 00:00:10,200\nदूसरी", text
            )  # placed where the dub actually plays

    def test_long_lines_become_short_cues(self):
        from backend.media.shorts import cues

        text = "मान लीजिए पहले time step के derivative के आधार पर, क्योंकि तब history शून्य थी, यह update हुआ, right? और मैं W1 पर गया, right?"
        pieces = cues(text)
        self.assertGreaterEqual(len(pieces), 3)
        self.assertTrue(all(len(p) <= 64 for p in pieces))
        self.assertEqual(" ".join(pieces), text)
        with TemporaryDirectory() as temp:
            srt = Path(temp) / "s.srt"
            n = write_subtitles([{"start": 0, "end": 24, "translation": text}], 0.0, 24.0, srt)
            self.assertEqual(n, len(pieces))

    def test_face_crop_follows_the_largest_face(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frames = np.array([0, 0, 10, 20])
            boxes = np.array(
                [
                    [1500, 100, 1700, 400],
                    [100, 100, 150, 150],
                    [1480, 90, 1690, 400],
                    [1520, 100, 1700, 390],
                ],
                dtype=np.float32,
            )
            np.savez(root / "faces.npz", frame=frames, bbox=boxes)
            self.assertEqual(
                face_crop_x(root, 0.0, 1.0, 1920, 1080), 1920 - 1080
            )  # the face is at the right edge: the window hugs it
            self.assertIsNone(face_crop_x(root, 5.0, 6.0, 1920, 1080))

    def test_short_is_vertical_and_subtitled(self):
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
                    "testsrc=size=640x360:rate=25:duration=3",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=330:duration=3",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(root / "dubbed_video.mp4"),
                ],
                check=True,
            )
            job = {
                "segments": [{"start": 0.2, "end": 2.5, "translation": "नमस्ते score-based models"}],
                "shorts": [],
            }
            short = make_short(job, root, 0.0, 2.0, "centre", True)
            self.assertEqual((short["number"], short["lines"], short["subtitles"]), (1, 1, True))
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0",
                    str(root / "shorts" / short["file"]),
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(out, "1080,1920")
            with self.assertRaises(ValueError):
                make_short(job, root, 0.0, 0.5, "centre", False)


if __name__ == "__main__":
    unittest.main()
