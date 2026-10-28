"""The finisher restores only the spans that need it and pastes cached spans into one final encode."""

import subprocess
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from backend.rendering import finish


def write_clip(path: Path, frames: list[np.ndarray]):
    h, w = frames[0].shape[:2]
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
            f"{w}x{h}",
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "8",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()


def read_clip(path: Path):
    reader, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        out.append(frame)
    return out


def solid(color, n, size=(64, 48)):
    return [np.full((size[1], size[0], 3), color, np.uint8) for _ in range(n)]


def targets(path: Path, widths: list[float]):
    n = len(widths)
    bbox = np.array([[0, 0, w, w] if w > 0 else [np.nan] * 4 for w in widths], np.float32)
    np.savez(
        path,
        frame_count=n,
        bbox=bbox,
        lmk=np.zeros((n, 106, 2), np.float32),
        track=np.array([0 if w > 0 else -1 for w in widths], np.int32),
    )


class PlanTest(unittest.TestCase):
    def test_each_span_gets_only_the_pass_its_faces_need(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            targets(root / "t.npz", [100] * 10 + [300] * 10 + [0] * 10)
            manifest = {
                "spans": [
                    {"first": 0, "end": 10, "key": "a", "file": str(root / "span_a.mp4")},
                    {"first": 10, "end": 20, "key": "b", "file": str(root / "span_b.mp4")},
                ]
            }
            plan = finish.plan_restoration(manifest, root / "t.npz", gfpgan=True, esrgan=True)
            self.assertEqual(
                [(p["gfpgan"], p["esrgan"]) for p in plan], [(True, False), (False, True)]
            )
            self.assertTrue(all(p["restored"] != p["raw"] and not p["cached"] for p in plan))
            self.assertIn(finish.restore_stamp(True, True), plan[0]["restored"].name)
            untouched = finish.plan_restoration(
                manifest, root / "t.npz", gfpgan=False, esrgan=False
            )
            self.assertTrue(
                all(p["restored"] == p["raw"] and p["cached"] for p in untouched)
            )  # nothing to do: raw spans go straight in


class AssembleTest(unittest.TestCase):
    def test_cached_spans_replace_the_picture_in_one_encode(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_clip(root / "render_source.mp4", solid((128, 128, 128), 12))
            write_clip(root / "span.mp4", solid((0, 0, 255), 4))
            with wave.open(str(root / "mix.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 8000)
            plan = [{"first": 4, "end": 8, "raw": root / "span.mp4", "restored": root / "span.mp4"}]
            finish.assemble(
                root / "render_source.mp4",
                plan,
                root / "mix.wav",
                root / "final.mp4",
                12,
                25.0,
                None,
            )
            frames = read_clip(root / "final.mp4")
            self.assertEqual(len(frames), 12)
            self.assertGreater(frames[5][..., 2].mean(), 200)  # red span frame pasted in
            self.assertLess(frames[5][..., 0].mean(), 60)
            self.assertTrue(
                100 < frames[2].mean() < 150 and 100 < frames[10].mean() < 150
            )  # picture before and after untouched


if __name__ == "__main__":
    unittest.main()
