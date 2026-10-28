"""Ultra-HD exports get a 1080p preview for the page player; smaller ones are played as they are."""

import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.media.video_input import probe, write_preview


def _clip(path: Path, height: int):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={height * 16 // 9}x{height}:rate=25:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed")
class PreviewTest(unittest.TestCase):
    def test_tall_master_gets_a_1080p_streamable_preview(self):
        with TemporaryDirectory() as folder:
            master, preview = Path(folder) / "master.mp4", Path(folder) / "preview.mp4"
            _clip(master, 1440)
            self.assertEqual(write_preview(master, preview), preview)
            info = probe(preview)
            self.assertEqual(
                (info["height"], info["video_codec"], info["audio_codec"]), (1080, "h264", "aac")
            )
            self.assertIn(
                b"moov", preview.read_bytes()[:4096]
            )  # index first, so playback starts before the download ends

    def test_small_master_needs_no_preview(self):
        with TemporaryDirectory() as folder:
            master, preview = Path(folder) / "master.mp4", Path(folder) / "preview.mp4"
            _clip(master, 720)
            preview.write_bytes(b"stale")
            self.assertIsNone(write_preview(master, preview))
            self.assertFalse(preview.exists())


if __name__ == "__main__":
    unittest.main()


class PortraitPreviewTest(unittest.TestCase):
    def test_a_portrait_phone_video_is_not_downscaled(self):
        import subprocess
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from backend.media.video_input import write_preview

        with TemporaryDirectory() as temp:
            master = Path(temp) / "m.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=1080x1920:rate=25:duration=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=duration=1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(master),
                ],
                check=True,
            )
            self.assertIsNone(write_preview(master, Path(temp) / "p.mp4"))
            self.assertFalse((Path(temp) / "p.mp4").exists())
