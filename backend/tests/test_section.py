"""Dubbing only a part of a video: the request is validated and the prepared source is cut to the part."""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from backend.api.helpers import validate_clip
from backend.jobs.stages.source import cut_section


class SectionTest(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(validate_clip(None, None), (None, None))
        self.assertEqual(validate_clip(10, 70), (10, 70))
        for start, end in (
            (10, None),
            (10, 10.5),
            (0, 10**6),
            (-1, 5),
            (0, float("inf")),
            (float("nan"), 5),
        ):
            with self.assertRaises(HTTPException):
                validate_clip(start, end)

    def test_cut_keeps_only_the_part(self):
        with TemporaryDirectory() as temp:
            video = Path(temp) / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=320x180:rate=25:duration=6",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=330:duration=6",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                check=True,
            )
            cut_section(video, 1.0, 3.5)
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(video),
                ],
                capture_output=True,
                text=True,
            ).stdout
            self.assertAlmostEqual(float(out), 2.5, delta=0.15)
            with self.assertRaises(Exception):
                cut_section(video, 2.0, 2.4)  # shorter than a second once clamped to the video


if __name__ == "__main__":
    unittest.main()


class LongVideoSectionTest(unittest.TestCase):
    """A one-minute section of a two-hour talk is a one-minute dub: the length limit applies to the section."""

    @staticmethod
    def fake_ytdl(directory, duration):
        class FakeYoutubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                return {"duration": duration, "title": "Long talk"}

            def download(self, urls):
                (directory / "download.mp4").write_bytes(b"\x00")

        return FakeYoutubeDL

    def test_a_section_of_a_long_video_is_allowed(self):
        import yt_dlp
        from backend.media import video_input

        with (
            TemporaryDirectory() as temp,
            patch.object(yt_dlp, "YoutubeDL", self.fake_ytdl(Path(temp), 7967)),
        ):
            path, title, meta, sectioned = video_input._download_with_ytdlp(
                "https://youtu.be/x", Path(temp), 7200, section=(1260.0, 1320.0)
            )
            self.assertTrue(path.is_file())
        self.assertTrue(sectioned)
        self.assertEqual(title, "Long talk")

    def test_the_whole_long_video_and_a_section_past_the_end_are_refused(self):
        import yt_dlp
        from backend.media import video_input

        with (
            TemporaryDirectory() as temp,
            patch.object(yt_dlp, "YoutubeDL", self.fake_ytdl(Path(temp), 7967)),
        ):
            with self.assertRaises(video_input.SourceError):
                video_input._download_with_ytdlp("https://youtu.be/x", Path(temp), 7200)
            with self.assertRaises(video_input.SourceError):
                video_input._download_with_ytdlp(
                    "https://youtu.be/x", Path(temp), 7200, section=(8000.0, 8060.0)
                )
