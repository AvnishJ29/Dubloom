"""Regressions from the 27 Oct review: each test fails on the code before the fix."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

JOB = "33333333-3333-4333-8333-333333333333"


def studio(root: Path):
    """A finished one-line job on disk and a test client whose job runner does nothing."""
    from fastapi.testclient import TestClient
    from backend.api import app, editing, helpers, jobs, system
    from backend.jobs import pipeline

    folder = root / JOB
    folder.mkdir()
    for name in ("source.mp4", "speaker_turns.json", "dubbed_video.mp4"):
        (folder / name).write_bytes(b"x")
    line = {
        "start": 0,
        "end": 2,
        "speaker": "SPEAKER_00",
        "text": "Hello there",
        "translation": "नमस्ते जी",
        "voiced_text": "नमस्ते जी",
    }
    (folder / "status.json").write_text(
        json.dumps(
            {
                "id": JOB,
                "status": "completed",
                "language": "hi",
                "title": "Talk",
                "source_url": "https://example.com/v",
                "segments": [line],
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        jobs_dir=root,
        studio_password="",
        mouth_enhance=True,
        lip_sync_model="latentsync",
        translation_script="codemix",
        background_mode="separate",
    )
    return TestClient(app), [
        patch.object(system, "settings", settings),
        patch.object(helpers, "settings", settings),
        patch.object(jobs, "settings", settings),
        patch.object(pipeline, "settings", SimpleNamespace(jobs_dir=root)),
        patch.object(jobs, "start_job"),
        patch.object(editing, "start_job"),
        patch.object(jobs, "check_ready"),
    ]


class ReviewFixesTest(unittest.TestCase):
    def test_a_re_render_keeps_each_lines_history(self):
        from backend.models import Segment
        from backend.jobs.pipeline import restore_line

        saved = {
            "text": "Hi",
            "translation": "नमस्कार",
            "edited": True,
            "ai_edited": True,
            "history": [{"at": "t", "field": "translation", "by": "bodhan", "text": "नमस्ते"}],
        }
        line = restore_line(Segment(0, 2, "SPEAKER_00"), saved)
        self.assertEqual(line.history, saved["history"])
        self.assertTrue(line.ai_edited and line.edited)

    def test_a_script_edit_archives_the_render_as_it_was_voiced(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            client, patches = studio(root)
            for p in patches:
                p.start()
            try:
                response = client.post(
                    f"/api/jobs/{JOB}/script",
                    json={"lines": [{"index": 0, "translation": "नमस्कार दोस्तों"}]},
                )
                self.assertEqual(response.status_code, 202, response.text)
            finally:
                for p in patches:
                    p.stop()
            archived = json.loads(
                (root / JOB / "versions" / "1" / "version.json").read_text(encoding="utf-8")
            )
        self.assertEqual(archived["segments"][0]["translation"], "नमस्ते जी")  # not the new words

    def test_correcting_the_original_clears_the_ai_mark(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            client, patches = studio(root)
            for p in patches:
                p.start()
            try:
                client.post(
                    f"/api/jobs/{JOB}/script",
                    json={"lines": [{"index": 0, "translation": "नमस्ते दोस्तों", "via": "ai"}]},
                )
                client.post(
                    f"/api/jobs/{JOB}/script",
                    json={"lines": [{"index": 0, "text": "Hello friends"}]},
                )
            finally:
                for p in patches:
                    p.stop()
            line = json.loads((root / JOB / "status.json").read_text(encoding="utf-8"))["segments"][
                0
            ]
        self.assertFalse(line["ai_edited"])

    def test_dub_again_reuses_the_saved_video_without_a_download(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            client, patches = studio(root)
            for p in patches:
                p.start()
            try:
                response = client.post("/api/jobs", json={"source_job": JOB, "language": "ta"})
                self.assertEqual(response.status_code, 202, response.text)
                new = response.json()["id"]
                job = json.loads((root / new / "status.json").read_text(encoding="utf-8"))
                self.assertTrue((root / new / "upload.mp4").is_file())
                self.assertEqual(
                    (job["source_type"], job["language"], job["clip_start"], job["reused_from"]),
                    ("upload", "ta", None, JOB),
                )
                self.assertEqual(
                    client.post("/api/jobs", json={"language": "ta"}).status_code, 422
                )  # neither a link nor a dub
            finally:
                for p in patches:
                    p.stop()

    def test_choosing_a_take_with_new_words_keeps_the_old_ones(self):
        from backend.jobs.takes import choose_take

        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "takes").mkdir()
            (folder / "takes" / "t1.wav").write_bytes(b"x")
            job = {
                "segments": [{"translation": "नमस्ते", "edited": False}],
                "takes": {
                    "0": [{"number": 1, "file": "t1.wav", "text": "नमस्कार", "style": "warm"}]
                },
            }
            line = choose_take(job, folder, 0, 1)
        self.assertEqual(
            [(h["field"], h["text"]) for h in line["history"]], [("translation", "नमस्ते")]
        )


class FreshDubTimingTest(unittest.TestCase):
    """27 Oct: a fresh dub failed at the mix with 'Object of type bool is not JSON serializable': its line times are
    NumPy floats straight from diarization, so the new 'last words cut' flag was a numpy.bool the job store rejected.
    """

    def test_the_timing_plan_returns_plain_python_values(self):
        import numpy as np
        from backend.media.timeline import plan_timeline

        plan = plan_timeline(
            [
                (np.float64(0.0), np.float64(1.0), np.float64(2.0)),
                (np.float64(1.2), np.float64(3.0), np.float64(1.0)),
            ],
            np.float64(5.0),
            exact=True,
        )
        self.assertTrue(all(type(p.trimmed) is bool and type(p.borrowed) is float for p in plan))

    def test_the_job_store_saves_numpy_values(self):
        import numpy as np
        from backend.jobs import pipeline

        with (
            TemporaryDirectory() as temp,
            patch.object(pipeline, "settings", SimpleNamespace(jobs_dir=Path(temp))),
        ):
            store = pipeline.JobStore()
            store.save(
                "job",
                segments=[{"timing_trimmed": np.bool_(True), "timing_borrowed": np.float64(0.25)}],
            )
            line = store.get("job")["segments"][0]
        self.assertEqual((line["timing_trimmed"], line["timing_borrowed"]), (True, 0.25))


class YoutubeFormatTest(unittest.TestCase):
    def test_hls_is_chosen_and_premium_skipped(self):
        """yt-dlp leaves format_note unset on HLS streams: a filter that rejected them sent every download to DASH."""
        from yt_dlp import YoutubeDL
        from backend.media.video_input import youtube_format

        formats = [
            {
                "format_id": "96",
                "protocol": "m3u8_native",
                "vcodec": "avc1",
                "acodec": "none",
                "height": 1080,
                "ext": "mp4",
                "url": "u",
            },
            {
                "format_id": "616",
                "protocol": "m3u8_native",
                "vcodec": "vp9",
                "acodec": "none",
                "height": 1080,
                "ext": "mp4",
                "format_note": "Premium",
                "tbr": 9000,
                "url": "u",
            },
            {
                "format_id": "233",
                "protocol": "m3u8_native",
                "vcodec": "none",
                "acodec": "mp4a",
                "ext": "mp4",
                "url": "u",
            },
            {
                "format_id": "137",
                "protocol": "https",
                "vcodec": "avc1",
                "acodec": "none",
                "height": 1080,
                "ext": "mp4",
                "format_note": "1080p",
                "url": "u",
            },
            {
                "format_id": "140",
                "protocol": "https",
                "vcodec": "none",
                "acodec": "mp4a",
                "ext": "m4a",
                "format_note": "medium",
                "url": "u",
            },
        ]
        with YoutubeDL({"quiet": True}) as ydl:
            selector = ydl.build_format_selector(youtube_format(2160))
            chosen = list(
                selector(
                    {"formats": formats, "has_merged_format": False, "incomplete_formats": False}
                )
            )
        self.assertEqual(chosen[0]["format_id"], "96+233")

    def test_a_refused_part_download_never_pulls_an_overlong_whole_video(self):
        import yt_dlp
        from yt_dlp.utils import DownloadError
        from backend.media import video_input

        class Site:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                return {"duration": 3 * 3600, "title": "Long"}

            def download(self, urls):
                if "download_ranges" in self.options:
                    raise DownloadError("ranges not supported")
                raise AssertionError("the whole video must not be fetched")

        with TemporaryDirectory() as temp, patch.object(yt_dlp, "YoutubeDL", Site):
            with self.assertRaises(video_input.SourceError):
                video_input._download_with_ytdlp(
                    "https://example.com/v", Path(temp), 7200, section=(60.0, 120.0)
                )


if __name__ == "__main__":
    unittest.main()
