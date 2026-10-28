"""A re-render archives the finished dub instead of overwriting it; nothing is archived from an unfinished job."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from backend.jobs.versions import archive, describe_revoice


class ArchiveTest(unittest.TestCase):
    def test_finished_render_moves_into_the_next_version(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in (
                "dubbed_video.mp4",
                "preview_video.mp4",
                "dubbed_audio.wav",
                "lip_sync_report.json",
                "tts_0000.wav",
            ):
                (root / name).write_bytes(b"x")
            job = {
                "status": "completed",
                "finished_at": "2026-10-16T20:00:00+00:00",
                "speaker_voices": {"SPEAKER_00": "Amit"},
                "lip_sync_model": "latentsync",
                "segments": [
                    {"start": 0, "end": 1, "speaker": "SPEAKER_00", "translation": "नमस्ते"}
                ],
                "lip_sync": {
                    "model": "LatentSync 1.6",
                    "lines_lip_synced": 1,
                    "coverage_percent": 100.0,
                },
            }
            versions = archive(job, root, "Speaker 1: voice Amit → Kavya")
            self.assertEqual([v["number"] for v in versions], [1])
            self.assertFalse(
                (root / "dubbed_video.mp4").exists()
            )  # moved, so the re-render cannot clobber it
            self.assertTrue((root / "versions" / "1" / "dubbed_video.mp4").is_file())
            self.assertTrue(
                (root / "tts_0000.wav").is_file()
            )  # voices stay where the re-render reuses them
            record = json.loads(
                (root / "versions" / "1" / "version.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["segments"][0]["translation"], "नमस्ते")
            self.assertEqual(record["settings"]["speaker_voices"], {"SPEAKER_00": "Amit"})
            self.assertEqual(versions[0]["reason"], "Speaker 1: voice Amit → Kavya")
            # a second finished render becomes version 2 and version 1 is untouched
            (root / "dubbed_video.mp4").write_bytes(b"y")
            job["versions"] = versions
            versions = archive(job, root, "Script edited: 1 line")
            self.assertEqual([v["number"] for v in versions], [1, 2])
            self.assertTrue((root / "versions" / "1" / "dubbed_video.mp4").is_file())

    def test_unfinished_job_archives_nothing(self):
        with TemporaryDirectory() as temp:
            self.assertEqual(
                archive({"status": "failed", "versions": [{"number": 1}]}, Path(temp), "x"),
                [{"number": 1}],
            )
            self.assertFalse((Path(temp) / "versions").exists())

    def test_reason_names_the_speaker_and_the_change(self):
        job = {
            "speaker_voices": {"SPEAKER_01": "Amit"},
            "speaker_profiles": {"SPEAKER_01": {"name": "Doom"}},
            "lip_sync_model": "latentsync",
        }
        request = SimpleNamespace(
            speaker_voices={"SPEAKER_01": "Sagar"},
            speaker_styles={},
            delivery_style=None,
            lip_sync_model="wav2lip",
            background_mode=None,
            tts_engine=None,
        )
        self.assertEqual(
            describe_revoice(job, request), "Doom: voice Amit → Sagar; lip sync → wav2lip"
        )


if __name__ == "__main__":
    unittest.main()


class RenameTest(unittest.TestCase):
    def test_rename_changes_only_the_title(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from backend import api

        with TemporaryDirectory() as temp:
            root = Path(temp)
            job_id = "11111111-1111-4111-8111-111111111111"
            (root / job_id).mkdir()
            (root / job_id / "status.json").write_text(
                json.dumps({"id": job_id, "status": "completed", "title": "Old", "language": "hi"}),
                encoding="utf-8",
            )
            from backend.jobs import pipeline

            with (
                patch.object(api, "settings", SimpleNamespace(jobs_dir=root, studio_password="")),
                patch.object(pipeline, "settings", SimpleNamespace(jobs_dir=root)),
            ):
                client = TestClient(api.app)
                response = client.patch(f"/api/jobs/{job_id}", json={"title": "  New name "})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    json.loads((root / job_id / "status.json").read_text(encoding="utf-8"))[
                        "title"
                    ],
                    "New name",
                )


class MergeTest(unittest.TestCase):
    def test_merge_relabels_lines_and_cached_turns(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from backend import api
        from backend.jobs import pipeline

        with TemporaryDirectory() as temp:
            root = Path(temp)
            job_id = "22222222-2222-4222-8222-222222222222"
            (root / job_id).mkdir()
            record = {
                "id": job_id,
                "status": "completed",
                "language": "hi",
                "segments": [
                    {"start": 0, "end": 1, "speaker": "S0", "voice": "Amit", "translation": "a"},
                    {"start": 2, "end": 3, "speaker": "S0b", "voice": "Kavya", "translation": "b"},
                ],
                "speaker_voices": {"S0": "Amit", "S0b": "Kavya"},
                "speaker_profiles": {
                    "S0": {"name": "Man 1", "speech_seconds": 1, "turns": 1},
                    "S0b": {
                        "name": "Second voice (female)",
                        "speech_seconds": 1,
                        "turns": 1,
                        "split_from": "S0",
                    },
                },
            }
            (root / job_id / "status.json").write_text(json.dumps(record), encoding="utf-8")
            (root / job_id / "speaker_turns.json").write_text(
                json.dumps(
                    {
                        "segments": [
                            {"start": 0, "end": 1, "speaker": "S0"},
                            {"start": 2, "end": 3, "speaker": "S0b"},
                        ],
                        "profiles": dict(record["speaker_profiles"]),
                        "split_speakers": {"S0b": "S0"},
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(api, "settings", SimpleNamespace(jobs_dir=root, studio_password="")),
                patch.object(pipeline, "settings", SimpleNamespace(jobs_dir=root)),
                patch.object(api, "start_job", lambda *a, **k: None),
            ):
                response = TestClient(api.app).post(
                    f"/api/jobs/{job_id}/speakers/S0b/merge", json={"into": "S0"}
                )
                self.assertEqual(response.status_code, 202, response.text)
            saved = json.loads((root / job_id / "status.json").read_text(encoding="utf-8"))
            self.assertEqual([s["speaker"] for s in saved["segments"]], ["S0", "S0"])
            self.assertEqual([s["voice"] for s in saved["segments"]], ["Amit", "Amit"])
            self.assertEqual(saved["speaker_voices"], {"S0": "Amit"})
            self.assertEqual(saved["speaker_profiles"]["S0"]["turns"], 2)
            turns = json.loads((root / job_id / "speaker_turns.json").read_text(encoding="utf-8"))
            self.assertEqual({t["speaker"] for t in turns["segments"]}, {"S0"})
            self.assertNotIn("S0b", turns["profiles"])


class HiddenBaselineTest(unittest.TestCase):
    def test_one_time_reset_keeps_files_and_future_history_visible(self):
        from backend.jobs.versions import hide_history_baseline

        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'dubbed_video.mp4').write_bytes(b'old')
            job = {'status': 'completed', 'segments': []}
            versions = hide_history_baseline(archive(job, root, 'Timing refresh'), root)
            self.assertTrue(versions[0]['hidden_from_history'])
            self.assertEqual((root / 'versions/1/dubbed_video.mp4').read_bytes(), b'old')
            self.assertTrue(
                json.loads((root / 'versions/1/version.json').read_text())['hidden_from_history']
            )
            (root / 'dubbed_video.mp4').write_bytes(b'new')
            job['versions'] = versions
            versions = archive(job, root, 'Changed speaker voice')
            self.assertTrue(versions[0]['hidden_from_history'])
            self.assertFalse(versions[1].get('hidden_from_history', False))


class RerenderReasonTest(unittest.TestCase):
    def test_an_empty_re_render_says_what_it_did(self):
        empty = SimpleNamespace(
            speaker_voices={},
            speaker_styles={},
            delivery_style=None,
            lip_sync_model=None,
            background_mode=None,
            tts_engine=None,
        )
        self.assertEqual(describe_revoice({"pending_takes": 2}, empty), "Takes applied to 2 lines")
        self.assertEqual(describe_revoice({}, empty), "Improved lip sync and timing")
