"""Changing a line keeps its earlier words: who wrote them and when they were replaced."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


class LineHistoryTest(unittest.TestCase):
    def test_edits_by_the_ai_and_by_you_stack_up_and_nothing_is_lost(self):
        from fastapi.testclient import TestClient
        from backend import api
        from backend.jobs import pipeline

        with TemporaryDirectory() as temp:
            root = Path(temp)
            job_id = "22222222-2222-4222-8222-222222222222"
            folder = root / job_id
            folder.mkdir()
            (folder / "source.mp4").write_bytes(b"x")
            (folder / "speaker_turns.json").write_text("[]", encoding="utf-8")
            segments = [
                {
                    "start": 0,
                    "end": 2,
                    "speaker": "SPEAKER_00",
                    "text": "Hello there",
                    "translation": "नमस्ते जी",
                    "machine_translation": "नमस्ते",
                }
            ]
            (folder / "status.json").write_text(
                json.dumps(
                    {"id": job_id, "status": "completed", "language": "hi", "segments": segments}
                ),
                encoding="utf-8",
            )
            with (
                patch.object(api, "settings", SimpleNamespace(jobs_dir=root, studio_password="")),
                patch.object(pipeline, "settings", SimpleNamespace(jobs_dir=root)),
                patch.object(api, "start_job"),
            ):
                client = TestClient(api.app)
                response = client.post(
                    f"/api/jobs/{job_id}/script",
                    json={"lines": [{"index": 0, "translation": "नमस्ते दोस्तों", "via": "ai"}]},
                )
                self.assertEqual(response.status_code, 202, response.text)
                response = client.post(
                    f"/api/jobs/{job_id}/script",
                    json={"lines": [{"index": 0, "translation": "नमस्कार दोस्तों"}]},
                )
                self.assertEqual(response.status_code, 202, response.text)
                response = client.post(
                    f"/api/jobs/{job_id}/script",
                    json={"lines": [{"index": 0, "text": "Hello friends"}]},
                )
                self.assertEqual(response.status_code, 202, response.text)
                line = json.loads((folder / "status.json").read_text(encoding="utf-8"))["segments"][
                    0
                ]
        self.assertEqual(line["text"], "Hello friends")
        self.assertEqual(
            [(entry["field"], entry["by"], entry["text"]) for entry in line["history"]],
            [
                ("translation", "editor", "नमस्ते जी"),
                ("translation", "ai", "नमस्ते दोस्तों"),
                ("text", "bodhan", "Hello there"),
                ("translation", "you", "नमस्कार दोस्तों"),
            ],
        )
        self.assertTrue(all(entry["at"] for entry in line["history"]))


if __name__ == "__main__":
    unittest.main()
