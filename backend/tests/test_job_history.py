import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from backend.jobs import pipeline


class JobHistoryTest(unittest.TestCase):
    def test_history_is_newest_first_and_clear_refuses_active_jobs(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(pipeline, "settings", SimpleNamespace(jobs_dir=Path(temp))),
        ):
            store = pipeline.JobStore()
            store.save("old", status="completed", source_url="https://example.com/old")
            store.save("active", status="running")

            self.assertEqual([job["id"] for job in store.list()], ["active", "old"])
            with self.assertRaisesRegex(RuntimeError, "active dubbing jobs"):
                store.clear()

            store.save("active", status="completed")
            self.assertEqual(store.clear(), 2)
            self.assertEqual(store.list(), [])


if __name__ == "__main__":
    unittest.main()
