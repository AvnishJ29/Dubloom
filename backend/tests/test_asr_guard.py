"""A recogniser that loops on one token is retried, then refused, never translated."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from backend.speech import asr


class FakeApi:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(json=lambda: {"text": self.answers.pop(0)})


class DegenerateTest(unittest.TestCase):
    def test_detects_token_loops_only(self):
        self.assertTrue(asr.degenerate("ୄ" * 70))
        self.assertTrue(asr.degenerate("The King Khalid Airport in Riyadh the Ar" + "ୄ" * 60))
        self.assertFalse(asr.degenerate("Saudi Arabia is stuck in a loop."))
        self.assertFalse(asr.degenerate("ok"))

    def test_retries_then_refuses(self):
        with TemporaryDirectory() as temp, patch.object(asr, "wav_seconds", lambda p: 1.0):
            audio = Path(temp) / "a.wav"
            audio.write_bytes(b"RIFF")
            api = FakeApi(["ୄ" * 40, "Saudi Arabia is stuck in a loop"])
            with self.assertLogs(asr.logger, "WARNING") as logs:
                self.assertEqual(
                    asr.transcribe(api, audio, "en", None, {}), "Saudi Arabia is stuck in a loop"
                )
            self.assertEqual(api.calls, 2)
            self.assertIn("retrying", logs.output[0])
            api = FakeApi(["ୄ" * 40] * 3)
            with (
                self.assertRaises(asr.DegenerateTranscript),
                self.assertLogs(asr.logger, "WARNING"),
            ):
                asr.transcribe(api, audio, "en", None, {})
            self.assertEqual(api.calls, 3)


if __name__ == "__main__":
    unittest.main()
