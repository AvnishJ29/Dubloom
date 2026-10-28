"""Director takes are voiced on demand and a chosen take becomes the line's clip without re-voicing it."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.jobs import takes


class TakesTest(unittest.TestCase):
    def job(self, root):
        return {
            "language": "hi",
            "speaker_voices": {"S0": "Amit"},
            "segments": [
                {
                    "start": 1.0,
                    "end": 3.5,
                    "speaker": "S0",
                    "translation": "नमस्ते दोस्तों",
                    "voiced_text": "नमस्ते दोस्तों",
                }
            ],
        }

    def test_generate_then_choose(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            job = self.job(root)

            def fake_synth(client, text, language, voice, output, style=None, usage=None):
                Path(output).write_bytes(b"RIFF" + style.encode())

            with (
                patch.object(takes, "synthesize", fake_synth),
                patch.object(takes, "_seconds", lambda p: 2.4),
                patch.object(takes, "BodhanAPI", lambda: None),
            ):
                take = takes.generate_take(job, root, 0, "angry", None)
                self.assertEqual(
                    (take["number"], take["style"], take["voice"], take["slot_seconds"]),
                    (1, "angry", "Amit", 2.5),
                )
                job["takes"] = {"0": [take]}
                second = takes.generate_take(job, root, 0, "neutral", "नमस्ते सब लोग")
                self.assertEqual((second["number"], second["text"]), (2, "नमस्ते सब लोग"))
                job["takes"]["0"].append(second)
            line = takes.choose_take(job, root, 0, 2)
            self.assertEqual(
                (line["translation"], line["style_override"], line["voiced_style"], line["edited"]),
                ("नमस्ते सब लोग", "neutral", "neutral", True),
            )
            self.assertEqual((root / "tts_0000.wav").read_bytes(), b"RIFFneutral")

    def test_unknown_take_is_refused(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                takes.choose_take(self.job(Path(temp)), Path(temp), 0, 9)


if __name__ == "__main__":
    unittest.main()
