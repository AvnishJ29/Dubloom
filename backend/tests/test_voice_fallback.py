"""Bodhan voices first; OpenRouter voices take over for the whole project when Bodhan cannot answer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.voices import engine as voices
from backend.providers.bodhan_api import BodhanBudgetError
from backend.voices.engine import VoiceState


class VoiceFallbackTest(unittest.TestCase):
    def test_bodhan_budget_error_switches_every_following_line_to_studio_voices(self):
        calls = []

        def bodhan(client, text, language, voice, output, style=None, usage=None):
            calls.append(("bodhan", voice))
            raise BodhanBudgetError("over budget")

        def gemini(text, voice, output, style=None, usage=None):
            calls.append(("gemini", voice))
            Path(output).write_bytes(b"RIFF")

        with (
            TemporaryDirectory() as temp,
            patch.object(voices, "bodhan_synthesize", bodhan),
            patch.object(voices, "gemini_synthesize", gemini),
            patch.object(voices.openrouter, "configured", return_value=True),
            patch.object(voices.openrouter, "covers_bodhan", return_value=True),
        ):  # the fallback is allowed (BODHAN_ONLY off)
            state = VoiceState(
                client=None,
                language="hi",
                speaker_voices={"SPEAKER_00": "Amit", "SPEAKER_01": "Kavya"},
            )
            self.assertEqual(
                state.render("SPEAKER_00", "नमस्ते", Path(temp) / "a.wav", None), "gemini"
            )
            self.assertEqual(
                state.render("SPEAKER_01", "नमस्ते", Path(temp) / "b.wav", None), "gemini"
            )
        self.assertEqual(calls, [("bodhan", "Amit"), ("gemini", "Charon"), ("gemini", "Kore")])
        self.assertEqual(state.labels()["SPEAKER_00"], "studio voice Charon")

    def test_without_openrouter_the_bodhan_error_surfaces(self):
        def bodhan(client, text, language, voice, output, style=None, usage=None):
            raise BodhanBudgetError("over budget")

        with (
            TemporaryDirectory() as temp,
            patch.object(voices, "bodhan_synthesize", bodhan),
            patch.object(voices.openrouter, "configured", return_value=False),
        ):
            state = VoiceState(client=None, language="hi", speaker_voices={"SPEAKER_00": "Amit"})
            with self.assertRaises(BodhanBudgetError):
                state.render("SPEAKER_00", "नमस्ते", Path(temp) / "a.wav", None)


if __name__ == "__main__":
    unittest.main()


class BodhanOnlyTest(unittest.TestCase):
    def test_bodhan_only_keeps_the_error_instead_of_switching(self):
        def bodhan(client, text, language, voice, output, style=None, usage=None):
            raise BodhanBudgetError("over budget")

        with (
            TemporaryDirectory() as temp,
            patch.object(voices, "bodhan_synthesize", bodhan),
            patch.object(voices.openrouter, "configured", return_value=True),
            patch.object(voices.openrouter, "covers_bodhan", return_value=False),
        ):
            state = VoiceState(client=None, language="hi", speaker_voices={"SPEAKER_00": "Amit"})
            with self.assertRaises(BodhanBudgetError):
                state.render("SPEAKER_00", "नमस्ते", Path(temp) / "a.wav", None)
