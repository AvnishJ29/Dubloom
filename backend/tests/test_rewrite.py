"""The AI script change rewrites only the allowed lines, in context, and reports just what changed."""

import unittest
from unittest.mock import patch

from backend.speech import script_supervisor


SEGMENTS = [
    {"speaker": "SPEAKER_00", "text": "Hello there.", "translation": "नमस्ते।"},
    {"speaker": "SPEAKER_00", "text": "Thank you for coming.", "translation": "आने के लिए धन्यवाद।"},
    {"speaker": "SPEAKER_01", "text": "My pleasure.", "translation": "मेरी खुशी।"},
]


class RewriteTest(unittest.TestCase):
    def test_only_requested_lines_change_and_unchanged_ones_are_dropped(self):
        seen = {}

        def fake_ask(system, user, expected, usage):
            seen["system"], seen["user"], seen["expected"] = system, user, expected
            return {1: "आने के लिए बहुत-बहुत धन्यवाद।", 2: "मेरी खुशी।"}  # 2 comes back unchanged

        with (
            patch.object(script_supervisor, "_ask", fake_ask),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
        ):
            changed = script_supervisor.rewrite_lines(
                SEGMENTS, "hi", "en", 'make it warmer', only={1, 2}
            )
        self.assertEqual(changed, {1: "आने के लिए बहुत-बहुत धन्यवाद।"})
        self.assertEqual(seen["expected"], {1, 2})
        self.assertIn("make it warmer", seen["system"])
        self.assertIn("Only lines 1, 2 may change", seen["user"])
        self.assertIn("Hindi", seen["system"])

    def test_nothing_without_a_key(self):
        with patch.object(script_supervisor.openrouter, "configured", lambda: False):
            self.assertEqual(script_supervisor.rewrite_lines(SEGMENTS, "hi", "en", "shorter"), {})


if __name__ == "__main__":
    unittest.main()


class SourceRewriteTest(unittest.TestCase):
    def test_rewrites_the_original_transcript_when_asked(self):
        seen = {}

        def fake_ask(system, user, expected, usage):
            seen["system"], seen["user"] = system, user
            return {0: "Hello there, everyone."}

        with (
            patch.object(script_supervisor, "_ask", fake_ask),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
        ):
            changed = script_supervisor.rewrite_lines(
                SEGMENTS, "hi", "en", "address the audience", field="text"
            )
        self.assertEqual(changed, {0: "Hello there, everyone."})
        self.assertIn("original English transcript", seen["system"])
        self.assertIn("Lines (id | speaker | text)", seen["user"])
