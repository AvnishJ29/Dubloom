"""English speech written in an Indic script by the recogniser is caught against the reviewer's transcript."""

import unittest

from backend.speech.asr import dominant_script, guess_source_language, script_mismatch


class ScriptGuardTest(unittest.TestCase):
    def test_scripts_are_recognised(self):
        self.assertEqual(dominant_script("where lambda one will be replaced"), "latin")
        self.assertEqual(dominant_script("దిస్ కెన్ వీ కెన్ దిస్ వర్క్స్"), "indic")
        self.assertEqual(
            dominant_script("इस term को W1 transpose कहते हैं"), None
        )  # a real code-mixed line: no single script wins
        self.assertIsNone(dominant_script("W1"))

    def test_mismatch_only_when_both_sides_are_clear(self):
        self.assertTrue(script_mismatch("దిస్ కెన్ వీ కెన్ దిస్ వర్క్స్", "This can we can this works"))
        self.assertFalse(script_mismatch("इस term को समझो", "इस टर्म को समझो"))
        self.assertFalse(script_mismatch("W1 transpose C W1", "Now what is W1 transpose C W1"))

    def test_english_is_guessed_from_latin_transcripts(self):
        self.assertEqual(guess_source_language(["This can we can", "where lambda one", ""]), "en")
        self.assertIsNone(guess_source_language(["यह एक हिंदी वाक्य है", "और यह भी"]))


if __name__ == "__main__":
    unittest.main()
