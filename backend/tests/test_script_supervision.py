"""Speaker review voting and script supervision change only what they should."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.speech import script_supervisor
from backend.speakers import speaker_review
from backend.models import Segment


class SpeakerReviewVotingTest(unittest.TestCase):
    def test_turns_are_grouped_where_most_reviews_agree_and_low_pitch_means_male(self):
        turns = [
            Segment(0, 2, "SPEAKER_00"),
            Segment(3, 5, "SPEAKER_01"),
            Segment(6, 8, "SPEAKER_02"),
            Segment(9, 11, "SPEAKER_03"),
        ]
        # Reviews 1 and 2 say turns 0 and 2 are one voice; only review 1 merges turn 3 into it.
        answers = [
            {
                "voices": [
                    {"id": "A", "name": "Doom", "gender": "female"},
                    {"id": "B", "name": "Man 1", "gender": "male"},
                ],
                "turns": [
                    {"id": 0, "voice": "A", "emotion": "angry"},
                    {"id": 1, "voice": "B"},
                    {"id": 2, "voice": "A"},
                    {"id": 3, "voice": "A"},
                ],
            },
            {
                "voices": [
                    {"id": "A", "name": "Doom", "gender": "male"},
                    {"id": "B", "name": "Man 1", "gender": "male"},
                    {"id": "C", "name": "Man 2", "gender": "male"},
                ],
                "turns": [
                    {"id": 0, "voice": "A", "emotion": "angry"},
                    {"id": 1, "voice": "B"},
                    {"id": 2, "voice": "A"},
                    {"id": 3, "voice": "C"},
                ],
            },
            {
                "voices": [
                    {"id": "X", "name": "Villain", "gender": "female"},
                    {"id": "Y", "name": "Man 1", "gender": "male"},
                    {"id": "Z", "name": "Man 2", "gender": "male"},
                ],
                "turns": [
                    {"id": 0, "voice": "X", "emotion": "neutral"},
                    {"id": 1, "voice": "Y"},
                    {"id": 2, "voice": "Z"},
                    {"id": 3, "voice": "W"},
                ],
            },
        ]
        replies = iter(json.dumps(a) for a in answers)
        with (
            TemporaryDirectory() as temp,
            patch.object(
                speaker_review, "_transcripts", lambda *a, **k: ["one", "two", "three", "four"]
            ),
            patch.object(speaker_review, "_pitches", lambda *a, **k: [100.0, 120.0, 105.0, 150.0]),
            patch.object(speaker_review.openrouter, "configured", lambda: True),
            patch.object(speaker_review.openrouter, "chat", lambda *a, **k: next(replies)),
            patch.object(speaker_review.subprocess, "run", lambda *a, **k: None),
            patch.object(Path, "read_bytes", lambda self: b"mp3"),
        ):
            reviewed, report = speaker_review.review_speakers(
                turns, Path(temp) / "a.wav", Path(temp)
            )
        labels = [t.speaker for t in reviewed]
        self.assertEqual(labels[0], labels[2])  # 2 of 3 reviews agree
        self.assertNotEqual(labels[0], labels[3])  # only 1 of 3 merged turn 3
        self.assertEqual(reviewed[0].style, "angry")
        doom = report["voices"][labels[0]]
        self.assertEqual(doom["name"], "Doom")
        self.assertEqual(doom["gender"], "male")  # 100-105 Hz overrules "female"


class PostEditTest(unittest.TestCase):
    def test_only_new_lines_are_edited_so_voiced_lines_keep_their_audio(self):
        lines = [
            Segment(0, 2, "A", text="For I am", translation="इससे पहले कि मैं हूँ"),
            Segment(3, 4, "A", text="Doom.", translation="विनाश"),
        ]
        lines[0].voiced_text = lines[0].translation
        captured = {}

        def fake_ask(system, user, expected, usage):
            captured["expected"] = expected
            return {0: "क्योंकि मैं हूँ", 1: "डूम।"}

        from backend.speech import fitting

        # the meaning check is a live model call: stubbed so the test is about which lines may change, and repeatable
        with (
            patch.object(script_supervisor, "_ask", fake_ask),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(
                fitting, "verify_rewrites", lambda items, *a, **k: {item["id"] for item in items}
            ),
        ):
            changed = script_supervisor.post_edit(lines, "hi", "en", only=[lines[1]])
        self.assertEqual(changed, 1)
        self.assertEqual(captured["expected"], {1})
        self.assertEqual(lines[0].translation, "इससे पहले कि मैं हूँ")
        self.assertEqual(lines[1].translation, "डूम।")
        self.assertEqual(lines[1].machine_translation, "विनाश")


if __name__ == "__main__":
    unittest.main()
