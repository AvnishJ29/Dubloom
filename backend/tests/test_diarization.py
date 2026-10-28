from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.speakers.diarization import speaker_turns
from backend.models import Segment


class DiarizationTest(unittest.TestCase):
    def test_public_speaker_model_and_asr_chunk_limit(self):
        with (
            patch(
                "backend.speakers.diarization.settings",
                SimpleNamespace(diarization_backend='speechbrain'),
            ),
            patch(
                "backend.speakers.diarization.diarize", return_value=[Segment(0, 52, "SPEAKER_00")]
            ),
        ):
            turns, mode = speaker_turns(Path("unused.wav"))
        self.assertEqual([(part.start, part.end) for part in turns], [(0, 25), (25, 50), (50, 52)])
        self.assertIn("SpeechBrain", mode)

    def test_long_turns_are_cut_at_the_quietest_pause(self):
        import numpy as np
        from backend.speakers.diarization import split_points

        rate = 16000
        samples = np.full(40 * rate, 0.3, dtype=np.float32)  # loud speech for 40 s ...
        samples[int(18.0 * rate) : int(18.6 * rate)] = 0.0  # ... with one pause at 18 s
        points = split_points(0.0, 40.0, samples, rate)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0], 18.3, delta=0.25)
        # Without audio the fixed 25 s rule still applies.
        self.assertEqual(split_points(0.0, 52.0, None), [25.0, 50.0])


if __name__ == "__main__":
    unittest.main()
