"""Turn boundaries snap to real speech and split speakers are merged back."""

import unittest

import numpy as np

from backend.speakers.local_diarization import consolidate_speakers, snap_boundaries
from backend.models import Segment


def speech(samples, start, end, rate=16000, level=0.3):
    rng = np.random.default_rng(1)
    samples[int(start * rate) : int(end * rate)] = level * rng.standard_normal(
        int((end - start) * rate)
    ).astype(np.float32)


class BoundaryTest(unittest.TestCase):
    def test_turn_starts_move_to_the_speech_onset(self):
        samples = np.zeros(16000 * 12, dtype=np.float32)
        speech(samples, 2.0, 4.0)  # the person actually speaks from 2.0 to 4.0 s
        speech(samples, 6.0, 8.5)
        turns = [
            Segment(1.75, 4.3, "A"),
            Segment(5.7, 8.2, "B"),
        ]  # diarization was early on both starts
        snapped = snap_boundaries(turns, samples)
        self.assertAlmostEqual(snapped[0].start, 2.0, delta=0.08)
        self.assertAlmostEqual(snapped[0].end, 4.0, delta=0.15)
        self.assertAlmostEqual(snapped[1].start, 6.0, delta=0.08)
        self.assertAlmostEqual(snapped[1].end, 8.5, delta=0.15)
        self.assertLess(snapped[0].end, snapped[1].start)


class ConsolidationTest(unittest.TestCase):
    def test_same_voice_split_in_two_is_merged_and_relabelled(self):
        vectors = {
            "RAW_0": np.array([1.0, 0.0, 0.0]),
            "RAW_1": np.array([0.97, 0.24, 0.0]),
            "RAW_2": np.array([0.0, 1.0, 0.0]),
        }
        turns = [
            Segment(0, 5, "RAW_0"),
            Segment(5.5, 9, "RAW_2"),
            Segment(10, 14, "RAW_1"),
            Segment(15, 20, "RAW_0"),
        ]
        samples = np.zeros(16000 * 21, dtype=np.float32)
        merged = consolidate_speakers(
            turns,
            samples,
            embed=lambda s, a, b: vectors[[t for t in turns if t.start == a][0].speaker],
        )
        self.assertEqual(
            [t.speaker for t in merged], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
        )

    def test_distinct_voices_stay_separate(self):
        vectors = {"RAW_0": np.array([1.0, 0.0, 0.0]), "RAW_1": np.array([0.0, 1.0, 0.0])}
        turns = [Segment(0, 5, "RAW_0"), Segment(6, 11, "RAW_1"), Segment(12, 17, "RAW_0")]
        samples = np.zeros(16000 * 18, dtype=np.float32)
        merged = consolidate_speakers(
            turns,
            samples,
            embed=lambda s, a, b: vectors[[t for t in turns if t.start == a][0].speaker],
        )
        self.assertEqual(len({t.speaker for t in merged}), 2)


if __name__ == "__main__":
    unittest.main()


class SlotTest(unittest.TestCase):
    def test_backchannel_inside_a_turn_is_dropped_and_crosstalk_split(self):
        from backend.speakers.diarization import make_exclusive

        turns = [Segment(0, 10, "A"), Segment(4, 4.4, "B"), Segment(9.6, 15, "B")]
        result = make_exclusive(turns)
        self.assertEqual(
            [(t.speaker, t.start, t.end) for t in result], [("A", 0, 9.8), ("B", 9.8, 15)]
        )

    def test_long_simultaneous_speech_is_kept(self):
        from backend.speakers.diarization import make_exclusive

        result = make_exclusive([Segment(0, 10, "A"), Segment(3, 7, "B")])
        self.assertEqual(len(result), 2)

    def test_slots_keep_real_pauses_and_join_tiny_ones(self):
        from backend.speakers.diarization import build_slots

        turns = [
            Segment(0, 6.8, "A"),
            Segment(6.95, 9.8, "A"),
            Segment(10.6, 12.8, "A"),
            Segment(15.4, 25, "A"),
        ]
        slots = build_slots(turns)
        self.assertEqual([(s.start, s.end) for s in slots], [(0, 9.8), (10.6, 12.8), (15.4, 25)])
