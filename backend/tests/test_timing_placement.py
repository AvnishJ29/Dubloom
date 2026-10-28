import unittest
from backend.media.timeline import dubbed_slots, placement_notes
from backend.models import Segment


class TimingPlacementTest(unittest.TestCase):
    def test_lips_follow_borrowed_pause_and_early_finish(self):
        lines = [
            dict(start=0, end=4, dub_start=0, dub_end=4.52, speaker='A', translation='hello'),
            dict(start=6, end=9, dub_start=6, dub_end=8.35, speaker='B', translation='hello'),
            dict(start=10, end=12, speaker='C', translation='hello'),
            dict(start=12, end=14, speaker='C', translation=''),
        ]
        # A borrows 0.52 s of the pause: its lips keep moving to 4.52 s. B ends early at 8.35 s: its mouth is still
        # redrawn (closed) to 9 s, where the original speaker is still talking but the dub is silent.
        self.assertEqual(dubbed_slots(lines), [(0, 4.52, 'A'), (6, 9, 'B'), (10, 12, 'C')])

    def test_null_placement_keeps_legacy_timings(self):
        self.assertEqual(
            dubbed_slots(
                [dict(start=1, end=2, dub_start=None, dub_end=None, speaker='A', translation='yes')]
            ),
            [(1, 2, 'A')],
        )

    def test_warnings_distinguish_cut_audio_from_borrowed_pause(self):
        cut = Segment(0, 2, 'A', timing_trimmed=True, timing_ratio=1.3, dub_end=2)
        borrowed = Segment(3, 5, 'B', timing_borrowed=0.5, timing_ratio=1.15, dub_end=5.5)
        early = Segment(6, 9, 'C', timing_ratio=0.85, dub_end=8)
        notes = placement_notes([cut, borrowed, early])
        self.assertIn('cut at the end', notes[0])
        self.assertIn('0.50s of the following pause', notes[1])
        self.assertIn('ends 1.00s early', notes[2])

    def test_well_fitted_line_needs_no_warning(self):
        self.assertEqual(placement_notes([Segment(0, 2, 'A', dub_end=2, timing_ratio=1.02)]), [])
