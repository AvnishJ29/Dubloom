"""Voice uniqueness, priority, and overlap reporting for multi-speaker dubbing."""

import unittest
from backend.voices.assignment import (
    assign_voices,
    voice_pool,
    overlapping_turns,
    voice_reuse_warnings,
)
from backend.models import Segment


class VoiceAssignmentTest(unittest.TestCase):
    def test_hindi_four_speakers_get_distinct_stable_voices(self):
        mapping = assign_voices(['a', 'b', 'c', 'a', 'd'], 'hi')
        self.assertEqual(list(mapping.values()), ['Kavya', 'Amit', 'Suhani', 'Anagha'])
        self.assertEqual(len(set(mapping.values())), 4)

    def test_language_priority_and_dynamic_capacity(self):
        self.assertEqual(voice_pool('te')[:4], ['Sravani', 'Vamsi', 'Anitha', 'Arun'])
        capacity = len(voice_pool('hi'))
        mapping = assign_voices([str(i) for i in range(capacity)], 'hi')
        self.assertEqual(len(set(mapping.values())), capacity)
        count = capacity * 3 + 7
        mapping = assign_voices([str(i) for i in range(count)], 'hi')
        self.assertEqual(len(mapping), count)
        self.assertEqual(mapping['0'], mapping[str(capacity)])
        self.assertEqual(len(voice_reuse_warnings(mapping)), 1)

    def test_preferred_voice_does_not_collapse_speakers(self):
        mapping = assign_voices(['a', 'b', 'c'], 'hi', 'Arun')
        self.assertEqual(mapping, {'a': 'Arun', 'b': 'Kavya', 'c': 'Amit'})

    def test_overlap_is_flagged_without_flagging_boundary_jitter(self):
        self.assertEqual(overlapping_turns([Segment(0, 1, 'a'), Segment(0.95, 2, 'b')]), [])
        warnings = overlapping_turns([Segment(0, 2, 'a'), Segment(1, 3, 'b')])
        self.assertEqual(len(warnings), 1)
        self.assertIn('1.0s', warnings[0])
