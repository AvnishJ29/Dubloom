"""Span planning of the LatentSync worker: which frames form a span and which Whisper chunk each
frame uses (both feed the span cache keys). Runs without torch: the worker imports it lazily."""

import sys
import unittest

import numpy as np

from backend.workers import paste

# the worker runs as a script and imports its sibling as `paste`
sys.modules.setdefault('paste', paste)
from backend.workers import latentsync_worker as worker  # noqa: E402


class SpanPlanningTest(unittest.TestCase):
    def test_spans_follow_one_face_and_are_capped(self):
        long_run = worker.MAX_SPAN + 5
        track = np.array([-1, 0, 0, 1, 1, -1, 0] + [2] * long_run)
        self.assertEqual(
            worker.spans_of(track),
            [(1, 3), (3, 5), (6, 7), (7, 7 + worker.MAX_SPAN), (7 + worker.MAX_SPAN, 7 + long_run)],
        )

    def test_short_window_holds_its_last_frame(self):
        chunks = list(range(10))
        self.assertEqual(worker.chunk_indices(chunks, 2, 4, 3, 5), [2, 3, 4, 4, 4])

    def test_frames_past_the_last_chunk_reuse_it(self):
        chunks = list(range(10))
        self.assertEqual(worker.chunk_indices(chunks, 0, 8, 4, 4), [8, 9, 9, 9])


if __name__ == '__main__':
    unittest.main()
