"""After an edit, only lip-sync chunks whose own speaking-face targets changed are rendered again."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.rendering.lip_sync import _slice_digest, _target_arrays


class ChunkStampTest(unittest.TestCase):
    def test_a_change_outside_a_chunk_keeps_its_stamp(self):
        with TemporaryDirectory() as temp:
            frames = 100
            bbox = np.zeros((frames, 4), np.float32)
            lmk = np.zeros((frames, 106, 2), np.float32)
            track = np.full(frames, -1, np.int16)
            track[10:30] = 0
            first = Path(temp) / "first.npz"
            np.savez_compressed(first, frame_count=frames, bbox=bbox, lmk=lmk, track=track)
            track = track.copy()
            track[70:90] = 0  # a new line later in the video
            second = Path(temp) / "second.npz"
            np.savez_compressed(second, frame_count=frames, bbox=bbox, lmk=lmk, track=track)
            before, after = _target_arrays(first), _target_arrays(second)
            self.assertEqual(_slice_digest(before, 0, 50), _slice_digest(after, 0, 50))
            self.assertNotEqual(_slice_digest(before, 50, 100), _slice_digest(after, 50, 100))
            self.assertEqual(_slice_digest(None, 0, 50), "detector")


if __name__ == "__main__":
    unittest.main()
