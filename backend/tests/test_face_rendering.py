"""Regression checks for frame-tail mel windows and upper-face preservation."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

# Localization dependencies are replaced here: these tests exercise pure image
# blending and timing only, without loading the large face detector checkpoint.
path = Path(__file__).resolve().parents[2] / 'Wav2Lip' / 'face_tracking.py'
spec = importlib.util.spec_from_file_location('face_tracking_helpers', path)
tracking = importlib.util.module_from_spec(spec)
with patch.dict(
    sys.modules, {'face_detection': types.SimpleNamespace(FaceAlignment=None, LandmarksType=None)}
):
    spec.loader.exec_module(tracking)


class FaceRenderingTest(unittest.TestCase):
    def test_scene_cut_never_reuses_the_previous_face(self):
        tracker = tracking.FaceTracker.__new__(tracking.FaceTracker)
        tracker.interval = 5
        tracker.frame_number = 1
        tracker.previous_gray = np.zeros((100, 100), dtype=np.uint8)
        tracker.previous_box = (10, 10, 90, 90)
        calls = []

        def detect(frames):
            calls.append(True)
            return [(None, 'no_face')]

        tracker._detect = detect
        result = tracker.detect([np.full((100, 100, 3), 255, dtype=np.uint8)])
        self.assertEqual(result, [(None, 'no_face')])
        self.assertEqual(calls, [True])
        self.assertIsNone(tracker.previous_box)

    def test_reliable_track_skips_detector_between_refreshes(self):
        tracker = tracking.FaceTracker.__new__(tracking.FaceTracker)
        tracker.interval = 5
        tracker.frame_number = 1
        rng = np.random.default_rng(7)
        tracker.previous_gray = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        tracker.previous_box = (10, 10, 90, 90)
        tracker._detect = lambda frames: self.fail(
            'Stable frame should use its validated optical flow'
        )
        frame = np.repeat(tracker.previous_gray[..., None], 3, axis=2)
        result = tracker.detect([frame])
        self.assertIsNone(result[0][1])
        np.testing.assert_allclose(result[0][0], (10, 10, 90, 90), atol=1)

    def test_end_of_audio_preserves_window_size(self):
        mel = np.arange(80 * 20).reshape(80, 20)
        window = tracking.mel_window(mel, 6, 25)
        self.assertEqual(window.shape, (80, 16))
        np.testing.assert_array_equal(window[:, 0], mel[:, 19])
        np.testing.assert_array_equal(window[:, -1], mel[:, -1])

    def test_upper_face_and_edges_are_preserved(self):
        original = np.full((100, 100, 3), 40, dtype=np.uint8)
        predicted = np.full((96, 96, 3), 200, dtype=np.uint8)
        result = tracking.blend_mouth(original, predicted)
        np.testing.assert_array_equal(result[:40], original[:40])
        np.testing.assert_array_equal(result[:, 0], original[:, 0])
        self.assertGreater(int(result[70, 50, 0]), 190)


class FaceChoiceTest(unittest.TestCase):
    def test_dominant_face_beats_a_small_bust_and_tracking_beats_size(self):
        big = [100, 100, 300, 300, 0.99]
        bust = [400, 300, 440, 340, 0.9]
        self.assertIs(tracking.choose_face([bust, big]), big)
        # Two similar faces with no track are ambiguous.
        other = [500, 100, 690, 290, 0.95]
        self.assertIsNone(tracking.choose_face([big, other]))
        # With a tracked box, continuity wins even over the larger face.
        self.assertIs(tracking.choose_face([big, other], previous_box=(505, 105, 695, 295)), other)


class BoxSmootherTest(unittest.TestCase):
    def test_smoothing_averages_only_the_same_face_track(self):
        smoother = tracking.BoxSmoother(window=3)
        self.assertEqual(smoother.apply((10, 10, 50, 50)), (10, 10, 50, 50))
        self.assertEqual(smoother.apply((12, 12, 52, 52)), (11, 11, 51, 51))
        # A far-away box is a different face or shot: the window must reset.
        self.assertEqual(smoother.apply((300, 300, 340, 340)), (300, 300, 340, 340))
        self.assertIsNone(smoother.apply(None))
        self.assertEqual(smoother.apply((20, 20, 60, 60)), (20, 20, 60, 60))


if __name__ == '__main__':
    unittest.main()
