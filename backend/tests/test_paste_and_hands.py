"""Only the generated lower face replaces original pixels, and a hand over the mouth stops the redraw."""

import unittest

import numpy as np

from backend.speakers.speaker_faces import hand_covered
from backend.workers.paste import paste_generated_region


class PasteTest(unittest.TestCase):
    def test_pixels_outside_the_region_stay_original(self):
        original = np.full((200, 200, 3), 50, np.uint8)
        pasted = np.full((200, 200, 3), 200, np.uint8)  # the model's (softer) version of everything
        region = np.zeros((64, 64), np.uint8)
        region[32:, :] = 255  # the lower half of the crop was generated
        crop = np.array(
            [[1.0, 0.0, -50.0], [0.0, 1.0, -50.0]]
        )  # the crop is the frame's 100x100 window at (50, 50)
        out = paste_generated_region(original, pasted, region, crop, 100, 100, feather_share=0.01)
        self.assertEqual(int(out[20, 20, 0]), 50)  # far outside the face: untouched
        self.assertEqual(int(out[60, 100, 0]), 50)  # upper face (eyes): untouched
        self.assertEqual(int(out[135, 100, 0]), 200)  # lower face: generated


class HandsTest(unittest.TestCase):
    def test_a_hand_over_the_mouth_removes_frames_with_margin(self):
        bbox = np.array([[100, 100, 300, 400]] * 20, np.float32)
        track = np.zeros(20, np.int32)
        hands = {
            "frame": np.array([10, 15]),
            "bbox": np.array([[120, 300, 280, 420], [600, 600, 700, 700]], np.float32),
        }
        covered = hand_covered(bbox, track, hands)
        self.assertTrue(covered[8:13].all())  # frame 10 and two frames either side
        self.assertFalse(covered[15])  # a hand elsewhere in the frame
        self.assertFalse(covered[0])
        self.assertFalse(hand_covered(bbox, track, None).any())


if __name__ == "__main__":
    unittest.main()
