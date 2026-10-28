"""Subtitle timestamps stay valid at every boundary."""

import unittest

from backend.media.subtitles import _timestamp


class TimestampTest(unittest.TestCase):
    def test_milliseconds_carry_into_the_next_second(self):
        self.assertEqual(_timestamp(1.9996), "00:00:02,000")  # never 00:00:01,1000
        self.assertEqual(_timestamp(59.9999), "00:01:00,000")
        self.assertEqual(_timestamp(3599.9999, "."), "01:00:00.000")

    def test_plain_times(self):
        self.assertEqual(_timestamp(0), "00:00:00,000")
        self.assertEqual(_timestamp(3725.25), "01:02:05,250")
        self.assertEqual(_timestamp(-1), "00:00:00,000")


if __name__ == "__main__":
    unittest.main()
