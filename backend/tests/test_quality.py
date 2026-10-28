"""The quality check names the failure modes seen in real dubs."""

import unittest

from backend.media.quality import assess


def job(**over):
    base = {
        "language": "ta",
        "segments": [
            {
                "start": 0,
                "end": 3,
                "speaker": "S",
                "text": "hello",
                "translation": "வணக்கம் world",
                "voiced_text": "வணக்கம் world",
                "timing_ratio": 1.0,
            },
            {
                "start": 3,
                "end": 6,
                "speaker": "S",
                "text": "W1",
                "translation": "W1 transpose",
                "voiced_text": "W1 transpose",
                "timing_ratio": 1.28,
                "text_source": "review",
            },
        ],
        "lip_sync": {
            "engine": "latentsync",
            "lines_lip_synced": 2,
            "coverage_percent": 97.0,
            "off_screen_speakers": [],
            "hand_covered_frames": 0,
            "export": {"width": 1920, "height": 1080, "duration": 6.0, "audio_peak_db": -1.1},
        },
    }
    base.update(over)
    return base


class QualityTest(unittest.TestCase):
    def test_flags_reviewer_fallback_symbol_lines_and_rushed_lines(self):
        report = assess(job())
        by = {item["label"]: item for item in report["items"]}
        self.assertEqual(by["Speech recognised"]["status"], "check")
        self.assertEqual(by["Translation"]["status"], "check")  # "W1 transpose" has no Tamil words
        self.assertEqual(by["Timing"]["status"], "check")  # 1.28x
        self.assertEqual(by["Lip sync"]["status"], "ok")
        self.assertEqual(by["Export"]["status"], "ok")
        self.assertEqual(report["verdict"], "check")

    def test_clean_job_passes(self):
        clean = job()
        clean["segments"][1].update(
            translation="W ஒன்று transpose",
            voiced_text="W ஒன்று transpose",
            timing_ratio=1.0,
            text_source="",
        )
        report = assess(clean)
        self.assertEqual(report["verdict"], "ok")
        self.assertTrue(all(item["status"] == "ok" for item in report["items"]))

    def test_silent_export_is_a_fix(self):
        bad = job()
        bad["lip_sync"]["export"]["audio_peak_db"] = -60
        self.assertEqual(assess(bad)["verdict"], "fix")


if __name__ == "__main__":
    unittest.main()
