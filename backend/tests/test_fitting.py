"""Lines are rewritten until they fit their slot within 0.9x-1.1x, and placed exactly on the slot."""

import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.speech import fitting
from backend.models import Segment
from backend.media.timeline import HARD_MAX_SPEED, plan_timeline


def tone(path: Path, seconds: float):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={seconds:.3f}",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
    )


_accept_all = patch.object(
    fitting, "verify_rewrites", lambda items, *a, **k: {item["id"] for item in items}
)


def setUpModule():
    _accept_all.start()  # the meaning check is an OpenRouter call; tests never reach the network


def tearDownModule():
    _accept_all.stop()


def run_stage(items, worker, workers, done, control=None):
    for index, item in enumerate(items):
        worker(index, item)


class PlacementTest(unittest.TestCase):
    def test_every_line_fills_exactly_its_slot(self):
        plan = plan_timeline([(2.0, 7.0, 5.4), (8.0, 11.0, 2.8)])
        self.assertEqual([p.start for p in plan], [2.0, 8.0])
        self.assertAlmostEqual(plan[0].speed, 1.08, places=3)
        self.assertAlmostEqual(plan[0].room, 5.0, places=3)
        self.assertAlmostEqual(plan[1].speed, 2.8 / 3.0, places=3)
        self.assertAlmostEqual(plan[1].room, 3.0, places=3)

    def test_hard_limits_trim_or_leave_silence_but_never_leave_the_slot(self):
        plan = plan_timeline([(0.0, 2.0, 4.0), (5.0, 9.0, 2.0)])
        self.assertEqual(plan[0].speed, HARD_MAX_SPEED)
        self.assertTrue(plan[0].trimmed)
        self.assertAlmostEqual(plan[0].room, 2.0)
        self.assertEqual(plan[1].speed, 0.8)
        self.assertAlmostEqual(plan[1].room, 2.5)


class FitLoopTest(unittest.TestCase):
    def test_long_and_short_lines_are_rewritten_into_the_band_and_good_ones_untouched(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            # Speaking rate of the fake voice: 10 characters per second.
            lines = [
                Segment(0, 5, "A", translation="x" * 50),  # 5.0 s for 5 s: fits (1.0)
                Segment(
                    6, 10, "A", text="a long line of many words", translation="y" * 60
                ),  # 6.0 s for 4 s: 1.5x, too long
                Segment(11, 16, "B", text="a line of six words here", translation="z" * 30),
            ]  # 3.0 s for 5 s: 0.6x, too short
            for index, line in enumerate(lines):
                line.audio_path = str(root / f"tts_{index:04d}.wav")
                tone(Path(line.audio_path), len(line.translation) / 10)
            rendered = []

            def render(segment):
                rendered.append(segment.start)
                tone(Path(segment.audio_path), len(segment.translation) / 10)

            def fake_retime(requests, language, usage=None, source_language=None):
                return {r["id"]: "w" * r["target_chars"] for r in requests}

            with patch.object(fitting, "retime_lines", fake_retime):
                report = fitting.fit_lines(lines, root, "hi", render, 1, run_stage)
            self.assertEqual(sorted(rendered), [6, 11])
            self.assertEqual(report.in_band, 3)
            self.assertEqual(report.outside, [])
            self.assertEqual(len(lines[0].translation), 50)

    def test_lines_edited_by_the_user_keep_their_words(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            line = Segment(0, 4, "A", translation="y" * 60)  # 6.0 s for 4 s: 1.5x
            line.edited = True
            line.audio_path = str(root / "tts_0000.wav")
            tone(Path(line.audio_path), 6.0)

            def fake_retime(requests, language, usage=None, source_language=None):
                raise AssertionError("an edited line must not be rewritten")

            with patch.object(fitting, "retime_lines", fake_retime):
                report = fitting.fit_lines([line], root, "hi", lambda segment: None, 1, run_stage)
            self.assertEqual(line.translation, "y" * 60)
            self.assertEqual(report.rewritten, 0)
            self.assertEqual(len(report.outside), 1)

    def test_a_rewrite_that_makes_things_worse_is_rolled_back(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            line = Segment(0, 4, "A", translation="y" * 60)
            line.audio_path = str(root / "tts_0000.wav")
            tone(Path(line.audio_path), 6.0)

            def render(segment):
                tone(Path(segment.audio_path), len(segment.translation) / 10)

            with patch.object(
                fitting,
                "retime_lines",
                lambda requests, language, usage=None, source_language=None: {0: "q" * 90},
            ):
                report = fitting.fit_lines([line], root, "hi", render, 1, run_stage)
            self.assertEqual(line.translation, "y" * 60)
            self.assertEqual(len(report.outside), 1)


class PrefitTest(unittest.TestCase):
    def test_lines_are_sized_before_the_first_voicing_and_good_ones_are_left_alone(self):
        rate = fitting.SPEAKING_RATE["hi"]
        lines = [
            Segment(0, 4, "A", translation="x" * round(rate * 4)),  # predicted 1.0: voiced as is
            Segment(
                5, 7, "A", text="a line that is far too long", translation="y" * round(rate * 3)
            ),  # predicted 1.5: shortened first
            Segment(8, 10, "B", text="a line typed by the user", translation="z" * round(rate * 3)),
        ]  # same, but typed by the user
        lines[2].edited = True
        seen = []

        def fake_retime(requests, language, usage=None, source_language=None):
            seen.extend(r["id"] for r in requests)
            return {r["id"]: "w" * r["target_chars"] for r in requests}

        with patch.object(fitting, "retime_lines", fake_retime):
            changed = fitting.prefit_lines(lines, "hi")
        self.assertEqual(seen, [1])
        self.assertEqual(changed, 1)
        self.assertEqual(len(lines[1].translation), round(rate * 2))
        self.assertTrue(lines[1].condensed)
        self.assertEqual(lines[2].translation, "z" * round(rate * 3))

    def test_a_rewrite_that_fails_the_meaning_check_is_not_used(self):
        rate = fitting.SPEAKING_RATE["hi"]
        line = Segment(
            0, 2, "A", text="stolen lives, you have lived", translation="y" * round(rate * 3)
        )
        with (
            patch.object(
                fitting,
                "retime_lines",
                lambda requests, *a, **k: {r["id"]: "short" for r in requests},
            ),
            patch.object(fitting, "verify_rewrites", lambda items, *a, **k: set()),
        ):
            self.assertEqual(fitting.prefit_lines([line], "hi"), 0)
        self.assertEqual(line.translation, "y" * round(rate * 3))

    def test_a_one_word_line_is_never_padded(self):
        line = Segment(0, 1.0, "A", text="Doom.", translation="डूम")
        with patch.object(
            fitting,
            "retime_lines",
            lambda requests, *a, **k: (
                self.fail("a one-word line must not be lengthened") if requests else {}
            ),
        ):
            self.assertEqual(fitting.prefit_lines([line], "hi"), 0)
        self.assertEqual(line.translation, "डूम")

    def test_a_slightly_short_take_is_accepted_without_revoicing(self):
        self.assertTrue(fitting.acceptable(0.88, 4.0, 0.9, 1.1))  # 0.09 s pause at 0.9x
        self.assertFalse(fitting.acceptable(0.7, 4.0, 0.9, 1.1))
        self.assertFalse(fitting.acceptable(1.2, 4.0, 0.9, 1.1))


if __name__ == "__main__":
    unittest.main()


class RollbackCacheTest(unittest.TestCase):
    def test_after_a_rollback_the_mix_uses_the_restored_take(self):
        from backend.media.audio_merge import natural_durations

        with TemporaryDirectory() as temp:
            root = Path(temp)
            line = Segment(0, 4, "A", translation="y" * 60)
            line.audio_path = str(root / "tts_0000.wav")
            tone(Path(line.audio_path), 6.0)

            def render(segment):
                tone(Path(segment.audio_path), len(segment.translation) / 10)

            with patch.object(
                fitting,
                "retime_lines",
                lambda requests, language, usage=None, source_language=None: {0: "q" * 90},
            ):
                fitting.fit_lines([line], root, "hi", render, 1, run_stage)
            self.assertAlmostEqual(natural_durations([line], root)[0], 6.0, delta=0.15)


class TailRuleTest(unittest.TestCase):
    def test_a_slightly_short_line_keeps_the_band_floor_and_ends_with_a_short_pause(self):
        plan = plan_timeline([(0.0, 4.8, 4.18)], min_speed=0.9)
        self.assertEqual(plan[0].speed, 0.9)
        self.assertAlmostEqual(plan[0].room, 4.18 / 0.9, places=3)

    def test_a_much_too_short_line_is_still_stretched_to_fill_its_slot(self):
        plan = plan_timeline([(0.0, 10.0, 8.2)], min_speed=0.9)
        self.assertAlmostEqual(plan[0].speed, 0.82, places=3)
        self.assertAlmostEqual(plan[0].room, 10.0, places=3)


class ExactTimelineTest(unittest.TestCase):
    def test_exact_mode_stays_in_the_band(self):
        from backend.media.timeline import plan_timeline

        # a 22 s slot voiced in 15 s (slow it down), and a 4 s slot voiced in 6.5 s (speed it up)
        plan = plan_timeline([(0.0, 22.0, 15.0), (30.0, 34.0, 6.5)], exact=True)
        slow, fast = plan
        self.assertEqual(slow.speed, 0.85)  # never dragged below the band
        self.assertAlmostEqual(slow.room, 15 / 0.85, places=2)  # ends early instead
        self.assertEqual(fast.speed, 1.15)  # never rushed above the band...
        self.assertGreater(fast.borrowed, 0)  # ...it borrows the silence after the slot

    def test_exact_mode_keeps_the_pace_band_and_borrows_the_gap(self):
        from backend.media.timeline import plan_timeline

        # line 1: 4 s slot voiced in 5.2 s (needs 1.3x) with 2 s of silence before the next line -> 1.15x, spilling 0.52 s
        # line 2: 3 s slot voiced in 2 s (0.67x) -> plays at 0.85x and ends early
        # line 3: 2 s slot voiced in 4 s with the next line right behind -> rushed to the hard max and trimmed
        plan = plan_timeline(
            [(0.0, 4.0, 5.2), (6.0, 9.0, 2.0), (10.0, 12.0, 4.0), (12.2, 14.0, 1.8)],
            20.0,
            exact=True,
        )
        long, short, crowded, last = plan
        self.assertAlmostEqual(long.speed, 1.15, places=3)
        self.assertAlmostEqual(long.room, 5.2 / 1.15, places=2)
        self.assertGreater(long.borrowed, 0.5)
        self.assertFalse(long.trimmed)
        self.assertEqual(short.speed, 0.85)
        self.assertAlmostEqual(short.room, 2.0 / 0.85, places=2)
        self.assertEqual(crowded.speed, 1.3)
        self.assertTrue(crowded.trimmed)
        self.assertLessEqual(crowded.start + crowded.room, 12.2 - 0.15 + 1e-6)
        self.assertEqual(last.speed, 1.0)

    def test_natural_mode_keeps_its_limits(self):
        from backend.media.timeline import plan_timeline

        slow, fast = plan_timeline([(0.0, 22.0, 15.0), (30.0, 34.0, 6.5)], exact=False)
        self.assertEqual(slow.speed, 0.8)
        self.assertLess(slow.room, 22.0)
        self.assertEqual(fast.speed, 1.35)
        self.assertTrue(fast.trimmed)


class RomanTermsTest(unittest.TestCase):
    def test_a_rewrite_that_drops_an_english_term_is_rejected(self):
        from backend.speech.fitting import drops_roman_terms

        self.assertTrue(drops_roman_terms("প্রথম boundary এই run chase এর।", "রান তাড়ার প্রথম চার।"))
        self.assertFalse(
            drops_roman_terms("প্রথম boundary এই run chase এর।", "run chase-এর প্রথম boundary।")
        )
        self.assertFalse(drops_roman_terms("W1 transpose", "W1 transpose C"))

    def test_condensing_may_drop_redundant_english_words_but_never_names_or_figures(self):
        # 26 Oct, "3 IIT Profs": every shortening of these lines was refused for dropping "set" or "right", so the voice
        # was sped to 1.3x and the last words were cut. Ordinary words may go; the meaning check judges them.
        from backend.speech.fitting import drops_roman_terms

        line = "उनके नवीनतम papers का set यह demonstrate करता है कि quantum codes तीन qubits से चलते हैं, right, classical computers की तरह।"
        self.assertFalse(
            drops_roman_terms(
                line,
                "उनके नए papers दिखाते हैं कि quantum codes तीन qubits से चलते हैं, classical computers की तरह।",
            )
        )
        self.assertTrue(
            drops_roman_terms("Hi मैं Amrit हूं, Story 99 का founder।", "Hi मैं founder हूं।")
        )  # a name and a figure
        self.assertTrue(drops_roman_terms("इसे W1 transpose कहते हैं।", "इसे transpose कहते हैं।"))
