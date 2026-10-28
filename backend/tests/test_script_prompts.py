"""The script passes ask for pauses the voice can breathe on and keep English in Roman letters when code-mixed."""

import unittest
from unittest.mock import patch

from backend.speech import script_supervisor
from backend.models import Segment


class ScriptPromptTest(unittest.TestCase):
    def test_post_edit_asks_for_pauses_and_roman_english(self):
        seen = {}
        segments = [
            Segment(
                0,
                2,
                "SPEAKER_00",
                text="We use a diffusion model.",
                translation="हम डिफ्यूजन मॉडल का उपयोग करते हैं।",
            )
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: seen.update(system=system) or {},
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
        ):
            script_supervisor.post_edit(segments, "hi", "en", script="codemix")
        self.assertIn("pauses on commas", seen["system"])
        self.assertIn("Roman letters", seen["system"])
        self.assertIn("never transliterated", seen["system"])

    def test_native_script_still_transliterates(self):
        self.assertIn("transliterate names", script_supervisor.script_rule("Hindi", "native"))


if __name__ == "__main__":
    unittest.main()


class FillerRuleTest(unittest.TestCase):
    def test_post_edit_drops_fillers_and_transcript_spells_math(self):
        self.assertIn("hesitation sounds", script_supervisor.POSTEDIT_PROMPT)
        self.assertIn("never as notation", script_supervisor.TRANSCRIPT_PROMPT)


class ContentRuleTest(unittest.TestCase):
    def test_post_edit_carries_the_content_register(self):
        seen = {}
        segments = [
            Segment(0, 2, "SPEAKER_00", text="So the gradient here", translation="तो यहाँ gradient")
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: seen.update(system=system) or {},
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
        ):
            script_supervisor.post_edit(
                segments,
                "hi",
                "en",
                script="codemix",
                content={
                    "genre": "lecture",
                    "label": "University lecture on optimisation",
                    "register": "formal teaching register with आप",
                },
            )
        self.assertIn("University lecture on optimisation", seen["system"])
        self.assertIn("formal teaching register", seen["system"])

    def test_profile_parses_the_model_answer(self):
        with (
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(
                script_supervisor.openrouter,
                "chat",
                lambda *a, **k: '{"genre": "Lecture", "label": "Deep learning lecture", "register": "formal"}',
            ),
        ):
            out = script_supervisor.profile_content(
                [Segment(0, 1, "S", text="hello")], {"categories": ["Education"]}, "L35", "en"
            )
        self.assertEqual(
            out, {"genre": "lecture", "label": "Deep learning lecture", "register": "formal"}
        )


class VerbatimSourceTest(unittest.TestCase):
    def test_the_enhancer_gets_the_second_transcript_as_a_hint_only(self):
        seen = {}
        segments = [
            Segment(
                0,
                2,
                "SPEAKER_00",
                text="beta times uh u t",
                translation="बीटा गुणा यू टी",
                reference_text="beta times u sub t",
            )
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: seen.update(system=system, user=user)
                or {5: "x"},
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
        ):
            script_supervisor.post_edit(segments, "hi", "en", script="codemix")
        self.assertIn("also heard as", seen["user"])
        self.assertIn("beta times u sub t", seen["user"])
        self.assertIn("never yours to change", seen["system"])
        self.assertEqual(segments[0].text, "beta times uh u t")  # the source is untouched


class EarlyProfileTest(unittest.TestCase):
    def test_profile_works_from_metadata_before_any_transcript(self):
        seen = {}

        def chat(model, messages, **kw):
            seen["user"] = messages[1]["content"]
            return '{"genre": "news", "label": "TV news analysis", "register": "composed anchor"}'

        with (
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(script_supervisor.openrouter, "chat", chat),
        ):
            out = script_supervisor.profile_content(
                [],
                {"categories": ["News & Politics"], "channel": "Firstpost"},
                "Aramco strike",
                "en",
            )
        self.assertEqual(out["genre"], "news")
        self.assertIn("not transcribed yet", seen["user"])
        with patch.object(script_supervisor.openrouter, "configured", lambda: True):
            self.assertEqual(script_supervisor.profile_content([], {}, "", "en"), {})


class ContentRuleTest(unittest.TestCase):
    def test_lectures_get_the_maths_as_speech_rule_and_everyone_gets_spoken_forms(self):
        lecture = script_supervisor.content_rule(
            {"genre": "lecture", "label": "Linear algebra lecture", "register": "formal"}, "Tamil"
        )
        self.assertIn("W one transpose C W one", lecture)
        self.assertIn("Linear algebra lecture", lecture)
        news = script_supervisor.content_rule(
            {"genre": "news", "label": "TV news", "register": "formal"}, "Hindi"
        )
        self.assertNotIn("W one transpose", news)
        self.assertIn("24 billion", news)
        self.assertIn("24 billion", script_supervisor.content_rule(None, "Hindi"))


class LengthAndEssenceTest(unittest.TestCase):
    def test_targets_reach_the_editor_and_rejected_edits_are_reverted(self):
        from backend.speech import fitting

        seen = {}
        segments = [
            Segment(
                0,
                3,
                "SPEAKER_00",
                text="We use a diffusion model here.",
                translation="हम यहाँ diffusion model का उपयोग करते हैं।",
            )
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: seen.update(system=system, user=user)
                or {0: "हम diffusion model use करते हैं।"},
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(fitting, "verify_rewrites", lambda items, *a, **k: set()),
        ):  # the meaning check says no
            changed = script_supervisor.post_edit(
                segments, "hi", "en", script="codemix", targets={0: 30}
            )
        self.assertIn("30 chars", seen["user"])
        self.assertIn("target length in characters", seen["system"])
        self.assertIn("this is a dub, not a rewrite", seen["system"])
        self.assertEqual(changed, 0)
        self.assertEqual(segments[0].translation, "हम यहाँ diffusion model का उपयोग करते हैं।")
        self.assertEqual(segments[0].machine_translation, "")

    def test_accepted_edits_keep_the_machine_line_for_the_ui(self):
        from backend.speech import fitting

        segments = [
            Segment(
                0,
                3,
                "SPEAKER_00",
                text="We use a diffusion model here.",
                translation="हम यहाँ diffusion model का उपयोग करते हैं।",
            )
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: {0: "हम यहाँ diffusion model use करते हैं।"},
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(fitting, "verify_rewrites", lambda items, *a, **k: {0}),
        ):
            changed = script_supervisor.post_edit(
                segments, "hi", "en", script="codemix", targets={0: 30}
            )
        self.assertEqual(changed, 1)
        self.assertEqual(segments[0].machine_translation, "हम यहाँ diffusion model का उपयोग करते हैं।")


class ConservativeEditTest(unittest.TestCase):
    def test_a_wholesale_rewrite_of_a_fitting_line_is_dropped(self):
        from backend.speech import fitting

        segments = [
            Segment(
                0,
                3,
                "SPEAKER_00",
                text="The first boundary of the run chase.",
                translation="run chase-এর প্রথম boundary।",
            )
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: {
                    0: "এই ম্যাচে প্রথমবার বল সীমানা পার হলো, দারুণ শট!"
                },
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(fitting, "verify_rewrites", lambda items, *a, **k: {0}),
        ):
            changed = script_supervisor.post_edit(
                segments, "bn", "en", script="codemix", targets={0: 28}
            )
        self.assertEqual(changed, 0)
        self.assertEqual(segments[0].translation, "run chase-এর প্রথম boundary।")

    def test_a_small_edit_is_kept(self):
        from backend.speech import fitting

        segments = [
            Segment(
                0,
                3,
                "SPEAKER_00",
                text="The first boundary of the run chase.",
                translation="run chase-এর প্রথম boundary।",
            )
        ]
        with (
            patch.object(
                script_supervisor,
                "_ask",
                lambda system, user, expected, usage: {0: "run chase-এর প্রথম boundary, দারুণ।"},
            ),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
            patch.object(fitting, "verify_rewrites", lambda items, *a, **k: {0}),
        ):
            changed = script_supervisor.post_edit(
                segments, "bn", "en", script="codemix", targets={0: 28}
            )
        self.assertEqual(changed, 1)


class AskAiTimeBudgetTest(unittest.TestCase):
    """26 Oct, W1_L4: AI rewrites added emphasis ("समझ लीजिए!") without knowing the slot; applied lines needed up to
    2.6x and lost their last words. The rewrite now gets each line's target length from its own voice.
    """

    def test_each_line_carries_the_characters_its_voice_can_say_in_the_slot(self):
        seen = {}

        def ask(system, user, expected, usage):
            seen["system"], seen["user"] = system, user
            return {}

        line = {
            "start": 0,
            "end": 2,
            "speaker": "SPEAKER_00",
            "text": "It can't be better than N",
            "translation": "यह capital N से बेहतर नहीं हो सकता",
            "voiced_text": "यह capital N से बेहतर नहीं हो सकता",
            "fit_ratio": 1.5,
        }
        with (
            patch.object(script_supervisor, "_ask", ask),
            patch.object(script_supervisor.openrouter, "configured", lambda: True),
        ):
            script_supervisor.rewrite_lines(
                [line], "hi", "en", "make it engaging", script="codemix"
            )
        self.assertIn(
            f"{round(len(line['voiced_text']) / 1.5)} chars (now {len(line['translation'])})",
            seen["user"],
        )
        self.assertIn("not run longer than the target", seen["system"])
        self.assertIsNone(script_supervisor.slot_chars({"translation": "x"}))
