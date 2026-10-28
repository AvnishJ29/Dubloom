"""The voice library is read from voices.md, and casting prefers native, genre-fitting, register-matching voices."""

import unittest

from backend.voices.library import LIBRARY, cast, speaking_rate


class LibraryTest(unittest.TestCase):
    def test_every_card_is_read(self):
        self.assertEqual(len(LIBRARY), 45)
        amit = LIBRARY["Amit"]
        self.assertEqual(
            (amit["gender"], amit["language"], amit["pitch_hz"], amit["chars_per_second"]),
            ("male", "Hindi", 152, 12.6),
        )
        self.assertIn("e-learning and lectures", amit["best_for"])
        self.assertEqual(LIBRARY["Suhani"]["language"], "Hindi")

    def test_lecture_casts_the_native_teaching_voice_and_a_distinct_second_male(self):
        profiles = {
            "S0": {"gender": "male", "pitch_hz": 150, "first_seen": 0},
            "S1": {"gender": "male", "pitch_hz": 170, "first_seen": 5},
        }
        chosen, reasons = cast(["S0", "S1"], "hi", profiles, genre="lecture")
        self.assertEqual(chosen["S0"], "Amit")
        self.assertNotEqual(chosen["S1"], "Amit")
        self.assertEqual(LIBRARY[chosen["S1"]]["gender"], "male")
        self.assertIn("native Hindi", reasons["S0"])

    def test_trailer_prefers_a_dramatic_voice_for_the_second_man(self):
        profiles = {
            "S0": {"gender": "male", "pitch_hz": 110, "first_seen": 0},
            "S1": {"gender": "male", "pitch_hz": 115, "first_seen": 3},
        }
        chosen, _ = cast(["S0", "S1"], "hi", profiles, genre="film")
        self.assertEqual(chosen["S0"], "Amit")  # the native voice still comes first
        self.assertTrue(
            any(
                word in " ".join(LIBRARY[chosen["S1"]]["best_for"]).lower()
                for word in ("dramatic", "trailer", "theatrical")
            )
        )

    def test_female_speaker_gets_a_female_voice_and_overrides_win(self):
        profiles = {"S0": {"gender": "female", "pitch_hz": 240, "first_seen": 0}}
        chosen, reasons = cast(["S0"], "hi", profiles, genre="news")
        self.assertEqual(LIBRARY[chosen["S0"]]["gender"], "female")
        chosen, reasons = cast(["S0"], "hi", profiles, overrides={"S0": "Anitha"})
        self.assertEqual((chosen["S0"], reasons["S0"]), ("Anitha", "chosen by you"))

    def test_chosen_voices_apply_in_speaking_order(self):
        profiles = {
            "S0": {"gender": "male", "pitch_hz": 150, "first_seen": 0},
            "S1": {"gender": "female", "pitch_hz": 240, "first_seen": 5},
            "S2": {"gender": "male", "pitch_hz": 120, "first_seen": 9},
        }
        chosen, reasons = cast(
            ["S0", "S1", "S2"], "hi", profiles, genre="lecture", preferred=["Sagar", "Anitha"]
        )
        self.assertEqual(
            (chosen["S0"], chosen["S1"]), ("Sagar", "Anitha")
        )  # the user's picks, as given
        self.assertEqual(reasons["S0"], "chosen by you")
        self.assertNotIn(
            chosen["S2"], ("Sagar", "Anitha")
        )  # the third speaker is matched automatically

    def test_speaking_rate_comes_from_the_card(self):
        self.assertEqual(speaking_rate("Madhukar"), 9.9)
        self.assertEqual(speaking_rate("nobody", 12.5), 12.5)


if __name__ == "__main__":
    unittest.main()
