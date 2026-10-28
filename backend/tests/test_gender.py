"""A confident classifier casts the voice; reviewer and pitch only decide when it abstains or is weak."""

import unittest

from backend.speakers.gender import decide_gender


class GenderVoteTest(unittest.TestCase):
    def test_confident_classifier_beats_reviewer_and_pitch_together(self):
        out = decide_gender(
            {'gender': 'male', 'gender_confidence': 1.0},
            {'gender': 'female', 'confidence': 0.95},
            'male',
        )
        self.assertEqual(out['gender'], 'female')
        self.assertIn('classifier', out['gender_source'])

    def test_weak_classifier_is_outvoted_when_both_others_agree(self):
        out = decide_gender(
            {'gender': 'male', 'gender_confidence': 1.0},
            {'gender': 'female', 'confidence': 0.62},
            'male',
        )
        self.assertEqual(out['gender'], 'male')

    def test_classifier_abstains_below_threshold(self):
        out = decide_gender(
            {'gender': 'female', 'gender_confidence': 0.35},
            {'gender': 'male', 'confidence': 0.55},
            None,
        )
        self.assertEqual(out['gender'], 'female')
        self.assertEqual(out['gender_source'], 'pitch')

    def test_nothing_known(self):
        self.assertEqual(
            decide_gender({'gender': 'unknown', 'gender_confidence': 0.0}, {}, None)['gender'],
            'unknown',
        )


if __name__ == "__main__":
    unittest.main()


class SplitTest(unittest.TestCase):
    def test_confident_minority_turns_become_a_second_speaker(self):
        from backend.speakers.gender import split_mixed_speakers
        from backend.models import Segment

        turns = [
            Segment(2.0, 4.7, "S0"),
            Segment(8.8, 9.9, "S0"),
            Segment(22.5, 23.2, "S0"),
            Segment(25.5, 27.8, "S0"),
            Segment(30.0, 33.0, "S0"),
        ]
        classified = {
            "S0": {
                "gender": "male",
                "confidence": 0.7,
                "turns": [
                    {"start": 2.0, "end": 4.7, "gender": "female", "confidence": 0.96},
                    {"start": 8.8, "end": 9.9, "gender": "female", "confidence": 0.77},
                    {"start": 22.5, "end": 23.2, "gender": "male", "confidence": 1.0},
                    {"start": 25.5, "end": 27.8, "gender": "male", "confidence": 0.7},
                    {"start": 30.0, "end": 33.0, "gender": "male", "confidence": 0.6},
                ],
            }
        }
        created = split_mixed_speakers(turns, classified)
        self.assertEqual(created, {"S0b": "S0"})
        self.assertEqual(
            [t.speaker for t in turns], ["S0b", "S0b", "S0", "S0", "S0"]
        )  # the 0.77 turn is sure enough to move too
        self.assertEqual(classified["S0b"]["gender"], "female")
        self.assertEqual((classified["S0"]["gender"], len(classified["S0"]["turns"])), ("male", 3))

    def test_a_single_odd_short_turn_does_not_split(self):
        from backend.speakers.gender import split_mixed_speakers
        from backend.models import Segment

        turns = [Segment(0, 0.5, "S0"), Segment(1, 5, "S0")]
        classified = {
            "S0": {
                "gender": "male",
                "turns": [
                    {"start": 0, "end": 0.5, "gender": "female", "confidence": 0.95},
                    {"start": 1, "end": 5, "gender": "male", "confidence": 0.99},
                ],
            }
        }
        self.assertEqual(split_mixed_speakers(turns, classified), {})
        self.assertEqual({t.speaker for t in turns}, {"S0"})

    def test_unsure_turns_follow_their_nearest_sure_neighbour(self):
        from backend.speakers.gender import split_mixed_speakers
        from backend.models import Segment

        turns = [
            Segment(0, 3, "S0"),
            Segment(10, 12, "S0"),
            Segment(12.5, 14, "S0"),
            Segment(20, 23, "S0"),
        ]
        classified = {
            "S0": {
                "gender": "male",
                "turns": [
                    {"start": 0, "end": 3, "gender": "female", "confidence": 0.95},
                    {"start": 10, "end": 12, "gender": "male", "confidence": 0.9},
                    {
                        "start": 12.5,
                        "end": 14,
                        "gender": "female",
                        "confidence": 0.52,
                    },  # unsure: sits right after a sure male turn
                    {"start": 20, "end": 23, "gender": "male", "confidence": 0.65},
                ],
            }
        }
        split_mixed_speakers(turns, classified)
        self.assertEqual([t.speaker for t in turns], ["S0b", "S0", "S0", "S0"])

    def test_the_reviewers_gender_keeps_the_original_id(self):
        from backend.speakers.gender import split_mixed_speakers
        from backend.models import Segment

        turns = [Segment(0, 3, "S0"), Segment(10, 12, "S0")]
        classified = {
            "S0": {
                "gender": "female",
                "turns": [
                    {
                        "start": 0,
                        "end": 3,
                        "gender": "female",
                        "confidence": 0.95,
                    },  # more speech...
                    {"start": 10, "end": 12, "gender": "male", "confidence": 0.9},
                ],
            }
        }
        split_mixed_speakers(
            turns, classified, heard={"S0": "male"}
        )  # ...but the reviewer named the man
        self.assertEqual([t.speaker for t in turns], ["S0b", "S0"])
        self.assertEqual(classified["S0"]["gender"], "male")


class PitchGuardTest(unittest.TestCase):
    def test_one_register_is_not_split_even_when_the_classifier_wavers(self):
        from backend.speakers.gender import split_mixed_speakers
        from backend.models import Segment

        turns = [
            Segment(0, 5, "S0"),
            Segment(6, 15, "S0"),
            Segment(18, 22, "S0"),
            Segment(38, 41, "S0"),
        ]
        classified = {
            "S0": {
                "gender": "male",
                "turns": [
                    {"start": 0, "end": 5, "gender": "male", "confidence": 0.78},
                    {"start": 6, "end": 15, "gender": "male", "confidence": 0.93},
                    {"start": 18, "end": 22, "gender": "female", "confidence": 0.85},
                    {"start": 38, "end": 41, "gender": "female", "confidence": 0.87},
                ],
            }
        }
        same_register = {
            (0, 5): 188,
            (6, 15): 193,
            (18, 22): 222,
            (38, 41): 227,
        }  # all within one voice's range
        self.assertEqual(
            split_mixed_speakers(turns, classified, pitch_of=lambda a, b: same_register[(a, b)]), {}
        )
        self.assertEqual({t.speaker for t in turns}, {"S0"})
        two_people = {(0, 5): 115, (6, 15): 120, (18, 22): 222, (38, 41): 227}  # a man and a woman
        self.assertEqual(
            split_mixed_speakers(turns, classified, pitch_of=lambda a, b: two_people[(a, b)]),
            {"S0b": "S0"},
        )


class FaceVoteTest(unittest.TestCase):
    def test_a_sure_speaking_face_settles_an_ambiguous_voice(self):
        # audio wavers (classifier male 0.67), the reviewer says male, pitch says female; the face is clearly male
        out = decide_gender(
            {
                'gender': 'male',
                'gender_confidence': 0.6,
                'gender_votes': {'male': 2.9, 'female': 1.0},
            },
            {'gender': 'male', 'confidence': 0.67},
            'male',
            seen={'face_gender': 'male', 'face_gender_share': 0.97},
        )
        self.assertEqual(out['gender'], 'male')
        self.assertIn('face', out['gender_source'])
        # and the face settles it when the audio signals are weak or at odds
        out = decide_gender(
            {'gender': 'female', 'gender_confidence': 0.4},
            {'gender': 'male', 'confidence': 0.62},
            None,
            seen={'face_gender': 'female', 'face_gender_share': 0.95},
        )
        self.assertEqual(out['gender'], 'female')
        # an unsure face does not vote
        out = decide_gender(
            {'gender': 'male', 'gender_confidence': 1.0},
            {'gender': 'male', 'confidence': 0.9},
            None,
            seen={'face_gender': 'female', 'face_gender_share': 0.6},
        )
        self.assertEqual(out['gender'], 'male')
