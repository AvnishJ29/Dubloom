"""The face whose lips follow the speech loudness is the one that gets lip-synced."""

import unittest

import numpy as np

from backend.speakers.speaker_faces import compute_targets

FPS = 25


def landmarks(cx, cy, opening, scale=1.0):
    points = np.zeros((106, 2), np.float32)
    points[:] = (cx, cy)
    points[52] = (cx - 20 * scale, cy)
    points[61] = (cx + 20 * scale, cy)
    points[71] = (cx, cy - (4 + opening) * scale)
    points[53] = (cx, cy + (4 + opening) * scale)
    return points


def scene(frame_count=250, speaker_on_screen=True, rng=None):
    rng = rng or np.random.default_rng(5)
    envelope = np.log(np.abs(np.sin(np.arange(frame_count) / 2.5)) + 0.05)  # syllable rhythm
    rows = {k: [] for k in ("frame", "bbox", "emb", "lmk", "mouth")}
    talker, statue = np.zeros(512), np.zeros(512)
    talker[0], statue[1] = 1.0, 1.0
    for f in range(frame_count):
        if speaker_on_screen:
            opening = 6 * np.exp(envelope[f]) + rng.normal(0, 0.3)
            rows["frame"].append(f)
            rows["bbox"].append([100, 100, 260, 300])
            rows["emb"].append(talker)
            rows["lmk"].append(landmarks(180, 250, opening))
            rows["mouth"].append(np.zeros((24, 48)))
        rows["frame"].append(f)
        rows["bbox"].append([400, 100, 480, 200])
        rows["emb"].append(statue)
        rows["lmk"].append(landmarks(440, 170, 1 + rng.normal(0, 0.05), 0.5))
        rows["mouth"].append(np.zeros((24, 48)))
    faces = {
        "frame_count": frame_count,
        "frame": np.array(rows["frame"]),
        "bbox": np.array(rows["bbox"], np.float32),
        "emb": np.array(rows["emb"], np.float16),
        "lmk": np.array(rows["lmk"], np.float32),
        "mouth": np.array(rows["mouth"], np.uint8),
        "thumbs": np.zeros((frame_count, 18, 32), np.uint8),
    }
    return faces, envelope.astype(np.float32)


class ActiveSpeakerTest(unittest.TestCase):
    def test_the_talking_face_is_chosen_and_the_statue_is_not(self):
        faces, envelope = scene()
        targets = compute_targets(faces, [(0.0, 6.0, "SPEAKER_00")], FPS, envelope)
        self.assertEqual(targets.report["speaker_faces"], {"SPEAKER_00": 0})
        self.assertTrue(np.allclose(targets.bbox[10], [100, 100, 260, 300]))
        self.assertFalse(targets.active[200])  # outside the dubbed line nothing is touched

    def test_an_off_screen_voice_drives_no_face(self):
        faces, envelope = scene(speaker_on_screen=False)
        targets = compute_targets(faces, [(0.0, 6.0, "SPEAKER_00")], FPS, envelope)
        self.assertEqual(targets.report["unmatched_speakers"], ["SPEAKER_00"])
        self.assertEqual(int(targets.active.sum()), 0)


class LightAsdTargetingTest(unittest.TestCase):
    """With Light-ASD scores, a face is re-rendered only in lines where it is the one speaking."""

    def scores(self, faces, talker_score):
        # rows alternate talker / statue per frame (scene() with the speaker on screen)
        statue = faces["bbox"][:, 0] == 400
        values = np.where(statue, -2.0, 0.0).astype(np.float32)
        frames = faces["frame"]
        values[~statue] = [talker_score(int(f)) for f in frames[~statue]]
        return values

    def test_only_the_speaking_face_is_synced_and_voice_over_is_left_alone(self):
        faces, envelope = scene(frame_count=400)
        # The talker speaks in frames 0-149; in frames 200-349 a voice-over plays while the talker is silent.
        asd = self.scores(faces, lambda f: 1.0 if f < 150 else -1.5)
        lines = [(0.0, 6.0, "SPEAKER_00"), (8.0, 14.0, "SPEAKER_01")]
        targets = compute_targets(faces, lines, FPS, envelope, asd=asd)
        self.assertTrue(targets.active[10])
        self.assertTrue(
            np.allclose(targets.bbox[10], [100, 100, 260, 300])
        )  # the talker, never the statue
        self.assertFalse(targets.active[250])  # voice-over: untouched
        self.assertEqual(targets.report["speaker_faces"], {"SPEAKER_00": 0})
        self.assertEqual(targets.report["unmatched_speakers"], ["SPEAKER_01"])
        self.assertEqual(targets.report["lines_lip_synced"], 1)

    def test_the_whole_line_of_the_speaking_face_is_redrawn(self):
        faces, envelope = scene(frame_count=300)
        # Speaking with short pauses for 0-99 (frames 8-9 of every 10 silent), then silent 100-149 within the same line:
        # the dubbed voice decides when the mouth opens, so the face is redrawn for the whole line.
        asd = self.scores(faces, lambda f: (-0.5 if f % 10 > 7 else 1.5) if f < 100 else -1.5)
        targets = compute_targets(faces, [(0.0, 6.0, "SPEAKER_00")], FPS, envelope, asd=asd)
        self.assertTrue(targets.active[8:10].all() and targets.active[48])
        self.assertTrue(targets.active[120])
        self.assertTrue(targets.active[:150].mean() > 0.95)

    def test_an_undecided_face_is_left_alone(self):
        faces, envelope = scene(frame_count=100)
        asd = self.scores(
            faces, lambda f: 0.15
        )  # just above Light-ASD's boundary: not confident speech
        targets = compute_targets(faces, [(0.0, 4.0, "SPEAKER_00")], FPS, envelope, asd=asd)
        self.assertEqual(int(targets.active.sum()), 0)

    def test_a_quiet_line_of_the_settled_face_is_still_redrawn(self):
        faces, envelope = scene(frame_count=250)
        asd = self.scores(
            faces, lambda f: 0.9 if f < 125 else 0.15
        )  # the detector is sure in line 1, undecided in line 2
        targets = compute_targets(
            faces, [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00")], FPS, envelope, asd=asd
        )
        self.assertGreater(
            int(targets.active[125:].sum()), 50
        )  # same speaker, same face: redrawn by identity
        other = compute_targets(
            faces, [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")], FPS, envelope, asd=asd
        )
        self.assertEqual(
            int(other.active[130:].sum()), 0
        )  # another voice over that face: left alone


if __name__ == "__main__":
    unittest.main()
