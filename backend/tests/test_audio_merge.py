import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.media.audio_merge import assemble_timeline
from backend.models import Segment


class AudioMergeTest(unittest.TestCase):
    def test_voice_tracks_are_mixed_over_a_finite_audio_bed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            voice = root / "voice.wav"
            voice.touch()
            segment = Segment(2, 4, "SPEAKER_00", audio_path=str(voice))
            with (
                patch("backend.media.audio_merge.duration", return_value=1),
                patch("backend.media.audio_merge.run") as run,
            ):
                assemble_timeline([segment], 10, root)

            mix_command = run.call_args_list[-1].args
            source = mix_command[mix_command.index("-i") + 1]
            self.assertEqual(source, "anullsrc=r=48000:cl=mono:d=10.000")
            filters = (root / "mix_filters.txt").read_text(encoding="utf-8")
            self.assertIn("duration=longest", filters)
            self.assertIn("alimiter=limit=0.95", filters)

    def test_bed_ramps_instead_of_stepping(self):
        from backend.media.audio_merge import duck_envelope

        expression = duck_envelope([(2.0, 4.0)], 0.9, 0.6)
        self.assertNotIn("if(", expression)
        self.assertIn("clip(", expression)
        self.assertEqual(duck_envelope([], 0.9, 0.6), "0.9")
        import shutil, subprocess

        if shutil.which("ffmpeg"):  # the expression must parse in ffmpeg's volume filter
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=duration=1",
                    "-af",
                    f"volume=volume='{expression}':eval=frame",
                    "-f",
                    "null",
                    "-",
                ],
                check=True,
            )

    def test_every_line_gets_short_fades(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            voice = root / "voice.wav"
            voice.touch()
            segment = Segment(2, 4, "SPEAKER_00", audio_path=str(voice))
            with (
                patch("backend.media.audio_merge.duration", return_value=1),
                patch("backend.media.audio_merge.run") as run,
            ):
                assemble_timeline([segment], 10, root)
            line_command = [
                call.args for call in run.call_args_list if "fitted_" in str(call.args[-1])
            ][0]
            chain = line_command[line_command.index("-af") + 1]
            self.assertIn("afade=t=in", chain)
            self.assertIn("afade=t=out", chain)


if __name__ == "__main__":
    unittest.main()
