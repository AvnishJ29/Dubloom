"""The orchestration order matters because every later stage uses timed speaker turns."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.jobs import pipeline
from backend.speech.fitting import FitReport
from backend.models import Segment


class PipelineOrderTest(unittest.TestCase):
    def test_rate_limited_job_reuses_saved_stages_on_retry(self):
        class FakeStore:
            state = {}

            def get(self, job_id):
                return self.state

            def save(self, job_id, **changes):
                self.state.update(changes)

        fake_store = FakeStore()
        attempts = []

        def process(job_id, resume, control):
            attempts.append(resume)
            fake_store.state = (
                {'status': 'failed', 'retryable': True}
                if len(attempts) == 1
                else {'status': 'completed', 'retryable': False}
            )

        with (
            patch.object(pipeline, 'store', fake_store),
            patch.object(pipeline, 'process_job', side_effect=process),
            patch.object(pipeline.time, 'sleep') as sleep,
        ):
            pipeline.run_job('job')
        self.assertEqual(attempts, [False, True])
        self.assertEqual(sleep.call_count, 180)

    def test_diarization_precedes_each_model_pass_and_lip_sync(self):
        calls = []

        class FakeStore:
            state = {}

            def get(self, job_id):
                return {
                    "id": job_id,
                    "language": "hi",
                    "source_language": "en",
                    "voice": None,
                    "keep_background": False,
                }

            def save(self, job_id, **changes):
                self.state.update(changes)
                calls.append(("status", changes.get("stage", "")))

        class FakeClient:
            def close(self):
                pass

        def fake_asr(client, audio, language, *args, **kwargs):
            calls.append(("asr", audio.name))
            return "Hello"

        def fake_translation(client, text, language, *args, **kwargs):
            calls.append(("translate", language))
            return "नमस्ते"

        class FakeVoices:
            def __init__(
                self, client, language, speaker_voices, mode="auto", references=None, usage=None
            ):
                self.speaker_voices, self.engine = speaker_voices, None

            def render(self, speaker, text, output, style):
                calls.append(("tts", self.speaker_voices[speaker]))
                self.engine = "bodhan"
                return "bodhan"

            def labels(self):
                return dict(self.speaker_voices)

        with TemporaryDirectory() as temp:
            (Path(temp) / 'test').mkdir()
            settings = SimpleNamespace(
                jobs_dir=Path(temp),
                max_video_minutes=10,
                max_video_height=720,
                model_request_workers=1,
                diarization_backend='speechbrain',
                keep_background_default=False,
                max_upload_mb=100,
                lip_sync_model='wav2lip',
                background_mode='separate',
                separation_max_minutes=20,
                mouth_enhance=False,
                esrgan_weights=Path(temp) / 'missing.pth',
                translation_script='native',
                timing_fit=False,
                timing_mode='exact',
                fit_min_speed=0.9,
                fit_max_speed=1.1,
                diarize_dialogue_stem=False,
                speaker_review=False,
                script_supervision=False,
            )

            def diarize(*args, **kwargs):
                calls.append(("diarization", ""))
                return [Segment(0, 1, "SPEAKER_00"), Segment(2, 3, "SPEAKER_01")], "test"

            def mix(*args):
                calls.append(("mix", ""))
                return Path(temp) / "dub.wav"

            def lip(*args, **kwargs):
                calls.append(("lip_sync", ""))
                return Path(temp) / "dub.mp4"

            with (
                patch.object(pipeline, "settings", settings),
                patch.object(pipeline, "store", FakeStore()),
                patch.object(pipeline, "acquire_source", return_value=Path(temp) / "source.mp4"),
                patch.object(
                    pipeline, "probe", return_value={"duration": 3, "width": 0, "height": 0}
                ),
                patch.object(pipeline, "make_poster"),
                patch.object(pipeline, "write_preview"),
                patch.object(pipeline, "speaker_profiles", return_value={}),
                patch.object(pipeline, "classify_genders", return_value={}),
                patch.object(pipeline, "extract_audio"),
                patch.object(pipeline, "cut_audio"),
                patch.object(pipeline, "speaker_turns", side_effect=diarize),
                patch.object(pipeline, "BodhanAPI", return_value=FakeClient()),
                patch.object(pipeline, "transcribe", side_effect=fake_asr),
                patch.object(pipeline, "translate", side_effect=fake_translation),
                patch.object(pipeline, "VoiceState", FakeVoices),
                patch.object(pipeline, "fit_lines", return_value=FitReport()),
                patch.object(pipeline, "assemble_timeline", side_effect=mix),
                patch.object(pipeline, "synchronize", side_effect=lip),
            ):
                pipeline.process_job("test")

        self.assertEqual(FakeStore.state.get("status"), "completed")
        stages = [
            name
            for name, _ in calls
            if name in {"diarization", "asr", "translate", "tts", "mix", "lip_sync"}
        ]
        self.assertEqual(
            stages,
            [
                "diarization",
                "asr",
                "asr",
                "translate",
                "translate",
                "tts",
                "tts",
                "mix",
                "lip_sync",
            ],
        )
        self.assertEqual([value for name, value in calls if name == "tts"], ["Kavya", "Amit"])


if __name__ == "__main__":
    unittest.main()
