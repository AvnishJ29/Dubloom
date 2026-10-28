"""Exercise actual encoded media and HTTP downloads, including silent exports."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from backend.api import app, helpers, system
from backend.media.media_validation import validate_video


class MediaDeliveryTest(unittest.TestCase):
    def test_export_audio_validation_and_mp4_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_id = str(uuid4())
            job = root / job_id
            job.mkdir()
            output = job / 'dubbed_video.mp4'
            self.encode(output, 'sine=frequency=440:sample_rate=48000')
            report = validate_video(output, 1)
            self.assertGreater(report['audio_peak_db'], -65)
            settings = SimpleNamespace(jobs_dir=root)
            with (
                patch.object(helpers, 'settings', settings),
                patch.object(system, 'settings', settings),
            ):
                client = TestClient(app)
                response = client.get(f'/api/jobs/{job_id}/video?download=true')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['content-type'], 'video/mp4')
                self.assertIn('attachment;', response.headers['content-disposition'])
                self.assertIn('.mp4', response.headers['content-disposition'])
                self.assertEqual(response.content, output.read_bytes())
                preview = client.get(f'/api/jobs/{job_id}/video', headers={'Range': 'bytes=0-99'})
                self.assertEqual(preview.status_code, 206)
                self.assertEqual(len(preview.content), 100)
                self.assertIn('inline;', preview.headers['content-disposition'])
            self.encode(output, 'anullsrc=r=48000:cl=mono')
            with self.assertRaisesRegex(RuntimeError, 'silent|quiet'):
                validate_video(output, 1)

    @staticmethod
    def encode(output, audio):
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-v',
                'error',
                '-f',
                'lavfi',
                '-i',
                'color=c=black:s=160x120:r=25',
                '-f',
                'lavfi',
                '-i',
                audio,
                '-t',
                '1',
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-c:a',
                'aac',
                str(output),
            ],
            check=True,
        )


if __name__ == '__main__':
    unittest.main()
