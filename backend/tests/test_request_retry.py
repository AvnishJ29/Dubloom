"""Rate-limited multipart requests must resend the original audio bytes."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from backend.providers.bodhan_api import BodhanAPI, BodhanRateLimitError, pacer_for


class RequestRetryTest(unittest.TestCase):
    def test_upload_is_rewound_for_retry(self):
        api = BodhanAPI()
        received = []

        def post(path, **kwargs):
            received.append(kwargs['files']['file'][1].read())
            return httpx.Response(429 if len(received) == 1 else 200, headers={'retry-after': '0'})

        try:
            with patch.object(api.client, 'post', side_effect=post):
                response = api.post(
                    '/test',
                    'test-key',
                    files={'file': ('sample.wav', io.BytesIO(b'audio'), 'audio/wav')},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(received, [b'audio', b'audio'])
        finally:
            api.close()

    def test_exhausted_rate_limit_is_distinct_from_other_api_errors(self):
        api = BodhanAPI()
        try:
            with (
                patch.object(api.client, 'post', return_value=httpx.Response(429)),
                patch('backend.providers.bodhan_api.time.sleep'),
            ):
                with self.assertRaises(BodhanRateLimitError):
                    api.post('/test', 'test-key', rate_limit_attempts=2)
        finally:
            api.close()


class FakeClock:
    """time.sleep advances time.monotonic, so waits are measured without waiting."""

    def __init__(self):
        self.now = 1000.0

    def sleep(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now


class RateLimitWaitTest(unittest.TestCase):
    """Bodhan's model-level limit ("Maximum rate limit reached for the model") comes without retry-after and lasts
    tens of seconds; a dub must wait it out, not fail after a few seconds."""

    def test_a_model_level_limit_without_retry_after_is_waited_out(self):
        api = BodhanAPI()
        clock = FakeClock()
        answers = [
            httpx.Response(429, json={"error": "Maximum rate limit reached for the model"})
        ] * 5 + [httpx.Response(200)]
        try:
            with (
                patch.object(api.client, 'post', side_effect=answers),
                patch('backend.providers.bodhan_api.time.sleep', clock.sleep),
                patch('backend.providers.bodhan_api.time.monotonic', clock.monotonic),
            ):
                response = api.post('/test', 'storm-key')
            self.assertEqual(response.status_code, 200)
            self.assertGreaterEqual(clock.now - 1000.0, 130)  # 5 + 10 + 20 + 40 + 60 s
        finally:
            api.close()

    def test_one_refusal_pauses_every_worker_on_the_key(self):
        api = BodhanAPI()
        clock = FakeClock()
        try:
            with (
                patch.object(api.client, 'post', return_value=httpx.Response(200)),
                patch('backend.providers.bodhan_api.time.sleep', clock.sleep),
                patch('backend.providers.bodhan_api.time.monotonic', clock.monotonic),
            ):
                pacer_for('held-key').hold(30)
                api.post('/test', 'held-key')
            self.assertGreaterEqual(clock.now - 1000.0, 30)
        finally:
            api.close()

    def test_transcription_survives_a_rate_limit_storm(self):
        from backend.speech import asr

        api = BodhanAPI()
        clock = FakeClock()
        answers = [httpx.Response(429)] * 4 + [
            httpx.Response(200, json={"text": "the running theme of this course"})
        ]
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "line.wav"
            audio.write_bytes(b"RIFF")
            try:
                with (
                    patch.object(api.client, 'post', side_effect=answers),
                    patch('backend.providers.bodhan_api.time.sleep', clock.sleep),
                    patch('backend.providers.bodhan_api.time.monotonic', clock.monotonic),
                ):
                    text = asr.transcribe(api, audio, "en")
            finally:
                api.close()
        self.assertEqual(text, "the running theme of this course")
