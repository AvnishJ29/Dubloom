"""The Bodhan integration must match the published API reference (console.bodhan.ai/api-docs)."""

import io
import json
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from backend.speech import asr
from backend.speech import translation
from backend.speech import tts
from backend.providers.bodhan_api import BodhanAPI, BodhanBudgetError, BodhanRateLimitError
from backend.providers.usage import Usage


def wav_bytes(seconds=0.5, rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * int(rate * seconds))
    return buffer.getvalue()


def api_with(handler):
    api = BodhanAPI()
    api.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test")
    return api


class BodhanSpeechTest(unittest.TestCase):
    def test_lines_are_chunked_per_sentence_with_language_and_style_instructions(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

        text = (
            "मुझे पता है कि आप सभी गेट की तैयारी कर रहे हैं। " * 6
        ).strip()  # far more than two sentences
        with TemporaryDirectory() as temp, patch.object(tts, "settings", asr.settings):
            output = Path(temp) / "line.wav"
            usage = Usage()
            tts.synthesize(
                api_with(handler), text, "hi", "Kavya", output, style="angry", usage=usage
            )
            with wave.open(str(output), "rb") as stream:
                self.assertEqual(stream.getframerate(), 24000)
                self.assertGreater(
                    stream.getnframes(), 24000 * 0.5 * len(requests)
                )  # pieces plus pauses
        self.assertGreaterEqual(len(requests), 3)
        for body in requests:
            self.assertEqual(body["model"], "indic-speak")
            self.assertEqual(body["voice"], "Kavya")
            self.assertLessEqual(body["input"].count("।"), 2)
            self.assertEqual(json.loads(body["instructions"]), {"lang": "hi", "style": "anger"})
            self.assertNotIn("speed", body)
        self.assertEqual(
            usage.data["bodhan_tts_characters"], sum(len(body["input"]) for body in requests)
        )

    def test_neutral_style_sends_only_the_language(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

        with TemporaryDirectory() as temp:
            tts.synthesize(
                api_with(handler), "Class starts at nine.", "en", "Arun", Path(temp) / "l.wav"
            )
        self.assertEqual(json.loads(seen["instructions"]), {"lang": "en"})


class BodhanTranslateTest(unittest.TestCase):
    def test_product_route_with_codemix_script_and_usage(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "translation": "मीटिंग postpone हो गई है।",
                    "target_language": "hi",
                    "script": "codemix",
                    "truncated": False,
                    "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
                },
            )

        usage = Usage()
        result = translation.translate(
            api_with(handler), "The meeting has been postponed.", "hi", "codemix", usage=usage
        )
        self.assertEqual(result, "मीटिंग postpone हो गई है।")
        self.assertEqual(seen["path"], "/translate")
        self.assertEqual(
            seen["body"],
            {
                "text": "The meeting has been postponed.",
                "target_language": "hi",
                "script": "codemix",
            },
        )
        self.assertEqual(usage.data["bodhan_translate_tokens"], 9)

    def test_chat_transport_when_product_route_is_absent_and_truncation_is_an_error(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/translate":
                return httpx.Response(404, json={"detail": "Not Found"})
            body = json.loads(request.content)
            self.assertEqual(body["model"], "indic-translate")
            self.assertEqual(body["target_language_code"], "kn")
            self.assertEqual(body["source_language_code"], "en")
            self.assertEqual(
                body["messages"], [{"role": "user", "content": "Class starts at nine."}]
            )
            return httpx.Response(
                200, json={"choices": [{"finish_reason": "length", "message": {"content": "ತರಗತಿ"}}]}
            )

        with self.assertRaisesRegex(RuntimeError, "truncated"):
            translation.translate(
                api_with(handler), "Class starts at nine.", "kn", source_language="en"
            )
        self.assertEqual(calls, ["/translate", "/v1/chat/completions"])


class BodhanTranscribeTest(unittest.TestCase):
    def test_language_hint_is_sent_only_for_supported_codes_and_audio_seconds_are_metered(self):
        seen = {}

        def handler(request):
            seen["content_type"] = request.headers.get("content-type", "")
            seen["body"] = request.content
            return httpx.Response(200, json={"text": "नमस्ते, आप कैसे हैं?"})

        with TemporaryDirectory() as temp:
            clip = Path(temp) / "turn.wav"
            clip.write_bytes(wav_bytes(seconds=2.0, rate=16000))
            usage = Usage()
            text = asr.transcribe(api_with(handler), clip, "hi", usage=usage)
            self.assertEqual(text, "नमस्ते, आप कैसे हैं?")
            self.assertIn("multipart/form-data", seen["content_type"])
            self.assertIn(b'name="language"\r\n\r\nhi', seen["body"])
            self.assertIn(b'name="model"\r\n\r\nindic-transcribe', seen["body"])
            self.assertAlmostEqual(usage.data["bodhan_asr_seconds"], 2.0, places=2)
            self.assertAlmostEqual(usage.bodhan_cost_inr(), 2.0 * 0.10 / 60, places=3)
            asr.transcribe(api_with(handler), clip, "xx")
            self.assertNotIn(b'name="language"', seen["body"])


class BodhanErrorEnvelopeTest(unittest.TestCase):
    def test_documented_error_codes_map_to_the_right_exceptions(self):
        def rate_limited(request):
            return httpx.Response(
                429,
                headers={"retry-after": "1"},
                json={"error": {"message": "slow down", "code": "rate_limit_exceeded_error"}},
            )

        def budget(request):
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "ExceededBudget: over budget. Spend=₹10.01, Budget=₹10.0",
                        "type": "budget_exceeded",
                        "code": "429",
                    }
                },
            )

        def bad_language(request):
            return httpx.Response(
                400, json={"error": {"message": "unknown code", "code": "unsupported_language"}}
            )

        with patch("backend.providers.bodhan_api.time.sleep"):
            with self.assertRaises(BodhanRateLimitError):
                api_with(rate_limited).post("/translate", "k", rate_limit_attempts=2, json={})
        with self.assertRaises(BodhanBudgetError):
            api_with(budget).post("/translate", "k", json={})
        with self.assertRaisesRegex(RuntimeError, "unknown code"):
            api_with(bad_language).post("/translate", "k", json={})


if __name__ == "__main__":
    unittest.main()


class PacerTest(unittest.TestCase):
    def test_the_published_limit_is_learned_and_respected(self):
        from backend.providers.bodhan_api import KeyPacer

        pacer = KeyPacer(per_minute=1000, parallel=4)
        pacer.learn({"x-ratelimit-api_key-limit-requests": "120"})
        self.assertEqual(pacer.per_minute, 108)
        for _ in range(3):
            pacer.acquire()
            pacer.release()
        self.assertEqual(len(pacer.sent), 3)
