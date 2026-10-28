import unittest

import httpx

from backend.speech.translation import translate


class TranslationFallbackTest(unittest.TestCase):
    def test_chat_route_used_when_product_route_is_missing(self):
        calls = []

        class FakeAPI:
            def post(self, path, key, **kwargs):
                calls.append((path, kwargs))
                if path == "/translate":
                    return httpx.Response(404)
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": "नमस्ते"}},
                        ]
                    },
                )

        self.assertEqual(translate(FakeAPI(), "Hello", "hi"), "नमस्ते")
        self.assertEqual([path for path, _ in calls], ["/translate", "/v1/chat/completions"])
        self.assertEqual(calls[1][1]["json"]["target_language_code"], "hi")


if __name__ == "__main__":
    unittest.main()
