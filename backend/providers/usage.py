"""Per-job provider usage and cost estimates."""

from __future__ import annotations

import threading

# Published Bodhan prices (console.bodhan.ai, rupees).
BODHAN_ASR_INR_PER_SECOND = 0.10 / 60
BODHAN_TRANSLATE_INR_PER_OUTPUT_TOKEN = 0.20 / 10_000
BODHAN_TTS_INR_PER_CHARACTER = 6.0 / 10_000


class Usage:
    """Accumulate per-job usage safely across concurrent model requests."""

    def __init__(self, initial: dict | None = None):
        self.lock = threading.Lock()
        self.data: dict[str, float] = dict(initial or {})

    def add(self, key: str, amount: float):
        if not amount:
            return
        with self.lock:
            self.data[key] = round(self.data.get(key, 0) + float(amount), 6)

    def bodhan_cost_inr(self) -> float:
        with self.lock:
            return round(
                self.data.get("bodhan_asr_seconds", 0) * BODHAN_ASR_INR_PER_SECOND
                + self.data.get("bodhan_translate_tokens", 0)
                * BODHAN_TRANSLATE_INR_PER_OUTPUT_TOKEN
                + self.data.get("bodhan_tts_characters", 0) * BODHAN_TTS_INR_PER_CHARACTER,
                3,
            )

    def as_dict(self) -> dict:
        with self.lock:
            data = dict(self.data)
        data["bodhan_cost_inr"] = self.bodhan_cost_inr()
        return data


def estimate_bodhan_inr(audio_seconds: float, characters: int, output_tokens: int = 0) -> float:
    return round(
        audio_seconds * BODHAN_ASR_INR_PER_SECOND
        + characters * BODHAN_TTS_INR_PER_CHARACTER
        + output_tokens * BODHAN_TRANSLATE_INR_PER_OUTPUT_TOKEN,
        2,
    )
