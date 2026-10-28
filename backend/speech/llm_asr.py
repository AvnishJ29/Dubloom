"""Speech recognition through OpenRouter (Gemini Transcribe, then a chat model that hears audio)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from backend.providers import openrouter
from backend.config import settings
from backend.models import LANGUAGES

logger = logging.getLogger(__name__)
ISO_639_1 = {
    "as",
    "bn",
    "en",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sd",
    "ta",
    "te",
    "ur",
    "ks",
}


def _prompt(source_language: str | None) -> str:
    language = LANGUAGES.get(source_language or "", "")
    hint = (
        f" The language is {language}; write it in its native script."
        if language
        else " Keep the original spoken language and write it in its native script."
    )
    return (
        "Transcribe this audio verbatim."
        + hint
        + " Output only the transcript with normal punctuation. Do not translate, describe sounds, "
        "add speaker labels or timestamps. If there is no intelligible speech, output nothing."
    )


def transcribe_clip(
    audio: Path, source_language: str | None = None, state: dict | None = None, usage=None
) -> str:
    """Transcribe one audio clip through OpenRouter; "" when there is no speech.

    The speech-to-text model is tried first. Once it fails, `state["stt_failed"]` is set so later
    clips sharing `state` go straight to the chat model that hears audio. Raises RuntimeError when
    both routes fail; costs are added to `usage`.
    """
    state = state if state is not None else {}
    data = base64.b64encode(audio.read_bytes()).decode()
    fmt = audio.suffix.lstrip(".").lower() or "wav"
    last_error: Exception | None = None
    if settings.openrouter_asr_model and not state.get("stt_failed"):
        body = {
            "model": settings.openrouter_asr_model,
            "input_audio": {"data": data, "format": fmt},
        }
        if source_language in ISO_639_1:
            body["language"] = source_language
        try:
            result = openrouter.post_json("/audio/transcriptions", body)
            if usage is not None:
                usage.add("openrouter_cost_usd", (result.get("usage") or {}).get("cost") or 0)
            return (result.get("text") or "").strip()
        except (openrouter.OpenRouterError, KeyError, ValueError, OSError) as exc:
            last_error = exc
            state["stt_failed"] = str(exc)[:160]
            logger.warning("OpenRouter speech-to-text failed (%s); using the chat audio route", exc)
    try:
        text = openrouter.chat(
            settings.openrouter_asr_chat_model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt(source_language)},
                        {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
                    ],
                }
            ],
            usage=usage,
            temperature=0,
        )
        if text.lower().strip("()[].") in {"nothing", "no speech", "no intelligible speech"}:
            text = ""
        return text
    except (openrouter.OpenRouterError, KeyError, ValueError, OSError) as exc:
        raise RuntimeError(
            f"OpenRouter transcription failed: {exc} (speech-to-text route: {last_error})"
        ) from exc
