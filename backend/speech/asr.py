"""Speech-to-text with Bodhan Indic-Transcribe.

One request takes at most 30 seconds of audio and is billed per second, so
every diarized turn is sent on its own. Naming the spoken language is what
separates a correct transcript from a garbled one, so pass it whenever known.
"""

from pathlib import Path
import wave

import logging

from backend.providers import openrouter
from backend.providers.bodhan_api import BodhanAPI, BodhanRateLimitError
from backend.config import settings
from backend.speech.llm_asr import transcribe_clip

logger = logging.getLogger(__name__)

# Languages Bodhan Indic-Transcribe accepts in its `language` field.
BODHAN_ASR_LANGUAGES = {
    "as",
    "bhb",
    "bho",
    "bn",
    "brx",
    "doi",
    "en",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
}


def wav_seconds(audio: Path) -> float:
    """Read WAV duration for provider usage accounting; return zero if unreadable."""
    try:
        with wave.open(str(audio), "rb") as stream:
            return stream.getnframes() / float(stream.getframerate() or 1)
    except (OSError, wave.Error, EOFError):
        return 0.0


class DegenerateTranscript(RuntimeError):
    """The recogniser answered, but with unusable text (one character repeated, no words)."""


def degenerate(text: str) -> bool:
    """True for the failure mode where the model loops on one token (e.g. 'ୄୄୄୄ…') instead of transcribing."""
    body = "".join(text.split())
    if len(body) < 8:
        return False
    top = max(body.count(ch) for ch in set(body))
    return top / len(body) >= 0.5 or len(set(body)) <= 2


def dominant_script(text: str) -> str | None:
    """'latin', 'indic' (Devanagari to Sinhala blocks) or 'arabic' when at least 70% of the letters are of one script."""
    counts = {"latin": 0, "indic": 0, "arabic": 0}
    for ch in text:
        if ch.isascii() and ch.isalpha():
            counts["latin"] += 1
        elif "\u0900" <= ch <= "\u0dff":
            counts["indic"] += 1
        elif "\u0600" <= ch <= "\u06ff":
            counts["arabic"] += 1
    total = sum(counts.values())
    if total < 4:
        return None
    script, count = max(counts.items(), key=lambda item: item[1])
    return script if count >= 0.7 * total else None


def script_mismatch(text: str, reference: str) -> bool:
    """True when the recogniser wrote the speech in a different script than an independent transcript did: the
    failure mode where English speech comes back as Telugu or Tamil letters spelling the English sounds.
    """
    a, b = dominant_script(text), dominant_script(reference)
    return bool(a and b and a != b)


def guess_source_language(texts: list[str]) -> str | None:
    """'en' when the independent transcripts of a video are written in Latin letters; None otherwise (an Indic script
    does not say which Indic language)."""
    body = " ".join(t for t in texts if t)
    return "en" if dominant_script(body) == "latin" else None


DEGENERATE_RETRIES = 2


def transcribe(
    api: BodhanAPI,
    audio: Path,
    source_language: str | None = None,
    usage=None,
    state: dict | None = None,
) -> str:
    """Bodhan Indic-Transcribe; OpenRouter transcription when Bodhan cannot answer.

    `state` is shared across a job: it records the engine that answered and why Bodhan was skipped.
    """
    state = state if state is not None else {}
    if not state.get("bodhan_failed"):
        data = {"model": "indic-transcribe"}
        if source_language in BODHAN_ASR_LANGUAGES:
            data["language"] = source_language
        try:
            with audio.open("rb") as file:
                response = api.post(
                    "/v1/audio/transcriptions",
                    settings.bodhan_asr_key,
                    data=data,
                    files={"file": (audio.name, file, "audio/wav")},
                )
            if usage is not None:
                usage.add("bodhan_asr_seconds", wav_seconds(audio))
            state.setdefault("engine", "bodhan")
            text = (response.json().get("text") or "").strip()
            for attempt in range(DEGENERATE_RETRIES):
                if not degenerate(text):
                    break
                logger.warning(
                    "Bodhan ASR returned degenerate text for %s (%r); retrying",
                    audio.name,
                    text[:30],
                )
                with audio.open("rb") as file:
                    response = api.post(
                        "/v1/audio/transcriptions",
                        settings.bodhan_asr_key,
                        data=data,
                        files={"file": (audio.name, file, "audio/wav")},
                    )
                text = (response.json().get("text") or "").strip()
            if degenerate(text):
                raise DegenerateTranscript(text)
            return text
        except BodhanRateLimitError as exc:
            if not openrouter.covers_bodhan():
                raise
            state["bodhan_failed"] = str(exc)[:160]
            logger.warning("Bodhan ASR unavailable (%s); OpenRouter transcription takes over", exc)
    text = transcribe_clip(audio, source_language, state.setdefault("openrouter", {}), usage)
    state["engine"] = "openrouter"
    return text
