"""Voice rendering for a job: Bodhan Indic-Speak first, OpenRouter voices when Bodhan cannot answer.

One engine voices a whole project so the voices stay consistent. `VoiceState`
remembers which engine is in use; if Bodhan drops out mid-way the pipeline
re-voices the earlier lines with the engine that took over.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from backend.providers import openrouter
from backend.providers.bodhan_api import BodhanAPI, BodhanRateLimitError
from backend.models import VOICE_GENDERS
from backend.speech.openrouter_tts import (
    GEMINI_VOICES,
    OPENROUTER_LANGUAGES,
    build_reference,
    clone_synthesize,
    gemini_synthesize,
)
from backend.speech.tts import synthesize as bodhan_synthesize

logger = logging.getLogger(__name__)
ENGINE_LABELS = {
    "bodhan": "Bodhan Indic-Speak",
    "gemini": "studio voice",
    "clone": "original voice (cloned)",
}


def clone_available(language: str) -> bool:
    """True when original-voice cloning through OpenRouter is possible for `language`."""
    return openrouter.configured() and language in OPENROUTER_LANGUAGES


def gemini_casts(speaker_voices: dict[str, str]) -> dict[str, str]:
    """A distinct Gemini voice per speaker, matching the register of the cast Bodhan voice."""
    counters = {"female": 0, "male": 0}
    casts = {}
    for speaker, voice in speaker_voices.items():
        gender = VOICE_GENDERS.get(voice, "female")
        gender = gender if gender in counters else "female"
        casts[speaker] = GEMINI_VOICES[gender][counters[gender] % len(GEMINI_VOICES[gender])]
        counters[gender] += 1
    return casts


def voice_references(directory: Path, segments, speakers, clean_audio: Path) -> dict:
    """One clean sample per speaker (longest turn, at most 14 s) for voice cloning."""
    references = {}
    for speaker in speakers:
        turns = sorted(
            (s for s in segments if s.speaker == speaker and s.end - s.start >= 2.5),
            key=lambda s: s.end - s.start,
            reverse=True,
        )
        if not turns:
            continue
        turn = turns[0]
        target = directory / f"ref_{speaker}.wav"
        try:
            if not target.is_file():
                build_reference(clean_audio, turn.start, turn.end, target)
            references[speaker] = {
                "path": str(target),
                "text": turn.text if turn.end - turn.start <= 14 else None,
            }
        except RuntimeError as exc:
            logger.warning("No voice reference for %s: %s", speaker, exc)
    return references


class VoiceState:
    """Track voice rendering, provider fallback, and casting consistently across a job."""

    def __init__(
        self,
        client: BodhanAPI,
        language: str,
        speaker_voices: dict[str, str],
        mode: str = "auto",
        references: dict | None = None,
        usage=None,
    ):
        self.client = client
        self.language = language
        self.speaker_voices = speaker_voices
        self.gemini = gemini_casts(speaker_voices)
        self.references = references or {}
        self.usage = usage
        self.mode = mode  # auto: Bodhan then OpenRouter; clone: original voices
        self.engine: str | None = None  # engine that is voicing this project
        self.bodhan_error: str | None = None
        self.lock = threading.Lock()

    def labels(self) -> dict[str, str]:
        """Display label per speaker for the engine that voices (or will voice) the project."""
        engine = self.engine or ("clone" if self.mode == "clone" else "bodhan")
        out = {}
        for speaker, voice in self.speaker_voices.items():
            if engine == "bodhan":
                out[speaker] = voice
            elif engine == "clone" and speaker in self.references:
                out[speaker] = "original voice, cloned"
            else:
                out[speaker] = f"studio voice {self.gemini[speaker]}"
        return out

    def _openrouter_available(self):
        return openrouter.configured() and self.language in OPENROUTER_LANGUAGES

    def render(self, speaker: str, text: str, output: Path, style: str | None) -> str:
        """Voice one line; returns the engine used."""
        with self.lock:
            engine = self.engine
        if engine is None:
            engine = "clone" if self.mode == "clone" and self._openrouter_available() else "bodhan"
        if engine == "bodhan":
            try:
                bodhan_synthesize(
                    self.client,
                    text,
                    self.language,
                    self.speaker_voices[speaker],
                    output,
                    style=style,
                    usage=self.usage,
                )
                with self.lock:
                    self.engine = "bodhan"
                return "bodhan"
            except BodhanRateLimitError as exc:
                with self.lock:
                    self.bodhan_error = str(exc)[:200]
                if not (openrouter.covers_bodhan() and self.language in OPENROUTER_LANGUAGES):
                    raise
                logger.warning("Bodhan voices unavailable (%s); OpenRouter voices take over", exc)
                engine = "clone" if self.mode == "clone" else "gemini"
        if engine == "clone" and speaker in self.references:
            reference = self.references[speaker]
            clone_synthesize(
                text, Path(reference["path"]), output, style, reference.get("text"), self.usage
            )
        else:
            engine = "gemini" if engine != "clone" else "clone"
            gemini_synthesize(text, self.gemini[speaker], output, style, self.usage)
        with self.lock:
            self.engine = engine
        return engine
