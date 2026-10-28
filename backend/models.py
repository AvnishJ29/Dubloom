"""Shared data: supported languages, the voice catalogue, job statuses and the Segment line."""

from dataclasses import dataclass, asdict, field
from typing import Optional

LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

from backend.voices.catalog import load_catalog

VOICE_CATALOG = load_catalog()
VOICES = VOICE_CATALOG["voices"]
VOICE_GENDERS = VOICE_CATALOG.get("genders", {})

ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class Segment:
    """A speaker turn with source timing, script, voice, and rendered-line metadata."""

    start: float
    end: float
    speaker: str
    text: str = ""
    translation: str = ""
    voice: str = ""
    timing_ratio: float = 1.0
    fit_ratio: float = 1.0  # natural dubbed duration / original slot, before the final stretch
    audio_path: Optional[str] = None
    voiced_text: str = ""  # the translation the current TTS clip was generated from
    voiced_style: str = ""  # delivery style the clip was generated with
    tts_engine: str = ""  # engine that produced the clip (bodhan, clone, gemini)
    tts_speed: float = 1.0  # native speed requested from the voice engine
    dub_start: Optional[float] = None  # where the line was finally placed on the timeline
    dub_end: Optional[float] = (
        None  # ... and where its audio ends (may run past the slot into the gap)
    )
    timing_trimmed: bool = False  # the final mix had to cut the audio tail
    timing_borrowed: float = 0.0  # seconds used from the following pause
    edited: bool = False  # translation changed by the user in the script editor
    source_edited: bool = (
        False  # original line corrected by the user; translated again from the new words
    )
    style_override: str = ""  # delivery chosen for this one line in the director panel
    take: str = ""  # the take file the line's clip came from, if the user picked one
    style: str = ""  # delivery style of this line from the speaker review (emotion)
    reference_text: str = ""  # independent transcript from the speaker review (OpenRouter)
    asr_text: str = ""  # Bodhan's transcript when the transcript check corrected it
    text_source: str = (
        ""  # 'review' when the line's words came from the reviewer because Bodhan returned junk
    )
    machine_translation: str = ""  # Bodhan's translation when the script editor corrected it
    condensed: bool = False  # shortened by the timing fit so it fits its slot
    state: str = ""  # furthest step this line has reached: heard, translated, voiced, lip-synced
    ai_edited: bool = (
        False  # the user asked the AI to rewrite this line (Ask AI), as opposed to typing it
    )
    history: list = field(
        default_factory=list
    )  # earlier words of this line: [{at, field, by, text}], newest last; nothing is deleted

    def public(self):
        """Serialize the line for job records without exposing its local audio path."""
        data = asdict(self)
        data.pop("audio_path")
        return data
