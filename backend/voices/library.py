"""The voice library (backend/voices/data/voices.md): who each voice is, and which voice fits which speaker.

Casting scores every voice of the speaker's gender on: native language first, then how well the voice's
"best for" matches the content's genre, then how close its register is to the speaker's pitch. Voices are
never repeated while a distinct one is left. Every choice comes with a one-line reason for the UI.
"""

from __future__ import annotations

import re
from pathlib import Path

LIBRARY_PATH = Path(__file__).with_name("data") / "voices.md"

GENRE_HINTS = {  # genre -> words that appear in the cards' "best for" lists
    "lecture": ("e-learning", "lecture", "explainer", "tutorial", "educational", "documentation"),
    "news": ("news", "bulletin", "announcement"),
    "film": ("dramatic", "trailer", "theatrical", "performance", "audiobook", "storytelling"),
    "advert": ("advertisement", "promo", "persuasive"),
    "documentary": ("documentar", "narration", "narrator"),
    "story": ("storytelling", "audiobook", "children"),
    "interview": ("conversational", "explainer", "news"),
    "service": ("customer care", "ivr", "social"),
}

_CARD = re.compile(
    r"^\*\*(\w+)\*\* — (Female|Male) · ~(\d+) Hz · ([\d.]+) chars/sec\.\s*(.*)", re.S | re.M
)


def load_library(path: Path = LIBRARY_PATH) -> dict[str, dict]:
    """Parse the voice cards of voices.md into {name: card}.

    A card holds gender, language (the ## heading it sits under), pitch in Hz, speaking rate in
    characters per second, description and "best for" list.
    """
    text = path.read_text(encoding="utf-8")
    library, language = {}, ""
    blocks = re.split(r"(?=^## |^\*\*\w+\*\* — )", text, flags=re.M)
    for block in blocks:
        if block.startswith("## "):
            language = block[3:].splitlines()[0].strip()
            continue
        match = _CARD.match(block)
        if not match:
            continue
        name, gender, hz, rate, rest = match.groups()
        description, _, tail = rest.partition("*Best for:*")
        best = tail.split("Data note:")[0].strip().rstrip(".")
        library[name] = {
            "name": name,
            "gender": gender.lower(),
            "language": language,
            "pitch_hz": int(hz),
            "chars_per_second": float(rate),
            "description": " ".join(description.split()).rstrip(". "),
            "best_for": [" ".join(item.split()) for item in best.split(",") if item.strip()],
        }
    return library


LIBRARY = load_library() if LIBRARY_PATH.is_file() else {}
LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "en": "Hindi",  # English lines are voiced by the Hindi cards (same voices in the catalogue)
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
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


def speaking_rate(voice: str, default: float = 12.5) -> float:
    """Return a voice-specific character rate, or the supplied fallback."""
    return LIBRARY.get(voice, {}).get("chars_per_second") or default


def score(
    voice: dict, language: str, genre: str | None, pitch_hz: float | None
) -> tuple[float, str]:
    """Higher is better. Returns (score, reason)."""
    reasons, total = [], 0.0
    if voice["language"] == LANGUAGE_NAMES.get(language):
        total += 10
        reasons.append(f"native {voice['language']}")
    if genre:
        best = " ".join(voice["best_for"]).lower()
        hits = [hint for hint in GENRE_HINTS.get(genre, ()) if hint in best]
        if hits:
            total += 4 + len(hits)
            reasons.append(f"made for {voice['best_for'][0]}")
    if pitch_hz:
        closeness = max(
            0.0, 1 - abs(voice["pitch_hz"] - pitch_hz) / 120
        )  # same register within ~120 Hz
        total += 3 * closeness
        if closeness > 0.6:
            reasons.append("similar register")
    return total, ", ".join(reasons) or "a distinct voice"


def cast(
    speakers: list[str],
    language: str,
    profiles: dict | None = None,
    genre: str | None = None,
    preferred: str | list[str] | None = None,
    overrides: dict | None = None,
    pool: list[str] | None = None,
) -> tuple[dict, dict]:
    """Assign one library voice per speaker. Returns ({speaker: voice}, {speaker: reason}).
    `preferred`: the user's voices in speaking order (a single name means the first speaker); their choice wins as is.
    """
    profiles = profiles or {}
    chosen_order = [preferred] if isinstance(preferred, str) else list(preferred or [])
    candidates = [LIBRARY[name] for name in (pool or list(LIBRARY)) if name in LIBRARY]
    assignments, reasons, used = {}, {}, set()
    for speaker, voice in (overrides or {}).items():
        if speaker in speakers and voice in LIBRARY:
            assignments[speaker], reasons[speaker] = voice, "chosen by you"
            used.add(voice)
    ordered = sorted(speakers, key=lambda s: profiles.get(s, {}).get("first_seen", 0))
    for index, speaker in enumerate(ordered):
        if speaker in assignments:
            continue
        profile = profiles.get(speaker, {})
        wanted = profile.get("gender") if profile.get("gender") in ("male", "female") else None
        if (
            index < len(chosen_order)
            and chosen_order[index] in LIBRARY
            and chosen_order[index] not in used
        ):
            assignments[speaker], reasons[speaker] = chosen_order[index], "chosen by you"
            used.add(chosen_order[index])
            continue
        options = (
            [
                v
                for v in candidates
                if v["name"] not in used and (not wanted or v["gender"] == wanted)
            ]
            or [v for v in candidates if not wanted or v["gender"] == wanted]
            or candidates
        )
        ranked = sorted(
            options,
            key=lambda v: score(v, language, genre, profile.get("pitch_hz"))[0],
            reverse=True,
        )
        best = ranked[0]
        assignments[speaker] = best["name"]
        reasons[speaker] = (
            f"{best['gender']} · {best['description']} · "
            + score(best, language, genre, profile.get("pitch_hz"))[1]
        )
        used.add(best["name"])
    return {s: assignments[s] for s in speakers}, reasons
