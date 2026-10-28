"""Director takes: alternate readings of one line (a delivery style, optionally new words), voiced on demand so
the user can listen and pick before anything is re-rendered. The chosen take replaces the line's clip.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from backend.providers.bodhan_api import BodhanAPI
from backend.jobs.versions import remember
from backend.speech.styles import normalise_style
from backend.speech.tts import synthesize

MAX_TAKES_PER_LINE = 8


def _seconds(path: Path) -> float:
    """Media duration in seconds (ffprobe), rounded to 0.01; 0.0 when unknown."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return round(float(out), 2) if out else 0.0


def generate_take(
    job: dict,
    directory: Path,
    index: int,
    style: str | None,
    text: str | None,
    client: BodhanAPI | None = None,
) -> dict:
    """Voice one more take of line `index`. Returns the take record; the caller stores it under job['takes']."""
    segment = (job.get("segments") or [])[index]
    voice = (job.get("speaker_voices") or {}).get(segment["speaker"])
    if not voice:
        raise ValueError("This speaker has no voice yet")
    words = (text or segment.get("translation") or "").strip()
    if not words:
        raise ValueError("The line has no text to voice")
    style = normalise_style(style)
    existing = (job.get("takes") or {}).get(str(index)) or []
    if len(existing) >= MAX_TAKES_PER_LINE:
        raise ValueError(f"At most {MAX_TAKES_PER_LINE} takes per line; pick one or delete some")
    number = max((t["number"] for t in existing), default=0) + 1
    folder = directory / "takes"
    folder.mkdir(exist_ok=True)
    output = folder / f"line_{index:04d}_take_{number}.wav"
    synthesize(client or BodhanAPI(), words, job["language"], voice, output, style=style)
    return {
        "number": number,
        "style": style,
        "text": words,
        "voice": voice,
        "file": output.name,
        "seconds": _seconds(output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slot_seconds": round(segment["end"] - segment["start"], 2),
    }


def choose_take(job: dict, directory: Path, index: int, number: int) -> dict:
    """Make the take the line's clip. Returns the segment changes to store; a re-render applies them."""
    takes = (job.get("takes") or {}).get(str(index)) or []
    take = next((t for t in takes if t["number"] == number), None)
    if not take:
        raise ValueError("No such take")
    source = directory / "takes" / take["file"]
    if not source.is_file():
        raise ValueError("The take's audio is gone; generate it again")
    shutil.copyfile(source, directory / f"tts_{index:04d}.wav")
    segment = dict((job.get("segments") or [])[index])
    changed_text = take["text"] != (segment.get("translation") or "")
    if changed_text:
        remember(segment, "translation")  # the words the take replaces stay in the line's history
    segment.update(
        translation=take["text"],
        edited=segment.get("edited") or changed_text,
        style_override=take["style"],
        voiced_text=take["text"],
        voiced_style=take["style"],
        take=take["file"],
        audio_path=str(directory / f"tts_{index:04d}.wav"),
    )
    return segment
