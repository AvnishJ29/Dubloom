"""Every finished render is kept. Before a re-render, the current output moves into versions/<n>/ together with the
script and the settings that produced it, so older dubs stay playable, downloadable and comparable.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

KEEP = ("dubbed_video.mp4", "preview_video.mp4", "dubbed_audio.wav", "lip_sync_report.json")
SETTINGS = (
    "speaker_voices",
    "speaker_styles",
    "delivery_style",
    "lip_sync_model",
    "background_mode",
    "translation_script",
    "tts_engine",
    "enhance_mouth",
)


def archive(job: dict, directory: Path, reason: str) -> list[dict]:
    """Move the current render into the next version folder. Returns the versions list for the job record."""
    versions = list(job.get("versions") or [])
    if job.get("status") != "completed" or not (directory / "dubbed_video.mp4").is_file():
        return versions
    number = len(versions) + 1
    folder = directory / "versions" / str(number)
    folder.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (directory / name).is_file():
            shutil.move(str(directory / name), str(folder / name))  # a rename: instant, no copy
    lip = job.get("lip_sync") or {}
    entry = {
        "number": number,
        "rendered_at": job.get("finished_at") or job.get("updated_at"),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "settings": {key: job.get(key) for key in SETTINGS},
        "lines": len(job.get("segments") or []),
        "lip_sync": {
            key: lip.get(key) for key in ("model", "lines_lip_synced", "coverage_percent")
        },
        "has_preview": (folder / "preview_video.mp4").is_file(),
    }
    (folder / "version.json").write_text(
        json.dumps({**entry, "segments": job.get("segments") or []}, ensure_ascii=False),
        encoding="utf-8",
    )
    versions.append(entry)
    return versions


def describe_revoice(job: dict, request) -> str:
    """One line saying what the user changed, for the history list."""
    parts = []
    names = {
        speaker: (job.get("speaker_profiles") or {}).get(speaker, {}).get("name")
        or speaker.replace("SPEAKER_", "Speaker ").replace("Speaker 0", "Speaker ")
        for speaker in set(request.speaker_voices) | set(request.speaker_styles)
    }
    for speaker, voice in request.speaker_voices.items():
        parts.append(
            f"{names[speaker]}: voice {(job.get('speaker_voices') or {}).get(speaker, '?')} → {voice}"
        )
    for speaker, style in request.speaker_styles.items():
        parts.append(f"{names[speaker]}: tone → {style}")
    if request.delivery_style:
        parts.append(f"delivery → {request.delivery_style}")
    if request.lip_sync_model and request.lip_sync_model != job.get("lip_sync_model"):
        parts.append(f"lip sync → {request.lip_sync_model}")
    if request.background_mode and request.background_mode != job.get("background_mode"):
        parts.append(f"background → {request.background_mode}")
    if request.tts_engine and request.tts_engine != job.get("tts_engine"):
        parts.append(f"voices → {request.tts_engine}")
    if parts:
        return "; ".join(parts)
    takes = job.get("pending_takes") or 0
    return (
        f"Takes applied to {takes} line{'s' if takes != 1 else ''}"
        if takes
        else "Improved lip sync and timing"
    )


def hide_history_baseline(versions: list[dict], directory: Path) -> list[dict]:
    """One-time UI reset; all archived media remains retrievable on the backend."""
    hidden = []
    for version in versions:
        entry = {**version, "hidden_from_history": True}
        metadata = directory / "versions" / str(version["number"]) / "version.json"
        if metadata.is_file():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            record["hidden_from_history"] = True
            metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        hidden.append(entry)
    return hidden


def remember(segment: dict, which: str):
    """Before a line's words change, keep the current ones: who wrote them and when they were replaced."""
    previous = segment.get(which) or ""
    if not previous:
        return
    if which == "text":
        by = "you" if segment.get("source_edited") else "bodhan"
    else:
        by = (
            "ai"
            if segment.get("ai_edited")
            else (
                "you"
                if segment.get("edited")
                else (
                    "editor"
                    if segment.get("machine_translation")
                    and segment["machine_translation"] != previous
                    else "bodhan"
                )
            )
        )
    segment["history"] = list(segment.get("history") or []) + [
        {"at": datetime.now(timezone.utc).isoformat(), "field": which, "by": by, "text": previous}
    ]
