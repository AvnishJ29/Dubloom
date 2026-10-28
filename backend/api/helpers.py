"""Request validation, job lookup and job scheduling shared by the route modules."""

from __future__ import annotations

import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException

from backend.config import settings
from backend.jobs.pipeline import run_job
from backend.jobs.store import store, utcnow
from backend.models import LANGUAGES
from backend.rendering.lip_models import LIP_SYNC_MODELS, available_models
from backend.speech.styles import DELIVERY_STYLES
from backend.voices.assignment import voice_pool
from backend.voices.engine import clone_available

# Model stages of several jobs may overlap; CPU-heavy stages are bounded inside the pipeline.
executor = ThreadPoolExecutor(max_workers=settings.job_workers, thread_name_prefix="job")
BACKGROUND_MODES = {"separate", "duck", "none"}
TRANSLATION_SCRIPTS = {"native", "codemix"}


def validate_voices(names: list[str] | None, language: str) -> list[str]:
    """Validate requested casting against the target-language voice pool."""
    chosen = [name.strip() for name in (names or []) if name and name.strip()]
    pool = voice_pool(language)
    for name in chosen:
        if name not in pool:
            raise HTTPException(422, f"Unsupported voice {name}")
    return chosen[:12]


def validate_clip(start: float | None, end: float | None) -> tuple[float | None, float | None]:
    """Validate optional source-clip boundaries and return rounded seconds."""
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise HTTPException(422, "Give both the start and the end of the part to dub")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= 0:
        raise HTTPException(422, "Clip times must be finite, non-negative seconds")
    if end - start < 1:
        raise HTTPException(422, "The part to dub must be at least one second long")
    if end - start > settings.max_video_minutes * 60:
        raise HTTPException(
            422, f"The part to dub can be at most {settings.max_video_minutes} minutes"
        )
    return round(start, 2), round(end, 2)


def check_ready(lip_sync_model: str | None = None):
    """Reject creation when required provider keys or the chosen media tools are missing."""
    if not all((settings.bodhan_asr_key, settings.bodhan_translate_key, settings.bodhan_tts_key)):
        raise HTTPException(400, "Add the missing Bodhan model keys to .env before creating a dub")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(400, "FFmpeg is required to process video")
    wanted = lip_sync_model or settings.lip_sync_model
    if wanted == "none":
        return
    if not any(model["id"] == wanted and model["available"] for model in available_models()):
        raise HTTPException(
            400,
            "The selected lip-sync model is unavailable. Choose Voice only or an installed model.",
        )


def validate_options(language: str, source_language: str | None, voice: str | None):
    """Reject unsupported target/source languages and a voice outside the target's pool."""
    if language not in LANGUAGES:
        raise HTTPException(422, "Unsupported target language")
    if source_language and source_language not in LANGUAGES:
        raise HTTPException(422, "Unsupported source language")
    if voice and voice not in voice_pool(language):
        raise HTTPException(422, "Unsupported voice")


def validate_choices(
    lip_sync_model: str | None = None,
    translation_script: str | None = None,
    delivery_style: str | None = None,
    tts_engine: str | None = None,
    language: str | None = None,
):
    """Reject unknown render options; cloned voices also need a language that supports them."""
    if lip_sync_model and lip_sync_model not in LIP_SYNC_MODELS:
        raise HTTPException(422, "Unknown lip-sync model")
    if tts_engine and tts_engine not in {"auto", "clone"}:
        raise HTTPException(422, "tts_engine must be auto or clone")
    if tts_engine == "clone" and language and not clone_available(language):
        raise HTTPException(422, "Cloned voices are not available for this language")
    if translation_script and translation_script not in TRANSLATION_SCRIPTS:
        raise HTTPException(422, "translation_script must be native or codemix")
    if delivery_style and delivery_style not in DELIVERY_STYLES:
        raise HTTPException(422, "Unknown delivery style")


def resolve_background(mode: str | None, keep_background: bool | None) -> str:
    """Pick the background mode: an explicit mode, else the legacy keep flag, else the default."""
    if mode:
        if mode not in BACKGROUND_MODES:
            raise HTTPException(422, "background_mode must be separate, duck or none")
        return mode
    if keep_background is False:
        return "none"
    return settings.background_mode


def job_directory(job_id: UUID) -> Path:
    """The folder holding a job's media, caches and status file."""
    return settings.jobs_dir / str(job_id)


def with_line_states(data: dict) -> dict:
    """Jobs rendered before lines carried a state get one derived from what they have, so every
    job reads the same."""
    lines = data.get("segments") or []
    if not lines or all(line.get("state") for line in lines):
        return data
    synced = {
        round(float(line["start"]), 1)
        for line in (data.get("lip_sync") or {}).get("lines", [])
        if line.get("lip_synced")
    }
    done = data.get("status") == "completed"
    for line in lines:
        if line.get("state"):
            continue
        if done and round(float(line["start"]), 1) in synced:
            line["state"] = "lip-synced"
        elif line.get("voiced_text") or (done and line.get("translation")):
            line["state"] = "voiced"
        elif line.get("translation"):
            line["state"] = "translated"
        elif line.get("text"):
            line["state"] = "heard"
    return data


def existing_job(job_id: UUID) -> dict:
    """Load a job's record (with line states filled in) or answer 404."""
    data = store.get(str(job_id))
    if not data:
        raise HTTPException(404, "Job not found")
    data = with_line_states(data)
    return data


def start_job(job_id: str, resume: bool = False):
    """Persist the queued state and submit one job to the studio executor."""
    store.save(
        job_id, status="queued", stage="Queued", stage_key=None, error=None, queued_at=utcnow()
    )
    executor.submit(run_job, job_id, resume)
