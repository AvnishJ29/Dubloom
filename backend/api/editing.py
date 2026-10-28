"""Editing a finished dub: re-voice, AI script proposals, script edits, speaker merges and
per-line takes. Changes that re-render archive the current render as a version first."""

from __future__ import annotations

import copy
import json
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api.helpers import (
    existing_job,
    job_directory,
    resolve_background,
    start_job,
    validate_choices,
)
from backend.api.schemas import Merge, Revoice, RewriteRequest, ScriptChange, TakeRequest
from backend.config import settings
from backend.jobs.store import store
from backend.jobs.takes import choose_take, generate_take
from backend.jobs.versions import archive, describe_revoice, remember
from backend.models import ACTIVE_STATUSES
from backend.providers import openrouter
from backend.speech.script_supervisor import rewrite_lines
from backend.speech.styles import DELIVERY_STYLES, TONE_TO_STYLE
from backend.voices.assignment import voice_pool

router = APIRouter()


@router.post("/api/jobs/{job_id}/revoice", status_code=202)
def revoice(job_id: UUID, request: Revoice):
    """Change voices, delivery styles or render options and rebuild the dub, reusing everything
    unchanged."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Wait for the job to finish before changing voices")
    directory = job_directory(job_id)
    if not (directory / "source.mp4").is_file() or not (directory / "speaker_turns.json").is_file():
        raise HTTPException(409, "This project predates re-voicing; create a new dub instead")
    pool = voice_pool(job["language"])
    overrides = {}
    for speaker, voice in request.speaker_voices.items():
        if voice not in pool:
            raise HTTPException(422, f"Unsupported voice {voice}")
        overrides[speaker] = voice
    validate_choices(
        request.lip_sync_model, None, request.delivery_style, request.tts_engine, job["language"]
    )
    for style in request.speaker_styles.values():
        if style not in DELIVERY_STYLES:
            raise HTTPException(422, f"Unknown delivery style {style}")
    changes = {
        "voice_overrides": {**(job.get("voice_overrides") or {}), **overrides},
        "lip_sync": None,
        "video_url": None,
    }
    if request.speaker_styles:
        changes["speaker_styles"] = {**(job.get("speaker_styles") or {}), **request.speaker_styles}
    if request.delivery_style:
        changes["delivery_style"] = request.delivery_style
    if request.lip_sync_model:
        changes["lip_sync_model"] = request.lip_sync_model
    if request.background_mode or request.keep_background is not None:
        changes["background_mode"] = resolve_background(
            request.background_mode, request.keep_background
        )
        changes["keep_background"] = changes["background_mode"] != "none"
    if request.enhance_mouth is not None:
        changes["enhance_mouth"] = request.enhance_mouth
    if request.tts_engine:
        changes["tts_engine"] = request.tts_engine
    changes["versions"] = archive(job, directory, describe_revoice(job, request))
    changes["pending_takes"] = 0
    store.save(identifier, **changes)
    start_job(identifier, resume=True)
    return {"id": identifier}


@router.post("/api/jobs/{job_id}/rewrite")
def propose_rewrite(job_id: UUID, request: RewriteRequest):
    """Ask the script editor model to change the translation as instructed. Returns proposals
    only; the user reviews them in the editor and applies the ones they keep through /script."""
    job = existing_job(job_id)
    segments = job.get("segments") or []
    if request.field not in ("translation", "text"):
        raise HTTPException(422, "field must be translation or text")
    if not any(s.get(request.field) for s in segments):
        raise HTTPException(409, "There is no script to change yet")
    if not openrouter.configured():
        raise HTTPException(400, "Add an OpenRouter key to .env to let the AI change the script")
    only = set(request.lines) if request.lines else None
    if only and any(i < 0 or i >= len(segments) for i in only):
        raise HTTPException(422, "A requested line does not exist")
    try:
        proposed = rewrite_lines(
            segments,
            job["language"],
            job.get("source_language"),
            request.instruction,
            job.get("translation_script") or settings.translation_script,
            only=only,
            content=job.get("script_profile"),
            field=request.field,
        )
    except Exception as exc:  # the model is a best-effort helper; tell the user instead of a 500
        raise HTTPException(502, f"The script editor could not answer: {str(exc)[:200]}") from exc
    return {
        "instruction": request.instruction,
        "field": request.field,
        "lines": [
            {"index": i, "before": segments[i].get(request.field) or "", "after": text}
            for i, text in sorted(proposed.items())
        ],
    }


@router.post("/api/jobs/{job_id}/speakers/{speaker}/merge", status_code=202)
def merge_speaker(job_id: UUID, speaker: str, request: Merge):
    """Fold one speaker into another (the same person split in two): one card, one voice; the
    moved lines are re-voiced."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Wait for the job to finish before merging speakers")
    voices = dict(job.get("speaker_voices") or {})
    if speaker not in voices or request.into not in voices or speaker == request.into:
        raise HTTPException(422, "Choose two different speakers of this dub")
    segments = [dict(s) for s in job.get("segments") or []]
    moved = 0
    for segment in segments:
        if segment.get("speaker") == speaker:
            segment["speaker"] = request.into
            segment["voice"] = voices[request.into]
            moved += 1
    profiles = dict(job.get("speaker_profiles") or {})
    merged = profiles.pop(speaker, {}) or {}
    target = dict(profiles.get(request.into) or {})
    target["speech_seconds"] = round(
        (target.get("speech_seconds") or 0) + (merged.get("speech_seconds") or 0), 2
    )
    target["turns"] = (target.get("turns") or 0) + (merged.get("turns") or 0)
    profiles[request.into] = target
    voices.pop(speaker, None)
    overrides = {k: v for k, v in (job.get("voice_overrides") or {}).items() if k != speaker}
    styles = {k: v for k, v in (job.get("speaker_styles") or {}).items() if k != speaker}
    reasons = {k: v for k, v in (job.get("voice_reasons") or {}).items() if k != speaker}
    directory = job_directory(job_id)
    turns_path = directory / "speaker_turns.json"
    # the cached speakers stage must agree, or a resume would recreate the split
    if turns_path.is_file():
        turns = json.loads(turns_path.read_text(encoding="utf-8"))
        for item in turns.get("segments") or []:
            if item.get("speaker") == speaker:
                item["speaker"] = request.into
        (turns.get("profiles") or {}).pop(speaker, None)
        (turns.get("split_speakers") or {}).pop(speaker, None)
        turns_path.write_text(json.dumps(turns), encoding="utf-8")
    store.save(
        identifier,
        segments=segments,
        speaker_profiles=profiles,
        speaker_voices=voices,
        voice_overrides=overrides,
        speaker_styles=styles,
        voice_reasons=reasons,
        speaker_count=len(voices),
        lip_sync=None,
        video_url=None,
        versions=archive(
            job,
            directory,
            f"{merged.get('name') or speaker} merged into {target.get('name') or request.into}",
        ),
    )
    start_job(identifier, resume=True)
    return {"id": identifier, "moved": moved}


@router.post("/api/jobs/{job_id}/lines/{index}/takes")
def add_take(job_id: UUID, index: int, request: TakeRequest):
    """Voice one alternate reading of a line (style and/or words) for the director panel."""
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Wait for the job to finish before directing lines")
    segments = job.get("segments") or []
    if index < 0 or index >= len(segments):
        raise HTTPException(404, "No such line")
    if request.style and request.style not in DELIVERY_STYLES:
        raise HTTPException(422, f"Unknown delivery style {request.style}")
    try:
        take = generate_take(job, job_directory(job_id), index, request.style, request.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001  (the voice service answered badly)
        raise HTTPException(502, f"The voice could not be generated: {str(exc)[:200]}") from exc
    takes = dict(job.get("takes") or {})
    takes[str(index)] = (takes.get(str(index)) or []) + [take]
    store.save(str(job_id), takes=takes)
    return {"index": index, "takes": takes[str(index)]}


@router.post("/api/jobs/{job_id}/lines/{index}/take")
def pick_take(job_id: UUID, index: int, payload: dict):
    """Use a take as the line's clip. Takes effect on the next re-render."""
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Wait for the job to finish before directing lines")
    segments = list(job.get("segments") or [])
    if index < 0 or index >= len(segments):
        raise HTTPException(404, "No such line")
    try:
        segments[index] = choose_take(
            job, job_directory(job_id), index, int(payload.get("number", 0))
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.save(str(job_id), segments=segments, pending_takes=(job.get("pending_takes") or 0) + 1)
    return {"index": index, "line": segments[index]}


@router.get("/api/jobs/{job_id}/takes/{name}")
def get_take(job_id: UUID, name: str):
    """Play one generated take; only `line_*.wav` names inside the job's takes folder."""
    if "/" in name or not name.startswith("line_") or not name.endswith(".wav"):
        raise HTTPException(404, "No such take")
    path = job_directory(job_id) / "takes" / name
    if not path.is_file():
        raise HTTPException(404, "No such take")
    return FileResponse(path, media_type="audio/wav")


@router.post("/api/jobs/{job_id}/script", status_code=202)
def change_script(job_id: UUID, request: ScriptChange):
    """Apply edited translations, then re-voice the changed lines and re-render."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Wait for the job to finish before editing the script")
    directory = job_directory(job_id)
    if not (directory / "source.mp4").is_file() or not (directory / "speaker_turns.json").is_file():
        raise HTTPException(409, "This project predates script editing; create a new dub instead")
    # the render being replaced, as voiced: the edits below change the lines in place
    before = copy.deepcopy(job)
    segments = job.get("segments") or []
    changed = 0
    for line in request.lines:
        if line.index >= len(segments):
            raise HTTPException(422, f"Line {line.index} does not exist")
        segment = segments[line.index]
        if (
            line.text is not None
            and line.text.strip()
            and line.text.strip() != (segment.get("text") or "")
        ):
            # a corrected original: this line is translated and enhanced again from the new
            # words, then re-voiced; the old dub line's words stay in the history
            remember(segment, "text")
            remember(segment, "translation")
            segment.update(
                text=line.text.strip(),
                source_edited=True,
                translation="",
                machine_translation="",
                voiced_text="",
                edited=False,
                ai_edited=False,
            )
            changed += 1
        elif line.translation is not None:
            text = line.translation.strip()
            if text and text != (segment.get("translation") or ""):
                remember(segment, "translation")
                segment["translation"] = text
                segment["ai_edited"] = line.via == "ai"
                segment["edited"] = True
                changed += 1
    if not changed:
        raise HTTPException(422, "No line was changed")
    changes = {
        "segments": segments,
        "lip_sync": None,
        "video_url": None,
        "script_edits": (job.get("script_edits") or 0) + changed,
        "versions": archive(
            before, directory, f"Script edited: {changed} line{'s' if changed != 1 else ''}"
        ),
    }
    if request.tone:
        changes["script_tone"] = request.tone if request.tone != "neutral" else None
        mapped = TONE_TO_STYLE.get(request.tone)
        if mapped and (job.get("delivery_style") or "neutral") == "neutral":
            changes["delivery_style"] = mapped
    store.save(identifier, **changes)
    start_job(identifier, resume=True)
    return {"id": identifier, "changed": changed}
