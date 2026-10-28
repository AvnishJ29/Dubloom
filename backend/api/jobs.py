"""Job lifecycle routes: create (from a link, an upload or an earlier dub), list, watch, resume,
cancel, rename and delete."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.api.helpers import (
    check_ready,
    executor,
    existing_job,
    job_directory,
    resolve_background,
    start_job,
    validate_choices,
    validate_clip,
    validate_options,
    validate_voices,
    with_line_states,
)
from backend.api.schemas import CreateJob, Rename
from backend.config import settings
from backend.jobs.pipeline import cancel_job, resume_lip_sync, store
from backend.media.video_input import MEDIA_EXTENSIONS, SourceError, validate_url
from backend.models import ACTIVE_STATUSES, TERMINAL_STATUSES

router = APIRouter()


@router.post("/api/jobs", status_code=202)
def create_job(request: CreateJob):
    """Validate render options and enqueue a dub from a URL or saved source."""
    validate_options(request.language, request.source_language, request.voice)
    validate_choices(
        request.lip_sync_model,
        request.translation_script,
        request.delivery_style,
        request.tts_engine,
        request.language,
    )
    check_ready(request.lip_sync_model)
    if request.source_job:
        return dub_again(request)
    if not request.url:
        raise HTTPException(422, "Paste a video link")
    try:
        url = validate_url(request.url)
    except SourceError as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = str(uuid4())
    clip_start, clip_end = validate_clip(request.clip_start, request.clip_end)
    chosen = validate_voices(request.voices, request.language)
    background_mode = resolve_background(request.background_mode, request.keep_background)
    store.save(
        job_id,
        source_type="url",
        source_url=url,
        title=(request.title or "").strip() or None,
        clip_start=clip_start,
        clip_end=clip_end,
        voices=chosen,
        language=request.language,
        source_language=request.source_language,
        voice=request.voice,
        num_speakers=request.num_speakers,
        keep_background=background_mode != "none",
        background_mode=background_mode,
        enhance_mouth=(
            settings.mouth_enhance if request.enhance_mouth is None else request.enhance_mouth
        ),
        lip_sync_model=request.lip_sync_model or settings.lip_sync_model,
        translation_script=request.translation_script or settings.translation_script,
        delivery_style=request.delivery_style or "neutral",
        tts_engine=request.tts_engine or "auto",
        pipeline_mode="staged" if request.pipeline_mode == "staged" else "streaming",
    )
    start_job(job_id)
    return {"id": job_id}


def dub_again(request: CreateJob) -> dict:
    """A new dub of an earlier dub's video: its prepared source is linked in as the upload, so
    there is no download (no YouTube bot check, no waiting) and the part it was cut to stays the
    same."""
    earlier = existing_job(request.source_job)
    saved = job_directory(request.source_job) / "source.mp4"
    if not saved.is_file():
        raise HTTPException(
            409, "That dub's saved video is gone; paste the link or upload the file again"
        )
    job_id = str(uuid4())
    directory = job_directory(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.link(saved, directory / "upload.mp4")  # same disk: instant, no copy
    except OSError:
        shutil.copyfile(saved, directory / "upload.mp4")
    background_mode = resolve_background(request.background_mode, request.keep_background)
    store.save(
        job_id,
        source_type="upload",
        source_name=earlier.get("source_name") or f"{earlier.get('title') or 'video'}.mp4",
        source_url=earlier.get("source_url"),
        title=(request.title or "").strip() or earlier.get("title"),
        clip_start=None,
        clip_end=None,
        voices=validate_voices(request.voices, request.language),
        language=request.language,
        source_language=request.source_language,
        voice=request.voice,
        num_speakers=request.num_speakers,
        keep_background=background_mode != "none",
        background_mode=background_mode,
        enhance_mouth=(
            settings.mouth_enhance if request.enhance_mouth is None else request.enhance_mouth
        ),
        lip_sync_model=request.lip_sync_model or settings.lip_sync_model,
        translation_script=request.translation_script or settings.translation_script,
        delivery_style=request.delivery_style or "neutral",
        tts_engine=request.tts_engine or "auto",
        pipeline_mode="staged" if request.pipeline_mode == "staged" else "streaming",
        reused_from=str(request.source_job),
    )
    start_job(job_id)
    return {"id": job_id}


@router.post("/api/jobs/upload", status_code=202)
async def create_upload_job(
    file: UploadFile = File(...),
    language: str = Form(...),
    source_language: str | None = Form(default=None),
    voice: str | None = Form(default=None),
    num_speakers: int | None = Form(default=None),
    keep_background: bool | None = Form(default=None),
    title: str | None = Form(default=None),
    lip_sync_model: str | None = Form(default=None),
    background_mode: str | None = Form(default=None),
    enhance_mouth: bool | None = Form(default=None),
    translation_script: str | None = Form(default=None),
    delivery_style: str | None = Form(default=None),
    pipeline_mode: str | None = Form(default=None),
    tts_engine: str | None = Form(default=None),
    clip_start: float | None = Form(default=None),
    clip_end: float | None = Form(default=None),
    voices: str | None = Form(default=None),
):
    """Validate and save a bounded video upload before scheduling its dub."""
    clip_start, clip_end = validate_clip(clip_start, clip_end)
    chosen = validate_voices((voices or "").split(","), language)
    source_language = source_language or None
    voice = voice or None
    validate_options(language, source_language, voice)
    validate_choices(
        lip_sync_model or None,
        translation_script or None,
        delivery_style or None,
        tts_engine or None,
        language,
    )
    check_ready(lip_sync_model)
    if num_speakers is not None and not 1 <= num_speakers <= 20:
        raise HTTPException(422, "Number of speakers must be between 1 and 20")
    mode = resolve_background(background_mode or None, keep_background)
    if title is not None and len(title) > 140:
        raise HTTPException(422, "The title must be at most 140 characters")
    name = Path(file.filename or "upload.mp4").name
    extension = Path(name).suffix.lower()
    if extension not in MEDIA_EXTENSIONS:
        raise HTTPException(
            422,
            f"Unsupported file type {extension or '(none)'}; upload a video such as MP4, MOV, MKV or WebM",
        )
    job_id = str(uuid4())
    directory = settings.jobs_dir / job_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"upload{extension}"
    limit = settings.max_upload_mb * 1024 * 1024
    received = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > limit:
                    raise HTTPException(
                        413, f"The upload is larger than the {settings.max_upload_mb} MB limit"
                    )
                output.write(chunk)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await file.close()
    if received == 0:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(422, "The uploaded file is empty")
    store.save(
        job_id,
        source_type="upload",
        clip_start=clip_start,
        clip_end=clip_end,
        voices=chosen,
        source_url=None,
        source_name=name,
        title=(title or "").strip() or Path(name).stem[:140],
        language=language,
        source_language=source_language,
        voice=voice,
        num_speakers=num_speakers,
        keep_background=mode != "none",
        background_mode=mode,
        enhance_mouth=settings.mouth_enhance if enhance_mouth is None else enhance_mouth,
        upload_bytes=received,
        lip_sync_model=lip_sync_model or settings.lip_sync_model,
        translation_script=translation_script or settings.translation_script,
        delivery_style=delivery_style or "neutral",
        tts_engine=tts_engine or "auto",
        pipeline_mode="staged" if pipeline_mode == "staged" else "streaming",
    )
    start_job(job_id)
    return {"id": job_id}


def queue_positions(jobs: list[dict]) -> dict[str, int]:
    """Return queue positions by enqueue time, falling back to creation time."""
    queued = sorted(
        (job for job in jobs if job.get("status") == "queued"),
        key=lambda j: j.get("queued_at") or j.get("created_at", ""),
    )
    return {job["id"]: index + 1 for index, job in enumerate(queued)}


@router.get("/api/jobs")
def list_jobs():
    """Every job's record, with its place in the queue while it waits."""
    jobs = store.list()
    positions = queue_positions(jobs)
    for job in jobs:
        if job["id"] in positions:
            job["queue_position"] = positions[job["id"]]
    return jobs


@router.get("/api/jobs/{job_id}")
def get_job(job_id: UUID):
    """One job's full record."""
    return existing_job(job_id)


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: UUID, request: Request):
    """Server-sent events: the job document whenever it changes, then close when terminal."""
    identifier = str(job_id)
    if not store.get(identifier):
        raise HTTPException(404, "Job not found")

    async def stream():
        last_version = None
        idle = 0
        # fills a proxy's first buffer so the events after it are flushed at once
        yield ":" + " " * 4096 + "\n\n"
        while True:
            if await request.is_disconnected():
                return
            version = store.version(identifier)
            data = store.get(identifier)
            if data is None:
                yield "event: gone\ndata: {}\n\n"
                return
            if version != last_version:
                last_version = version
                idle = 0
                yield f"data: {json.dumps(with_line_states(data), ensure_ascii=False)}\n\n"
                if data.get("status") in TERMINAL_STATUSES:
                    yield "event: done\ndata: {}\n\n"
                    return
            else:
                idle += 1
                if idle % 20 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.75)

    # no-transform: a proxy in front (Cloudflare) must not compress and buffer the stream
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.post("/api/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: UUID):
    """Resume an interrupted job, using saved render inputs when available."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] not in {"failed", "cancelled"}:
        raise HTTPException(409, "Only a stopped job can be resumed")
    directory = job_directory(job_id)
    if not (directory / "source.mp4").is_file() and not (
        job.get("source_type") == "url" or list(directory.glob("upload.*"))
    ):
        raise HTTPException(409, "The saved source video is missing; create the project again")
    if (
        (directory / "dubbed_audio.wav").is_file()
        and job.get("stage_key") in {"lipsync", None}
        and job.get("segments")
    ):
        store.save(
            identifier,
            status="queued",
            error=None,
            stage="Queued to resume lip sync",
            stage_key=None,
        )
        executor.submit(resume_lip_sync, identifier)
    else:
        start_job(identifier, resume=True)
    return {"id": identifier}


@router.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: UUID):
    """Cancel an active job: a running one is told to stop; one with no live worker is marked
    cancelled at once."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] not in ACTIVE_STATUSES:
        raise HTTPException(409, "The job is not running")
    if not cancel_job(identifier):
        store.save(identifier, status="cancelled", stage="Cancelled", stage_key=None)
    else:
        store.save(identifier, stage="Cancelling…")
    return {"id": identifier}


@router.patch("/api/jobs/{job_id}")
def rename_job(job_id: UUID, request: Rename):
    """Rename a dub at any time; nothing else changes."""
    existing_job(job_id)
    store.save(str(job_id), title=request.title.strip())
    return {"id": str(job_id), "title": request.title.strip()}


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: UUID):
    """Delete a stopped job and all of its files."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Cancel the job before deleting it")
    try:
        store.delete(identifier)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"deleted": identifier}


@router.delete("/api/jobs")
def clear_jobs():
    """Delete every job and its files; refused while any job is active."""
    try:
        return {"deleted": store.clear()}
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
