"""HTTP API for creating, watching and retrieving dubbing jobs (Bodhan Indic models only)."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.providers.bodhan_api import provider_status
from backend.config import ROOT, settings
from backend.providers import openrouter
from backend.speech.script_supervisor import rewrite_lines
from backend.jobs.versions import archive, describe_revoice, remember
from backend.media.shorts import make_short
from backend.jobs.takes import choose_take, generate_take
from backend.rendering.lip_models import LIP_SYNC_MODELS, available_models
from backend.models import ACTIVE_STATUSES, LANGUAGES, TERMINAL_STATUSES, VOICES
from backend.jobs.pipeline import cancel_job, is_running, resume_lip_sync, run_job, store, utcnow
from backend.media.separation import separation_available
from backend.speech.styles import DELIVERY_STYLES, TONE_TO_STYLE, style_options
from backend.media.subtitles import build_subtitles
from backend.media.video_input import MEDIA_EXTENSIONS, SourceError, validate_url
from backend.voices.assignment import voice_details, voice_pool
from backend.voices.engine import clone_available

logger = logging.getLogger(__name__)
app = FastAPI(title="Dubloom API", version="2.0.0")
frontend_dist = ROOT / "frontend" / "dist"
if (frontend_dist / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)
# Model stages of several jobs may overlap; CPU-heavy stages are bounded inside the pipeline.
executor = ThreadPoolExecutor(max_workers=settings.job_workers, thread_name_prefix="job")
SESSION_COOKIE = "dubloom_session"
BACKGROUND_MODES = {"separate", "duck", "none"}
TRANSLATION_SCRIPTS = {"native", "codemix"}


def session_token() -> str:
    """Derive the studio session token from the configured password."""
    secret = hashlib.sha256(("dubloom:" + settings.studio_password).encode()).hexdigest()
    return hmac.new(secret.encode(), b"session", hashlib.sha256).hexdigest()


@app.middleware("http")
async def require_password(request: Request, call_next):
    """Require a valid session for protected API routes when sign-in is enabled."""
    path = request.url.path
    password = getattr(settings, "studio_password", "")
    if password and path.startswith("/api/") and path not in {"/api/health", "/api/login"}:
        cookie = request.cookies.get(SESSION_COOKIE, "")
        header = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not (
            hmac.compare_digest(cookie, session_token())
            or hmac.compare_digest(header, session_token())
        ):
            return JSONResponse({"detail": "Sign in to use the studio"}, status_code=401)
    return await call_next(request)


@app.on_event("startup")
def recover_interrupted_jobs():
    """Mark orphaned active jobs as interrupted so the owner can resume them."""
    for job in store.list():
        if job.get("status") in ACTIVE_STATUSES and not is_running(job["id"]):
            store.save(
                job["id"],
                status="failed",
                stage="Interrupted",
                stage_key=None,
                retry_at=None,
                error="The studio stopped while this job was running. Resume it to continue.",
            )


class Login(BaseModel):
    password: str


class CreateJob(BaseModel):
    url: str | None = Field(default=None, min_length=8, max_length=2048)
    source_job: UUID | None = (
        None  # dub again: that dub's saved video (already cut to its part) is the source
    )
    language: str = Field(min_length=2)
    source_language: str | None = None
    voice: str | None = None
    num_speakers: int | None = Field(default=None, ge=1, le=20)
    keep_background: bool | None = None
    title: str | None = Field(default=None, max_length=140)
    lip_sync_model: str | None = None
    background_mode: str | None = None
    enhance_mouth: bool | None = None
    translation_script: str | None = None
    delivery_style: str | None = None
    tts_engine: str | None = None  # auto (studio voices) or clone (original speakers' voices)
    voices: list[str] | None = (
        None  # your voices in speaking order (advanced casting); blank = automatic matching
    )
    pipeline_mode: str | None = (
        None  # staged (default) or streaming: each line translated and shown as soon as it is heard
    )
    clip_start: float | None = Field(
        default=None, ge=0
    )  # dub only this part of the video (seconds)
    clip_end: float | None = Field(default=None, gt=0)


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


class RewriteRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=600)
    lines: list[int] | None = None  # indexes that may change; blank = the whole script
    field: str = (
        "translation"  # "translation" (the dub's lines) or "text" (the original transcript)
    )


class ScriptLine(BaseModel):
    index: int = Field(ge=0)
    translation: str | None = Field(default=None, max_length=2000)  # a new target-language line
    text: str | None = Field(
        default=None, max_length=2000
    )  # or a corrected original line: re-translated, then re-voiced
    via: str | None = Field(
        default=None, max_length=10
    )  # 'ai' when the line came from an Ask-AI proposal


class ScriptChange(BaseModel):
    lines: list[ScriptLine] = Field(min_length=1)
    tone: str | None = Field(default=None, max_length=40)


class Revoice(BaseModel):
    speaker_voices: dict[str, str] = Field(default_factory=dict)
    speaker_styles: dict[str, str] = Field(default_factory=dict)
    delivery_style: str | None = None
    keep_background: bool | None = None
    lip_sync_model: str | None = None
    background_mode: str | None = None
    enhance_mouth: bool | None = None
    tts_engine: str | None = None


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
    if mode:
        if mode not in BACKGROUND_MODES:
            raise HTTPException(422, "background_mode must be separate, duck or none")
        return mode
    if keep_background is False:
        return "none"
    return settings.background_mode


def job_directory(job_id: UUID) -> Path:
    return settings.jobs_dir / str(job_id)


def with_line_states(data: dict) -> dict:
    """Jobs rendered before lines carried a state get one derived from what they have, so every job reads the same."""
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
    data = store.get(str(job_id))
    if not data:
        raise HTTPException(404, "Job not found")
    data = with_line_states(data)
    return data


def queue_positions(jobs: list[dict]) -> dict[str, int]:
    """Return queue positions by enqueue time, falling back to creation time."""
    queued = sorted(
        (job for job in jobs if job.get("status") == "queued"),
        key=lambda j: j.get("queued_at") or j.get("created_at", ""),
    )
    return {job["id"]: index + 1 for index, job in enumerate(queued)}


def start_job(job_id: str, resume: bool = False):
    """Persist the queued state and submit one job to the studio executor."""
    store.save(
        job_id, status="queued", stage="Queued", stage_key=None, error=None, queued_at=utcnow()
    )
    executor.submit(run_job, job_id, resume)


@app.post("/api/login")
def login(request: Login):
    if not settings.studio_password:
        return {"ok": True, "required": False}
    if not hmac.compare_digest(request.password, settings.studio_password):
        raise HTTPException(401, "Incorrect password")
    response = JSONResponse({"ok": True, "required": True})
    response.set_cookie(
        SESSION_COOKIE, session_token(), max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax"
    )
    return response


@app.get("/api/config")
def configuration():
    """Expose supported languages, voices, modes, and limits to the studio UI."""
    return {
        "languages": LANGUAGES,
        "voices": {code: voice_details(code) for code in LANGUAGES},
        "native_voices": VOICES,
        "diarization_backend": settings.diarization_backend,
        "lip_sync_runtime": settings.lip_sync_runtime,
        "max_video_minutes": settings.max_video_minutes,
        "max_upload_mb": settings.max_upload_mb,
        "max_video_height": settings.max_video_height,
        "job_workers": settings.job_workers,
        "render_workers": settings.render_workers,
        "upload_extensions": sorted(MEDIA_EXTENSIONS),
        "password_required": bool(settings.studio_password),
        "lip_sync_models": available_models(),
        "lip_sync_model_default": settings.lip_sync_model,
        "background": {
            "default": settings.background_mode,
            "separation_available": separation_available(),
            "separation_max_minutes": settings.separation_max_minutes,
        },
        "mouth_enhance": {
            "default": settings.mouth_enhance,
            "available": settings.esrgan_weights.is_file(),
        },
        "styles": style_options(),
        "translation_scripts": [
            {"id": "native", "label": "Native script"},
            {"id": "codemix", "label": "Code-mixed (English words stay in English)"},
        ],
        "translation_script_default": settings.translation_script,
        "clone_languages": sorted(code for code in LANGUAGES if clone_available(code)),
    }


@app.get("/api/health")
def health():
    """Report local configuration, provider status, and active-job counts."""
    connections = {
        "asr": bool(settings.bodhan_asr_key),
        "translation": bool(settings.bodhan_translate_key),
        "tts": bool(settings.bodhan_tts_key),
    }
    sherpa = settings.jobs_dir / "models" / "sherpa"
    return {
        "ok": True,
        "bodhan_configured": all(connections.values()),
        "connections": connections,
        "wav2lip_checkpoint_found": settings.wav2lip_checkpoint.is_file()
        or settings.wav2lip_accurate_checkpoint.is_file(),
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
        "diarization_models_found": (
            sherpa / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        ).is_file(),
        "password_required": bool(settings.studio_password),
        "bodhan_status": dict(provider_status),
        "active_jobs": sum(1 for job in store.list() if job.get("status") in ACTIVE_STATUSES),
    }


@app.post("/api/jobs", status_code=202)
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
    """A new dub of an earlier dub's video: its prepared source is linked in as the upload, so there is no download (no
    YouTube bot check, no waiting) and the part it was cut to stays the same."""
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


@app.post("/api/jobs/upload", status_code=202)
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


@app.get("/api/jobs")
def list_jobs():
    jobs = store.list()
    positions = queue_positions(jobs)
    for job in jobs:
        if job["id"] in positions:
            job["queue_position"] = positions[job["id"]]
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: UUID):
    return existing_job(job_id)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: UUID, request: Request):
    """Server-sent events: the job document whenever it changes, then close when terminal."""
    identifier = str(job_id)
    if not store.get(identifier):
        raise HTTPException(404, "Job not found")

    async def stream():
        last_version = None
        idle = 0
        yield ":" + " " * 4096 + "\n\n"  # fills a proxy's first buffer so the events after it are flushed at once
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


@app.post("/api/jobs/{job_id}/resume", status_code=202)
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


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: UUID):
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] not in ACTIVE_STATUSES:
        raise HTTPException(409, "The job is not running")
    if not cancel_job(identifier):
        store.save(identifier, status="cancelled", stage="Cancelled", stage_key=None)
    else:
        store.save(identifier, stage="Cancelling…")
    return {"id": identifier}


@app.post("/api/jobs/{job_id}/revoice", status_code=202)
def revoice(job_id: UUID, request: Revoice):
    """Change voices, delivery styles or render options and rebuild the dub, reusing everything unchanged."""
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


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=140)


@app.patch("/api/jobs/{job_id}")
def rename_job(job_id: UUID, request: Rename):
    """Rename a dub at any time; nothing else changes."""
    existing_job(job_id)
    store.save(str(job_id), title=request.title.strip())
    return {"id": str(job_id), "title": request.title.strip()}


@app.post("/api/jobs/{job_id}/rewrite")
def propose_rewrite(job_id: UUID, request: RewriteRequest):
    """Ask the script editor model to change the translation as instructed. Returns proposals only; the user
    reviews them in the editor and applies the ones they keep through /script."""
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


class ShortRequest(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    layout: str = "face"  # face | centre | blur
    subtitles: bool = True


@app.post("/api/jobs/{job_id}/shorts")
def create_short(job_id: UUID, request: ShortRequest):
    """Cut a vertical short from the finished dub with the dubbed lines burned in."""
    job = existing_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Finish the dub first")
    if request.layout not in ("face", "centre", "blur"):
        raise HTTPException(422, "layout must be face, centre or blur")
    try:
        short = make_short(
            job,
            job_directory(job_id),
            request.start,
            request.end,
            request.layout,
            request.subtitles,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    shorts = list(job.get("shorts") or []) + [short]
    store.save(str(job_id), shorts=shorts)
    return short


@app.get("/api/jobs/{job_id}/shorts/{number}")
def get_short(job_id: UUID, number: int, download: bool = False):
    path = job_directory(job_id) / "shorts" / f"short_{number}.mp4"
    if not path.is_file():
        raise HTTPException(404, "No such short")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"dubloom-{job_id}-short{number}.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


@app.delete("/api/jobs/{job_id}/shorts/{number}")
def delete_short(job_id: UUID, number: int):
    job = existing_job(job_id)
    folder = job_directory(job_id) / "shorts"
    for name in (f"short_{number}.mp4", f"short_{number}.srt"):
        (folder / name).unlink(missing_ok=True)
    store.save(str(job_id), shorts=[s for s in job.get("shorts") or [] if s["number"] != number])
    return {"deleted": number}


class Merge(BaseModel):
    into: str = Field(min_length=1)


@app.post("/api/jobs/{job_id}/speakers/{speaker}/merge", status_code=202)
def merge_speaker(job_id: UUID, speaker: str, request: Merge):
    """Fold one speaker into another (the same person split in two): one card, one voice; the moved lines are re-voiced."""
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
    if (
        turns_path.is_file()
    ):  # the cached speakers stage must agree, or a resume would recreate the split
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


class TakeRequest(BaseModel):
    style: str | None = None
    text: str | None = Field(default=None, max_length=2000)


@app.post("/api/jobs/{job_id}/lines/{index}/takes")
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


@app.post("/api/jobs/{job_id}/lines/{index}/take")
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


@app.get("/api/jobs/{job_id}/takes/{name}")
def get_take(job_id: UUID, name: str):
    if "/" in name or not name.startswith("line_") or not name.endswith(".wav"):
        raise HTTPException(404, "No such take")
    path = job_directory(job_id) / "takes" / name
    if not path.is_file():
        raise HTTPException(404, "No such take")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/jobs/{job_id}/script", status_code=202)
def change_script(job_id: UUID, request: ScriptChange):
    """Apply edited translations, then re-voice the changed lines and re-render."""
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Wait for the job to finish before editing the script")
    directory = job_directory(job_id)
    if not (directory / "source.mp4").is_file() or not (directory / "speaker_turns.json").is_file():
        raise HTTPException(409, "This project predates script editing; create a new dub instead")
    before = copy.deepcopy(
        job
    )  # the render being replaced, as voiced: the edits below change the lines in place
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
            # a corrected original: this line is translated and enhanced again from the new words, then re-voiced
            remember(segment, "text")
            remember(
                segment, "translation"
            )  # the dub line is translated anew; its words stay in the history
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


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: UUID):
    identifier = str(job_id)
    job = existing_job(job_id)
    if job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "Cancel the job before deleting it")
    try:
        store.delete(identifier)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"deleted": identifier}


@app.delete("/api/jobs")
def clear_jobs():
    try:
        return {"deleted": store.clear()}
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def media_file(job_id: UUID, name: str, media_type: str, filename: str, download: bool):
    path = job_directory(job_id) / name
    if not path.is_file():
        raise HTTPException(404, "Media not ready")
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: UUID, download: bool = False):
    return media_file(job_id, "dubbed_video.mp4", "video/mp4", f"dubloom-{job_id}.mp4", download)


def version_folder(job_id: UUID, number: int) -> Path:
    folder = job_directory(job_id) / "versions" / str(number)
    if number < 1 or not (folder / "version.json").is_file():
        raise HTTPException(404, "No such version")
    return folder


@app.get("/api/jobs/{job_id}/versions/{number}")
def get_version(job_id: UUID, number: int):
    """The archived render's record, including the script as it was voiced."""
    return JSONResponse(
        json.loads((version_folder(job_id, number) / "version.json").read_text(encoding="utf-8"))
    )


@app.get("/api/jobs/{job_id}/versions/{number}/{kind}")
def get_version_file(job_id: UUID, number: int, kind: str, download: bool = False):
    folder = version_folder(job_id, number)
    if kind == "video":
        name, media = "dubbed_video.mp4", "video/mp4"
    elif kind == "preview":
        name = (
            "preview_video.mp4" if (folder / "preview_video.mp4").is_file() else "dubbed_video.mp4"
        )
        media = "video/mp4"
    elif kind == "audio":
        name, media = "dubbed_audio.wav", "audio/wav"
    else:
        raise HTTPException(404, "Unknown file")
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, "That version has no such file")
    return FileResponse(
        path,
        media_type=media,
        filename=f"dubloom-{job_id}-v{number}{path.suffix}",
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/jobs/{job_id}/preview")
def get_preview(job_id: UUID):
    """The lighter rendition the page plays; the full-size master when none was needed (or for older jobs)."""
    name = (
        "preview_video.mp4"
        if (job_directory(job_id) / "preview_video.mp4").is_file()
        else "dubbed_video.mp4"
    )
    return media_file(job_id, name, "video/mp4", f"dubloom-{job_id}.mp4", False)


@app.get("/api/jobs/{job_id}/review")
def get_review(job_id: UUID):
    """While a render runs: the dub so far (the new voice over the original picture, with the lip-synced spans done so far)."""
    return media_file(
        job_id, "review_video.mp4", "video/mp4", f"dubloom-{job_id}-review.mp4", False
    )


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: UUID, download: bool = False):
    return media_file(job_id, "dubbed_audio.wav", "audio/wav", f"dubloom-{job_id}.wav", download)


@app.get("/api/jobs/{job_id}/source")
def get_source(job_id: UUID, download: bool = False):
    return media_file(job_id, "source.mp4", "video/mp4", f"source-{job_id}.mp4", download)


@app.get("/api/jobs/{job_id}/poster")
def get_poster(job_id: UUID):
    path = job_directory(job_id) / "poster.jpg"
    if not path.is_file():
        raise HTTPException(404, "No poster yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/jobs/{job_id}/subtitles")
def get_subtitles(
    job_id: UUID,
    track: str = "target",
    fmt: str = "srt",
    speakers: bool = True,
    download: bool = False,
):
    job = existing_job(job_id)
    if track not in {"target", "source"} or fmt not in {"srt", "vtt"}:
        raise HTTPException(422, "track must be target or source; fmt must be srt or vtt")
    text = build_subtitles(job.get("segments") or [], track, fmt, speakers)
    media = "text/vtt" if fmt == "vtt" else "application/x-subrip"
    label = job.get("language") if track == "target" else (job.get("source_language") or "source")
    headers = {
        "Content-Disposition": f'{"attachment" if download else "inline"}; filename="dubloom-{job_id}-{label}.{fmt}"'
    }
    return Response(text, media_type=f"{media}; charset=utf-8", headers=headers)


@app.get("/")
def studio():
    index = frontend_dist / "index.html"
    if not index.is_file():
        raise HTTPException(503, "Build the frontend with npm run build in frontend/")
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/{path:path}")
def spa_fallback(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "Not found")
    candidate = frontend_dist / path
    if path and candidate.is_file() and candidate.resolve().is_relative_to(frontend_dist.resolve()):
        return FileResponse(candidate)
    return studio()
