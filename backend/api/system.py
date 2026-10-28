"""Studio-wide endpoints: sign-in and its password gate, the UI's configuration, and health."""

from __future__ import annotations

import hashlib
import hmac
import shutil

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.api.schemas import Login
from backend.config import settings
from backend.jobs.store import store
from backend.media.separation import separation_available
from backend.media.video_input import MEDIA_EXTENSIONS
from backend.models import ACTIVE_STATUSES, LANGUAGES, VOICES
from backend.providers.bodhan_api import provider_status
from backend.rendering.lip_models import available_models
from backend.speech.styles import style_options
from backend.voices.assignment import voice_details
from backend.voices.engine import clone_available

router = APIRouter()
SESSION_COOKIE = "dubloom_session"


def session_token() -> str:
    """Derive the studio session token from the configured password."""
    secret = hashlib.sha256(("dubloom:" + settings.studio_password).encode()).hexdigest()
    return hmac.new(secret.encode(), b"session", hashlib.sha256).hexdigest()


async def require_password(request: Request, call_next):
    """HTTP middleware: require a valid session for protected API routes when sign-in is
    enabled."""
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


@router.post("/api/login")
def login(request: Login):
    """Check the studio password and set a 30-day session cookie."""
    if not settings.studio_password:
        return {"ok": True, "required": False}
    if not hmac.compare_digest(request.password, settings.studio_password):
        raise HTTPException(401, "Incorrect password")
    response = JSONResponse({"ok": True, "required": True})
    response.set_cookie(
        SESSION_COOKIE, session_token(), max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax"
    )
    return response


@router.get("/api/config")
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


@router.get("/api/health")
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
