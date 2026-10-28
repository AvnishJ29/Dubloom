"""A job's media for playback and download: the dub, its preview and review renders, the source,
poster, subtitles, and archived versions."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.api.helpers import existing_job, job_directory
from backend.media.subtitles import build_subtitles

router = APIRouter()


def media_file(job_id: UUID, name: str, media_type: str, filename: str, download: bool):
    """Serve one file from the job folder inline, or as an attachment when downloading."""
    path = job_directory(job_id) / name
    if not path.is_file():
        raise HTTPException(404, "Media not ready")
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        content_disposition_type="attachment" if download else "inline",
    )


def version_folder(job_id: UUID, number: int) -> Path:
    """The archive folder of one earlier render, or 404 when it does not exist."""
    folder = job_directory(job_id) / "versions" / str(number)
    if number < 1 or not (folder / "version.json").is_file():
        raise HTTPException(404, "No such version")
    return folder


@router.get("/api/jobs/{job_id}/video")
def get_video(job_id: UUID, download: bool = False):
    """The full-size dubbed video."""
    return media_file(job_id, "dubbed_video.mp4", "video/mp4", f"dubloom-{job_id}.mp4", download)


@router.get("/api/jobs/{job_id}/versions/{number}")
def get_version(job_id: UUID, number: int):
    """The archived render's record, including the script as it was voiced."""
    return JSONResponse(
        json.loads((version_folder(job_id, number) / "version.json").read_text(encoding="utf-8"))
    )


@router.get("/api/jobs/{job_id}/versions/{number}/{kind}")
def get_version_file(job_id: UUID, number: int, kind: str, download: bool = False):
    """An archived render's video, preview or audio."""
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


@router.get("/api/jobs/{job_id}/preview")
def get_preview(job_id: UUID):
    """The lighter rendition the page plays; the full-size master when none was needed (or for
    older jobs)."""
    name = (
        "preview_video.mp4"
        if (job_directory(job_id) / "preview_video.mp4").is_file()
        else "dubbed_video.mp4"
    )
    return media_file(job_id, name, "video/mp4", f"dubloom-{job_id}.mp4", False)


@router.get("/api/jobs/{job_id}/review")
def get_review(job_id: UUID):
    """While a render runs: the dub so far (the new voice over the original picture, with the
    lip-synced spans done so far)."""
    return media_file(
        job_id, "review_video.mp4", "video/mp4", f"dubloom-{job_id}-review.mp4", False
    )


@router.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: UUID, download: bool = False):
    """The dubbed audio track as WAV."""
    return media_file(job_id, "dubbed_audio.wav", "audio/wav", f"dubloom-{job_id}.wav", download)


@router.get("/api/jobs/{job_id}/source")
def get_source(job_id: UUID, download: bool = False):
    """The prepared source video the dub was made from."""
    return media_file(job_id, "source.mp4", "video/mp4", f"source-{job_id}.mp4", download)


@router.get("/api/jobs/{job_id}/poster")
def get_poster(job_id: UUID):
    """The job's thumbnail frame, cached by the browser for an hour."""
    path = job_directory(job_id) / "poster.jpg"
    if not path.is_file():
        raise HTTPException(404, "No poster yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@router.get("/api/jobs/{job_id}/subtitles")
def get_subtitles(
    job_id: UUID,
    track: str = "target",
    fmt: str = "srt",
    speakers: bool = True,
    download: bool = False,
):
    """Subtitles built from the current script: the dub (`target`) or the original (`source`)
    track, as SRT or WebVTT."""
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
