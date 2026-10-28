"""Vertical shorts cut from a finished dub: create, play or download, delete."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api.helpers import existing_job, job_directory
from backend.api.schemas import ShortRequest
from backend.jobs.pipeline import store
from backend.media.shorts import make_short

router = APIRouter()


@router.post("/api/jobs/{job_id}/shorts")
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


@router.get("/api/jobs/{job_id}/shorts/{number}")
def get_short(job_id: UUID, number: int, download: bool = False):
    """Stream a short inline, or as an attachment with `download=true`."""
    path = job_directory(job_id) / "shorts" / f"short_{number}.mp4"
    if not path.is_file():
        raise HTTPException(404, "No such short")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"dubloom-{job_id}-short{number}.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


@router.delete("/api/jobs/{job_id}/shorts/{number}")
def delete_short(job_id: UUID, number: int):
    """Remove a short's video and subtitles and drop it from the job's list."""
    job = existing_job(job_id)
    folder = job_directory(job_id) / "shorts"
    for name in (f"short_{number}.mp4", f"short_{number}.srt"):
        (folder / name).unlink(missing_ok=True)
    store.save(str(job_id), shorts=[s for s in job.get("shorts") or [] if s["number"] != number])
    return {"deleted": number}
