"""The FastAPI application: middleware, startup recovery, the API routers, and the built
frontend. The SPA catch-all is registered last so it never shadows an API route."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import editing, jobs, media, shorts, system
from backend.config import ROOT
from backend.jobs.control import is_running
from backend.jobs.store import store
from backend.models import ACTIVE_STATUSES

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
app.middleware("http")(system.require_password)


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


for module in (system, jobs, editing, shorts, media):
    app.include_router(module.router)


@app.get("/")
def studio():
    """The built studio page."""
    index = frontend_dist / "index.html"
    if not index.is_file():
        raise HTTPException(503, "Build the frontend with npm run build in frontend/")
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/{path:path}")
def spa_fallback(path: str):
    """Serve a built frontend file, or the studio page for client-side routes; unknown API paths
    stay 404."""
    if path.startswith("api/"):
        raise HTTPException(404, "Not found")
    candidate = frontend_dist / path
    if path and candidate.is_file() and candidate.resolve().is_relative_to(frontend_dist.resolve()):
        return FileResponse(candidate)
    return studio()
