"""Job records: one `status.json` per job folder, merged and replaced atomically."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.config import settings
from backend.models import ACTIVE_STATUSES


def _plain(value):
    """NumPy scalars as Python values: a fresh dub's timings come from diarization as NumPy floats, so anything computed
    from them (a comparison gives numpy.bool) must still be storable. Anything else is a real bug and still raises.
    """
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def utcnow() -> str:
    """The current UTC time in ISO 8601, the format of every timestamp in a job record."""
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persist job records atomically and serialize mutations within this process."""

    def __init__(self):
        self.lock = Lock()
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        """Where the job's record lives."""
        return settings.jobs_dir / job_id / "status.json"

    def save(self, job_id: str, **changes):
        """Merge changes into a job record and replace its JSON file atomically."""
        path = self.path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            now = utcnow()
            data = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {
                    "id": job_id,
                    "status": "queued",
                    "progress": 0,
                    "stage": "Queued",
                    "stage_key": None,
                    "segments": [],
                    "created_at": now,
                }
            )
            data.update(changes)
            data["updated_at"] = now
            temp = path.with_suffix(".tmp")
            temp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=_plain), encoding="utf-8"
            )
            temp.replace(path)
        return data

    def get(self, job_id: str):
        """Read one job record, returning None for a missing or unreadable record."""
        path = self.path(job_id)
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def version(self, job_id: str) -> int:
        """A number that changes whenever the record is written (the live-update stream watches it)."""
        try:
            return self.path(job_id).stat().st_mtime_ns
        except OSError:
            return 0

    def list(self):
        """Every job, newest first, without the per-line script (the sidebar needs only the summary)."""
        with self.lock:
            jobs = []
            for directory in settings.jobs_dir.iterdir():
                path = directory / "status.json"
                if not directory.is_dir() or not path.is_file():
                    continue
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                job["segment_count"] = len(job.get("segments") or [])
                job.pop("segments", None)
                jobs.append(job)
            return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def delete(self, job_id: str) -> bool:
        """Remove a finished job's folder. Refuses running jobs and anything outside the jobs folder."""
        with self.lock:
            directory = (settings.jobs_dir / job_id).resolve()
            if (
                directory.parent != settings.jobs_dir.resolve()
                or not (directory / "status.json").is_file()
            ):
                return False
            try:
                if (
                    json.loads((directory / "status.json").read_text(encoding="utf-8")).get(
                        "status"
                    )
                    in ACTIVE_STATUSES
                ):
                    raise RuntimeError("Cancel the job before deleting it")
            except (OSError, json.JSONDecodeError):
                pass
            shutil.rmtree(directory)
            return True

    def clear(self):
        """Remove every job folder (and stray files) once no job is running; returns how many went."""
        with self.lock:
            root = settings.jobs_dir.resolve()
            direct_children = [path for path in root.iterdir() if path.resolve().parent == root]
            folders = [
                path
                for path in direct_children
                if path.is_dir() and (path / "status.json").is_file()
            ]
            files = [path for path in direct_children if path.is_file()]
            active = []
            for folder in folders:
                status_file = folder / "status.json"
                if status_file.is_file():
                    try:
                        if (
                            json.loads(status_file.read_text(encoding="utf-8")).get("status")
                            in ACTIVE_STATUSES
                        ):
                            active.append(folder.name)
                    except (OSError, json.JSONDecodeError):
                        continue
            if active:
                raise RuntimeError("Wait for active dubbing jobs to finish before clearing history")
            for path in folders:
                shutil.rmtree(path)
            for path in files:
                path.unlink()
            return len(folders) + len(files)


store = JobStore()
