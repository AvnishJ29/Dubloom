"""Cancellation, concurrency limits and the thread pools that running jobs share."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore, Lock

from backend.config import settings
from backend.jobs.gpus import pool
from backend.jobs.store import store


class CancelledError(RuntimeError):
    """The job was cancelled by the user."""


class JobControl:
    """Cancellation token plus the child processes that must die with it."""

    def __init__(self):
        self.event = threading.Event()
        self.processes: set = set()
        self.lock = Lock()

    def cancel(self):
        """Flag the job as cancelled and stop its child processes."""
        self.event.set()
        self.terminate_processes()

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    def check(self):
        """Raise CancelledError once the job is cancelled; stages call this between steps."""
        if self.event.is_set():
            raise CancelledError("Cancelled")

    def terminate_processes(self):
        """Stop child processes (for example a separation still running after a failure)."""
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass

    def register(self, process):
        """Track a worker process so cancelling the job stops it."""
        with self.lock:
            self.processes.add(process)
        if self.event.is_set():
            process.terminate()

    def unregister(self, process):
        """Stop tracking a worker process that has exited."""
        with self.lock:
            self.processes.discard(process)


_controls: dict[str, JobControl] = {}
_controls_lock = Lock()
heavy_slots = pool  # one GPU per render slot (backend/jobs/gpus.py)
# Separation is 20 s of light GPU work: it must not queue behind long renders.
separation_slots = BoundedSemaphore(1)
# Vocal separation and face analysis overlap with the network-bound model stages of the same job.
background_executor = ThreadPoolExecutor(
    max_workers=max(1, settings.job_workers), thread_name_prefix="separation"
)


def control_for(job_id: str) -> JobControl:
    """The job's live control, created fresh when it has none or the old one was cancelled."""
    with _controls_lock:
        control = _controls.get(job_id)
        if control is None or control.cancelled:
            control = _controls[job_id] = JobControl()
        return control


def release_control(job_id: str):
    """Forget the job's control once its run has ended."""
    with _controls_lock:
        _controls.pop(job_id, None)


def cancel_job(job_id: str) -> bool:
    """Cancel a running job; False when it is not running in this process."""
    with _controls_lock:
        control = _controls.get(job_id)
    if control is None:
        return False
    control.cancel()
    return True


def is_running(job_id: str) -> bool:
    """Whether this process is currently running the job."""
    with _controls_lock:
        return job_id in _controls


def run_segment_stage(segments, worker, max_workers, completed, control: JobControl | None = None):
    """Run independent requests concurrently, keeping stage barriers and order."""

    def guarded(index, segment):
        if control:
            control.check()
        return worker(index, segment)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(guarded, index, segment) for index, segment in enumerate(segments)
        ]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                completed(count)
        except Exception:
            for future in futures:
                future.cancel()
            raise


def heavy_stage(control: JobControl, job_id: str, waiting_stage: str):
    """Context manager holding one GPU slot for heavy work; the job shows `waiting_stage` while it queues."""

    class Slot:
        def __enter__(self):
            if not heavy_slots.acquire(blocking=False):
                store.save(job_id, stage=waiting_stage, waiting=True)
                while not heavy_slots.acquire(timeout=1):
                    control.check()
                store.save(job_id, waiting=False)
            return self

        def __exit__(self, *exc):
            heavy_slots.release()
            return False

    return Slot()
