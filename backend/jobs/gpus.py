"""Render slots with a GPU each. A slot is acquired like a semaphore; the worker launched by the thread that
holds it sees only that GPU (CUDA_VISIBLE_DEVICES), so two renders never share a card when two are given.
"""

from __future__ import annotations

import os
import threading
from threading import BoundedSemaphore, Lock

from backend.config import settings


class GpuPool:
    def __init__(self, gpus: list[str]):
        self.gpus = list(gpus)
        self.free = list(gpus)
        self.semaphore = BoundedSemaphore(len(gpus))
        self.lock = Lock()
        self.local = threading.local()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        ok = (
            self.semaphore.acquire(blocking, -1 if timeout is None else timeout)
            if blocking
            else self.semaphore.acquire(False)
        )
        if ok:
            with self.lock:
                self.local.gpu = self.free.pop()
        return ok

    def release(self):
        gpu = getattr(self.local, 'gpu', None)
        if gpu is not None:
            with self.lock:
                self.free.append(gpu)
            self.local.gpu = None
        self.semaphore.release()

    def env(self, base: dict | None = None) -> dict:
        """Environment for a worker launched by the current thread: its slot's GPU, or the process default."""
        merged = dict(base if base is not None else os.environ)
        gpu = getattr(self.local, 'gpu', None)
        if gpu is not None:
            merged['CUDA_VISIBLE_DEVICES'] = gpu
        return merged


def _slots() -> list[str]:
    listed = [
        g.strip()
        for g in (os.getenv('GPUS') or os.getenv('CUDA_VISIBLE_DEVICES') or '0').split(',')
        if g.strip()
    ]
    return [listed[i % len(listed)] for i in range(settings.render_workers)]


pool = GpuPool(_slots())
