"""Render slots with a GPU each. A slot is acquired like a semaphore; the worker launched by the thread that
holds it sees only that GPU (CUDA_VISIBLE_DEVICES), so two renders never share a card when two are given.
"""

from __future__ import annotations

import os
import threading
from threading import BoundedSemaphore, Lock

from backend.config import settings


class GpuPool:
    """Render slots, one GPU id each; a GPU appears once per slot that uses it."""

    def __init__(self, gpus: list[str]):
        self.gpus = list(gpus)
        self.free = list(gpus)
        self.semaphore = BoundedSemaphore(len(gpus))
        self.lock = Lock()
        self.local = threading.local()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Take a render slot and bind its GPU to the calling thread (see env()).

        Returns False when no slot is free and `blocking` is off or `timeout` seconds pass. Every
        successful call must be paired with release() on the same thread.
        """
        ok = self.semaphore.acquire(True, timeout) if blocking else self.semaphore.acquire(False)
        if ok:
            with self.lock:
                self.local.gpu = self.free.pop()
        return ok

    def release(self):
        """Return the calling thread's GPU to the pool and free its slot."""
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
