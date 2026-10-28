"""Shared HTTP transport for Bodhan model requests."""

import logging
import re
import threading
import time
from datetime import datetime, timezone

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)


class BodhanRateLimitError(RuntimeError):
    """The model remained rate-limited after retrying."""


class BodhanBudgetError(BodhanRateLimitError):
    """The account budget is exhausted; retrying will not help until credits are added."""


# Last provider-side problem per key, surfaced by /api/health so the UI can explain fallbacks.
provider_status: dict[str, dict] = {}
_status_lock = threading.Lock()


def _service_for(key: str) -> str:
    return {
        settings.bodhan_asr_key: "asr",
        settings.bodhan_translate_key: "translation",
        settings.bodhan_tts_key: "tts",
    }.get(key, "unknown")


def note_status(key: str, **fields):
    with _status_lock:
        provider_status[_service_for(key)] = {
            **fields,
            "at": datetime.now(timezone.utc).isoformat(),
        }


def clear_status(key: str):
    with _status_lock:
        provider_status.pop(_service_for(key), None)


def budget_summary(detail: str) -> str:
    """Turn the provider's budget error into a short, account-id-free sentence."""
    spend = re.search(r"Spend=\s*([^\d.]*)([\d.]+)", detail)
    budget = re.search(r"Budget=\s*([^\d.]*)([\d.]+)", detail)
    if spend and budget:
        try:
            return (
                f"Bodhan budget exhausted ({spend.group(1)}{float(spend.group(2)):.2f} spent of "
                f"{budget.group(1)}{float(budget.group(2)):.2f})"
            )
        except ValueError:
            pass
    return "Bodhan budget exhausted; add credits at console.bodhan.ai"


class KeyPacer:
    """Keep each Bodhan key under its published per-minute and parallel limits.

    Bodhan reports the limits on every response (x-ratelimit-api_key-limit-requests,
    x-ratelimit-api_key-limit-max_parallel_requests); waiting a moment locally is far
    cheaper than a 429 followed by a retry-after of up to half a minute.
    """

    def __init__(self, per_minute: int = 100, parallel: int = 8):
        self.per_minute = per_minute
        self.parallel = threading.BoundedSemaphore(parallel)
        self.parallel_limit = parallel
        self.sent: list[float] = []
        self.hold_until = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        self.parallel.acquire()
        while True:
            with self.lock:
                now = time.monotonic()
                self.sent = [t for t in self.sent if now - t < 60.0]
                if now >= self.hold_until and len(self.sent) < self.per_minute:
                    self.sent.append(now)
                    return
                wait = self.hold_until - now
                if len(self.sent) >= self.per_minute:
                    wait = max(wait, 60.0 - (now - self.sent[0]) + 0.05)
            time.sleep(max(0.05, wait))

    def release(self):
        self.parallel.release()

    def hold(self, seconds: float):
        """A 429 answers every worker on this key alike: nobody sends again until the provider's pause is over."""
        with self.lock:
            self.hold_until = max(self.hold_until, time.monotonic() + seconds)

    def learn(self, headers):
        try:
            limit = int(headers.get("x-ratelimit-api_key-limit-requests") or 0)
        except ValueError:
            limit = 0
        if limit > 0:
            with self.lock:
                self.per_minute = max(
                    1, int(limit * 0.9)
                )  # keep a margin for other clients of the key


_pacers: dict[str, KeyPacer] = {}
_pacers_lock = threading.Lock()


def pacer_for(key: str) -> KeyPacer:
    with _pacers_lock:
        if key not in _pacers:
            _pacers[key] = KeyPacer()
        return _pacers[key]


class BodhanAPI:
    """Authenticated Bodhan transport with per-key pacing, retries, and budget handling."""

    def __init__(self, on_rate_limit=None):
        self.client = httpx.Client(base_url=settings.bodhan_base_url, timeout=120)
        self.on_rate_limit = on_rate_limit
        self.exhausted: set[str] = set()
        self.lock = threading.Lock()

    def post(
        self, path: str, key: str, allow_404: bool = False, rate_limit_attempts: int = 8, **kwargs
    ) -> httpx.Response:
        with self.lock:
            if key in self.exhausted:
                raise BodhanBudgetError("Bodhan budget is exhausted for this model")
        # Multipart files must be rewound after a rate-limited request; otherwise
        # retrying ASR can accidentally submit an empty upload.
        uploads = []
        for value in kwargs.get('files', {}).values():
            stream = value[1] if isinstance(value, tuple) else value
            if hasattr(stream, 'seek') and hasattr(stream, 'tell'):
                uploads.append((stream, stream.tell()))
        for attempt in range(rate_limit_attempts):
            for stream, position in uploads:
                stream.seek(position)
            pacer = pacer_for(key)
            pacer.acquire()
            try:
                response = self.client.post(
                    path,
                    headers={"Authorization": f"Bearer {key}"},
                    **kwargs,
                )
            finally:
                pacer.release()
            pacer.learn(response.headers)
            if response.status_code == 429:
                detail = self._detail(response)
                if self._budget_exceeded(response):
                    summary = budget_summary(detail)
                    with self.lock:
                        self.exhausted.add(key)
                    note_status(key, budget_exceeded=True, message=summary)
                    raise BodhanBudgetError(summary)
                logger.info(
                    "Bodhan 429 on %s (attempt %d): %s | retry-after=%s remaining=%s/%s",
                    path,
                    attempt + 1,
                    detail[:240],
                    response.headers.get("retry-after"),
                    response.headers.get("x-ratelimit-api_key-remaining-requests"),
                    response.headers.get("x-ratelimit-api_key-remaining-max_parallel_requests"),
                )
                if attempt < rate_limit_attempts - 1:
                    # no retry-after means a model-level limit (litellm): it clears in tens of seconds, not two
                    try:
                        delay = max(
                            1, min(float(response.headers.get("retry-after", 5 * 2**attempt)), 60)
                        )
                    except ValueError:
                        delay = min(5 * 2**attempt, 60)
                    pacer.hold(delay)
                    if self.on_rate_limit:
                        self.on_rate_limit(delay)
                    time.sleep(delay)
                    continue
                note_status(key, rate_limited=True, message=detail[:200])
                raise BodhanRateLimitError("Bodhan model rate limit did not clear after retries")
            if response.status_code == 404 and allow_404:
                return response
            if response.is_error:
                raise RuntimeError(
                    f"Bodhan API ({response.status_code}): {self._detail(response)[:300]}"
                )
            clear_status(key)
            return response
        raise BodhanRateLimitError("Bodhan rate limit exceeded")

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text
        if not isinstance(payload, dict):
            return response.text
        detail = payload.get("error", payload.get("detail", response.text))
        if isinstance(detail, dict):
            detail = detail.get("message", "Model request failed")
        return str(detail)

    @staticmethod
    def _budget_exceeded(response: httpx.Response) -> bool:
        try:
            payload = response.json()
        except ValueError:
            return "budget" in response.text.lower()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return (
                error.get("type") == "budget_exceeded"
                or "budget" in str(error.get("message", "")).lower()
            )
        return "budget" in response.text.lower()

    def close(self):
        self.client.close()
