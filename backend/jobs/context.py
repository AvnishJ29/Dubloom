"""The state one pipeline attempt hands from stage to stage, and its progress reporting."""

from __future__ import annotations

import time
from pathlib import Path

from backend.config import settings
from backend.jobs.control import JobControl
from backend.jobs.store import store, utcnow
from backend.models import Segment
from backend.providers.usage import Usage
from backend.speech.styles import TONE_TO_STYLE, normalise_style

# Each stage owns a slice of the progress bar (percent).
STAGE_PROGRESS = {
    "source": (2, 12),
    "audio": (12, 15),
    "speakers": (15, 22),
    "transcribe": (22, 44),
    "translate": (44, 58),
    "voice": (58, 78),
    "mix": (78, 84),
    "lipsync": (84, 100),
}
BACKGROUND_MODES = {"separate", "duck", "none"}


def stage_progress(key: str, fraction: float) -> float:
    """Overall progress (percent) for a fraction (0-1) of one stage."""
    low, high = STAGE_PROGRESS[key]
    return low + (high - low) * max(0.0, min(1.0, fraction))


class DubRun:
    """One attempt at a job: its options, its progress reporting, and what each stage hands on.

    The stages in `backend/jobs/stages/` read and fill these attributes in pipeline order; see
    `process_job` in `backend/jobs/pipeline.py` for that order.
    """

    def __init__(self, job_id: str, resume: bool, control: JobControl):
        self.job_id = job_id
        self.resume = resume
        self.control = control
        self.directory: Path = settings.jobs_dir / job_id
        self.client = None  # BodhanAPI client, shared by the model stages
        self.workers = settings.model_request_workers
        self._stage_key: str | None = None
        self._stage_at = time.monotonic()
        self.stage_seconds: dict[str, float] = {}

        # Filled by start()
        self.job: dict = {}
        self.previous: dict | None = None  # the saved record when resuming: its lines are reused
        self.usage = Usage()
        self.language = ''
        self.source_language: str | None = None
        self.voice: str | None = None
        self.num_speakers: int | None = None
        self.background_mode = 'none'
        self.overrides: dict = {}
        self.script = ''
        self.delivery_style = 'neutral'
        self.speaker_styles: dict[str, str] = {}
        self.streaming = False

        # Filled by the stages
        self.video: Path | None = None  # source stage
        self.video_duration = 0.0
        self.source_audio: Path | None = None
        self.turns_path = self.directory / 'speaker_turns.json'  # speaker stage
        self.turns: dict = {}  # speaker_turns.json: segments, profiles, review, split speakers
        self.diarization_mode = ''
        self.segments: list[Segment] = []
        self.cached_segments: dict = {}  # earlier run's lines by (start, end, speaker)
        self.profiles: dict = {}
        self.speakers: list[str] = []
        self.content: dict = {}  # genre/tone profile from the script editor
        self.speaker_voices: dict[str, str] = {}
        self.voice_reasons: dict = {}
        self.background_notes: list[str] = []  # background work
        self.separation = None  # Future: the background vocal separation
        self.faces_ahead = None  # Future: the early face analysis
        self.spoken: list[Segment] = []  # script stages
        self.translated: list[Segment] = []
        self.timing_warnings: list = []  # mix stage

    def start(self):
        """Read the job's choices and mark the attempt as running."""
        job = self.job = store.get(self.job_id) or {}
        self.previous = job if self.resume else None
        self.usage = Usage(job.get('usage') if self.resume else None)
        self.language, self.source_language = job['language'], job.get('source_language')
        self.voice, self.num_speakers = job.get('voice'), job.get('num_speakers')
        keep_background = job.get('keep_background', settings.keep_background_default)
        self.background_mode = (
            job.get('background_mode') or (settings.background_mode if keep_background else 'none')
        ).lower()
        if self.background_mode not in BACKGROUND_MODES:
            self.background_mode = settings.background_mode
        self.overrides = job.get('voice_overrides') or {}
        self.script = job.get('translation_script') or settings.translation_script
        # Delivery style: project default plus per-speaker overrides (Bodhan Indic-Speak styles).
        self.delivery_style = normalise_style(
            job.get('delivery_style') or TONE_TO_STYLE.get(job.get('script_tone') or '', 'neutral')
        )
        self.speaker_styles = {
            speaker: normalise_style(style)
            for speaker, style in (job.get('speaker_styles') or {}).items()
        }
        # Streaming translates each line the moment it is heard.
        self.streaming = (job.get('pipeline_mode') or 'staged') == 'streaming'
        # stage_seconds is this run only: carrying earlier runs over made resumed jobs look slower than they were
        self.save(
            status="running",
            retryable=False,
            retry_at=None,
            error=None,
            started_at=utcnow(),  # this run: the UI's 'took' is the latest dub
            delivery_style=self.delivery_style,
            translation_script=self.script,
        )

    def save(self, **changes):
        """Merge changes into the job record."""
        return store.save(self.job_id, **changes)

    def mark(self, key: str | None, label: str, progress: float | None = None, **extra):
        """Report the current stage, timing each stage; also the cancellation checkpoint between steps."""
        self.control.check()
        now = time.monotonic()
        previous = self._stage_key
        if previous and previous != key:
            self.stage_seconds[previous] = round(
                self.stage_seconds.get(previous, 0) + now - self._stage_at, 1
            )
            extra['stage_seconds'] = self.stage_seconds
        if previous != key:
            self._stage_key, self._stage_at = key, now
        payload = {'stage': label, 'stage_key': key, **extra}
        if progress is not None:
            payload['progress'] = round(progress, 1)
        self.save(**payload)

    def span(self, key: str, fraction: float) -> float:
        """Overall progress for a fraction of one stage."""
        return stage_progress(key, fraction)

    def add_notes(self, *notes):
        """Append notes the owner should read (for example, why the background was ducked)."""
        notes = [note for note in notes if note]
        if notes:
            self.save(
                review_warnings=(store.get(self.job_id) or {}).get('review_warnings', []) + notes
            )

    def rate_limited(self, delay: float):
        """Show that Bodhan asked us to slow down, while the client waits."""
        self.save(stage=f'Bodhan model busy (rate limit); retrying in {delay:g} s')

    def close(self):
        """Release the model client."""
        if self.client:
            self.client.close()
