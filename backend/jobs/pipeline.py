"""Run the dubbing pipeline for one job and persist its progress.

`process_job` below is the whole pipeline at a glance; each step lives in `backend/jobs/stages/`.
Completed work is cached inside the job folder, so a stopped job resumes without repeating paid
model requests and a finished job is re-voiced cheaply after an edit.
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.config import settings
from backend.jobs.context import DubRun
from backend.jobs.control import (
    CancelledError,
    JobControl,
    control_for,
    heavy_stage,
    release_control,
)
from backend.jobs.stages.background import start_background_work
from backend.jobs.stages.mix import mix_dub
from backend.jobs.stages.render import render_dub, render_lips
from backend.jobs.stages.script import transcribe_script, translate_script, understand_content
from backend.jobs.stages.source import fetch_source
from backend.jobs.stages.speakers import cast_voices, find_speakers, match_faces_to_voices
from backend.jobs.stages.voicing import voice_script
from backend.jobs.store import store, utcnow
from backend.providers.bodhan_api import BodhanAPI, BodhanBudgetError, BodhanRateLimitError

logger = logging.getLogger(__name__)

# What a finished job folder keeps: playback files plus everything needed to resume or re-voice cheaply.
RETAINED_FILES = {
    "status.json",
    "dubbed_video.mp4",
    "preview_video.mp4",
    "dubbed_audio.wav",
    "dubbed_voice.wav",
    "lip_sync_report.json",
    "source.mp4",
    "source_audio.wav",
    "speaker_turns.json",
    "speaker_turns_raw.json",
    "poster.jpg",
    "background.wav",
    "vocals.wav",
    "separation.txt",
    "faces.npz",
    "faces.sha256",
    "render_source.mp4",
    "render_source.sha256",
    "render_source.mp4.digest",
    "source.mp4.digest",
    "targets.npz",
    "latentsync_cache",
    "asd.npz",
    "asd.sha256",
    "versions",
    "takes",
    "shorts",
    "gender.json",
    "gender.sha256",
    "hands.npz",
    "hands.sha256",
}
RETAINED_PREFIXES = (
    "tts_",
    "ref_",
    "lip_done_",
    "lip_report_",
)  # voices and rendered lip-sync chunks are reused on edits

RETRY_ATTEMPTS = 8  # attempts at a job that keeps hitting model rate limits
RETRY_DELAY = 180  # seconds between those attempts


def process_job(job_id: str, resume: bool = False, control: JobControl | None = None):
    """Run one pipeline attempt, reuse saved stages on resume, and persist progress."""
    dub = DubRun(job_id, resume, control or control_for(job_id))
    try:
        dub.start()
        fetch_source(dub)  # source.mp4 and its audio
        find_speakers(dub)  # who speaks when -> one script line per turn
        understand_content(dub)  # source language, genre and tone
        cast_voices(dub)  # a studio voice per speaker
        start_background_work(dub)  # vocal separation and face analysis, in parallel from here
        dub.client = BodhanAPI(on_rate_limit=dub.rate_limited)
        transcribe_script(dub)  # Bodhan ASR (+ translation as each line is heard, when streaming)
        translate_script(dub)  # Bodhan translation for the rest, then the script editor
        match_faces_to_voices(dub)  # faces correct the speakers' genders and false splits
        voice_script(dub)  # Bodhan TTS, fitted to each line's timing
        dubbed_audio = mix_dub(dub)  # voices placed over the background
        render_dub(dub, dubbed_audio)  # review cut, then lip sync and the final video
        dub.mark(
            None,
            'Ready',
            100,
            status='completed',
            finished_at=utcnow(),
            usage=dub.usage.as_dict(),
            review_url=None,
            timing_warnings=dub.timing_warnings,
            segments=[piece.public() for piece in dub.translated],
        )
        remove_intermediates(dub.directory)
    except CancelledError:
        logger.info('Job %s cancelled', job_id)
        store.save(
            job_id,
            status='cancelled',
            stage='Cancelled',
            stage_key=None,
            retryable=False,
            retry_at=None,
        )
    except Exception as exc:
        logger.exception('Dubbing job %s failed', job_id)
        dub.control.terminate_processes()
        store.save(
            job_id,
            status="failed",
            stage="Failed",
            stage_key=None,
            error=str(exc)[:500],
            retryable=isinstance(exc, BodhanRateLimitError)
            and not isinstance(exc, BodhanBudgetError),
        )
    finally:
        dub.close()


def run_job(job_id: str, resume: bool = False):
    """Wait out model quotas while reusing completed stages between attempts."""
    control = control_for(job_id)
    try:
        if (store.get(job_id) or {}).get('status') == 'cancelled':
            return  # cancelled while waiting its turn
        for cycle in range(RETRY_ATTEMPTS):
            process_job(job_id, resume or cycle > 0, control)
            job = store.get(job_id)
            if not job or not job.get('retryable') or job.get('status') != 'failed':
                return
            if cycle == RETRY_ATTEMPTS - 1:
                return
            retry_at = (datetime.now(timezone.utc) + timedelta(seconds=RETRY_DELAY)).isoformat()
            store.save(
                job_id,
                status='running',
                error=None,
                stage='Model rate limit; waiting to retry',
                retry_at=retry_at,
                retry_attempt=cycle + 1,
            )
            for _ in range(RETRY_DELAY):
                if control.cancelled:
                    store.save(
                        job_id,
                        status='cancelled',
                        stage='Cancelled',
                        retry_at=None,
                        retryable=False,
                    )
                    return
                time.sleep(1)
    finally:
        release_control(job_id)


def resume_lip_sync(job_id: str):
    """Resume rendering saved speech without repeating paid model requests."""
    control = control_for(job_id)
    directory = settings.jobs_dir / job_id
    try:
        store.save(
            job_id,
            status='running',
            error=None,
            progress=84,
            stage='Resuming lip sync',
            stage_key='lipsync',
        )
        with heavy_stage(control, job_id, 'Waiting for a render slot'):
            render_lips(job_id, directory, control)
        store.save(
            job_id,
            status='completed',
            progress=100,
            stage='Ready',
            stage_key=None,
            finished_at=utcnow(),
            review_url=None,
        )
        remove_intermediates(directory)
    except CancelledError:
        store.save(job_id, status='cancelled', stage='Cancelled', stage_key=None)
    except Exception as exc:
        logger.exception('Lip sync resume failed for job %s', job_id)
        store.save(job_id, status='failed', stage='Failed', stage_key=None, error=str(exc)[:500])
    finally:
        release_control(job_id)


def remove_intermediates(directory: Path):
    """Keep playback files plus everything needed to resume or re-voice cheaply."""
    for artifact in directory.iterdir():
        if artifact.name in RETAINED_FILES or artifact.name.startswith(RETAINED_PREFIXES):
            continue
        try:
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        except OSError:
            pass
