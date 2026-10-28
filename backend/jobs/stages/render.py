"""Render stage: a quick review cut, then lip sync (or voice-only) and the final video."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.config import settings
from backend.jobs.context import DubRun, stage_progress
from backend.jobs.control import CancelledError, JobControl, heavy_stage
from backend.jobs.store import store, utcnow
from backend.media.timeline import dubbed_slots
from backend.media.video_input import run, write_preview
from backend.rendering.lip_models import checkpoint_for
from backend.rendering.lip_sync import synchronize, voice_only

logger = logging.getLogger(__name__)

RENDER_VERSION = 7  # recorded with each render: which revision of the renderer made it


def render_dub(dub: DubRun, dubbed_audio: Path):
    """Make the dub watchable at once (review cut), then render the lips and mark which lines got them."""
    dub.mark(
        'lipsync',
        'Synchronizing lips',
        dub.span('lipsync', 0),
        timing_warnings=dub.timing_warnings,
        segments=[piece.public() for piece in dub.translated],
        audio_url=f'/api/jobs/{dub.job_id}/audio',
    )
    # The dub is audible now, over the original picture.
    write_review_cut(dub.job_id, dub.directory, dub.video, dubbed_audio)
    if dub.faces_ahead is not None and not dub.faces_ahead.done():
        dub.mark('lipsync', 'Finishing face analysis', dub.span('lipsync', 0))
    try:
        if dub.faces_ahead is not None:
            dub.faces_ahead.result()
    except CancelledError:
        raise
    except (
        Exception
    ) as exc:  # noqa: BLE001  (the render redoes whatever the early pass did not finish)
        logger.warning('Early face analysis for %s did not finish: %s', dub.job_id, exc)
    with heavy_stage(dub.control, dub.job_id, 'Waiting for a render slot'):
        dub.control.check()
        synced = render_lips(dub.job_id, dub.directory, dub.control) or set()
    for piece in dub.translated:
        piece.state = 'lip-synced' if round(piece.start, 1) in synced else 'voiced'


def render_lips(job_id: str, directory: Path, control: JobControl) -> set[float]:
    """Render the final video from the saved dub audio and return the starts (rounded to 0.1 s) of the
    lines whose lips were synced. Used by fresh dubs and by lip-sync resumes."""
    job = store.get(job_id) or {}
    model, checkpoint = checkpoint_for(job.get('lip_sync_model'))
    enhance = (
        bool(job.get('enhance_mouth', settings.mouth_enhance)) and settings.esrgan_weights.is_file()
    )
    slots = dubbed_slots(job.get('segments') or [])

    def progress(fraction, label):
        store.save(
            job_id,
            progress=round(stage_progress('lipsync', fraction), 1),
            stage=f'{label} {int(fraction * 100)}%',
        )

    store.save(job_id, lip_sync_model=model, enhance_mouth=enhance)

    def on_preview(ready, total):
        store.save(
            job_id,
            review_url=f'/api/jobs/{job_id}/review',
            review_version=utcnow(),
            review_note=f'{ready} of {total} face spans lip-synced so far' if total else '',
        )

    if model == 'none':
        progress(0.5, 'Writing the dub')
        voice_only(directory / 'source.mp4', directory / 'dubbed_audio.wav', directory)
    else:
        synchronize(
            directory / 'source.mp4',
            directory / 'dubbed_audio.wav',
            directory,
            progress,
            control,
            sync_audio=directory / 'dubbed_voice.wav',
            checkpoint=checkpoint,
            enhance=enhance,
            slots=slots,
            engine='latentsync' if model == 'latentsync' else 'wav2lip',
            on_preview=on_preview,
        )
    store.save(job_id, stage='Preparing the preview')
    write_preview(directory / 'dubbed_video.mp4', directory / 'preview_video.mp4')
    report_path = directory / 'lip_sync_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    synced = {
        round(float(line['start']), 1) for line in report.get('lines', []) if line.get('lip_synced')
    }
    lines = job.get('segments') or []
    for line in lines:
        if line.get('voiced_text'):
            line['state'] = 'lip-synced' if round(float(line['start']), 1) in synced else 'voiced'
    store.save(
        job_id,
        video_url=f'/api/jobs/{job_id}/video',
        audio_url=f'/api/jobs/{job_id}/audio',
        lip_sync=report,
        render_version=RENDER_VERSION,
        segments=lines,
    )
    return synced


def write_review_cut(job_id: str, directory: Path, video: Path, dubbed_audio: Path):
    """The dubbed mix over the original picture (a stream copy, seconds): playable while the lips render."""
    try:
        duration = float(
            json.loads(
                run(
                    'ffprobe',
                    '-v',
                    'error',
                    '-show_entries',
                    'format=duration',
                    '-of',
                    'json',
                    video,
                )
            )['format']['duration']
        )
        partial = directory / 'review_video.part.mp4'
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-i',
            video,
            '-i',
            dubbed_audio,
            '-map',
            '0:v:0',
            '-map',
            '1:a:0',
            '-c:v',
            'copy',
            '-c:a',
            'aac',
            '-b:a',
            '160k',
            '-ar',
            '48000',
            '-af',
            'apad',
            '-t',
            duration,
            '-movflags',
            '+faststart',
            partial,
        )
        partial.replace(directory / 'review_video.mp4')
        store.save(
            job_id,
            review_url=f'/api/jobs/{job_id}/review',
            review_version=utcnow(),
            review_note='dubbed voice over the original picture; lips not rendered yet',
        )
    except Exception as exc:  # noqa: BLE001  (a courtesy for the wait, never a reason to fail)
        logger.warning('Review cut for %s skipped: %s', job_id, exc)
