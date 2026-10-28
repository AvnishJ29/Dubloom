"""Work that runs beside the model stages: vocal separation and the early face analysis.

Both depend only on the source video, so they start right after the speakers are known and
finish while the script is transcribed, translated and voiced.
"""

from __future__ import annotations

from pathlib import Path

from backend.config import settings
from backend.jobs.context import DubRun
from backend.jobs.control import JobControl, background_executor, heavy_slots, separation_slots
from backend.jobs.store import store
from backend.media.separation import separate_background, separation_available, separation_stamp
from backend.rendering.lip_sync import precompute_faces


def start_background_work(dub: DubRun):
    """Choose how the original background is kept, then start separation and face analysis."""
    if dub.background_mode == 'separate':
        if not separation_available():
            dub.background_mode = 'duck'
            dub.background_notes.append(
                'Vocal separation is not installed; the original mix was ducked under the dub instead.'
            )
        elif dub.video_duration > settings.separation_max_minutes * 60:
            dub.background_mode = 'duck'
            dub.background_notes.append(
                f'Vocal separation is skipped above {settings.separation_max_minutes} minutes; '
                'the original mix was ducked under the dub instead.'
            )
    if dub.background_mode == 'separate':
        dub.separation = background_executor.submit(
            separate_in_background, dub.job_id, dub.directory, dub.video, dub.control
        )
    else:
        dub.save(
            background={
                'mode': dub.background_mode,
                'status': 'ready' if dub.background_mode == 'duck' else 'off',
                'progress': 1,
            }
        )
    # Faces found now let the lip-sync stage start straight at the render.
    voice_only_dub = (dub.job.get('lip_sync_model') or '').lower() == 'none'
    dub.faces_ahead = (
        None
        if voice_only_dub
        else background_executor.submit(
            prepare_faces_in_background,
            dub.job_id,
            dub.directory,
            dub.video,
            dub.control,
            dub.separation,
        )
    )


def separate_in_background(job_id: str, directory: Path, video: Path, control: JobControl) -> Path:
    """Split the background from the dialogue (reusing a saved split); returns `background.wav`."""
    stamp = directory / 'separation.txt'
    if (
        (directory / 'background.wav').is_file()
        and stamp.is_file()
        and stamp.read_text() == separation_stamp()
    ):
        store.save(job_id, background={'mode': 'separate', 'status': 'ready', 'progress': 1})
        return directory / 'background.wav'
    store.save(job_id, background={'mode': 'separate', 'status': 'queued', 'progress': 0})
    while not separation_slots.acquire(timeout=1):
        control.check()
    try:
        control.check()
        last = {'step': -1}

        def progress(fraction):
            step = int(fraction * 25)
            if step != last['step']:
                last['step'] = step
                store.save(
                    job_id,
                    background={
                        'mode': 'separate',
                        'status': 'running',
                        'progress': round(fraction, 2),
                    },
                )

        store.save(job_id, background={'mode': 'separate', 'status': 'running', 'progress': 0})
        result = separate_background(video, directory, control, progress)
        store.save(job_id, background={'mode': 'separate', 'status': 'ready', 'progress': 1})
        return result
    finally:
        separation_slots.release()


def prepare_faces_in_background(
    job_id: str, directory: Path, video: Path, control: JobControl, separation
):
    """Face analysis and active-speaker detection inside a GPU slot while the model stages run."""
    while not heavy_slots.acquire(timeout=1):
        control.check()
    try:
        control.check()
        precompute_faces(video, directory, control, separation)
    finally:
        heavy_slots.release()
