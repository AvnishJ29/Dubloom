"""LatentSync 1.6, the GPU lip-sync engine: redraws the lower face of each speaking target in spans that the
worker (`backend/workers/latentsync_worker.py`, own environment) caches, and a growing review preview of those
spans while the render runs."""

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path

from backend.config import ROOT, settings
from backend.media.video_input import run
from backend.rendering import finish
from backend.rendering.face_analysis import face_analysis_available
from backend.rendering.video_tools import FPS, frame_rate, frame_size, run_worker

logger = logging.getLogger(__name__)

PREVIEW_EVERY = 45  # seconds between rebuilds of the growing preview while spans render
PREVIEW_HEIGHT = 720
PREVIEW_MAX_SOURCE = 1080  # the growing preview is built only for sources up to 1080p (decoding 4K every minute is too slow)


def latentsync_available() -> bool:
    """True when the LatentSync checkpoint, its config and face analysis are installed."""
    return (
        bool(settings.latentsync_python)
        and (settings.latentsync_checkpoints / 'latentsync_unet.pt').is_file()
        and (settings.latentsync_dir / 'configs' / 'unet' / 'stage2_512.yaml').is_file()
        and face_analysis_available()
    )


def render_latentsync(
    normalized,
    sync_audio,
    targets_path,
    output,
    directory,
    control,
    progress,
    spans_only: bool = False,
) -> dict:
    """Redraw the speaking targets with LatentSync and return the render summary (synced and reused frames,
    seconds). With `spans_only` the worker writes only the rendered spans and their manifest
    (`latentsync_cache/spans.json`) for `finish` to paste; otherwise the whole video goes to `output`.
    """
    voice16 = directory / 'lips_voice16k.wav'
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        sync_audio,
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'pcm_s16le',
        voice16,
    )
    command = [
        settings.latentsync_python,
        '-u',
        str(ROOT / 'backend' / 'workers' / 'latentsync_worker.py'),
        str(normalized),
        str(voice16),
        str(targets_path),
        str(output),
        '--repo',
        str(settings.latentsync_dir),
        '--checkpoints',
        str(settings.latentsync_checkpoints),
        '--vae',
        str(settings.latentsync_vae),
        '--steps',
        str(settings.latentsync_steps),
        '--guidance',
        str(settings.latentsync_guidance),
        '--device',
        settings.device,
        '--cache',
        str(directory / 'latentsync_cache'),
    ]
    if (directory / 'render_source.sha256').is_file():
        command += ['--video-key', (directory / 'render_source.sha256').read_text()]
    if settings.latentsync_deepcache:
        command.append('--deepcache')
    if spans_only:
        command.append('--spans-only')
    command += ['--windows', str(settings.latentsync_windows)]
    result = {
        'synced_frames': 0,
        'render_seconds': 0.0,
        'model': 'LatentSync 1.6',
        'settings': {'steps': settings.latentsync_steps, 'guidance': settings.latentsync_guidance},
    }

    def on_line(line):
        if line.startswith('Rendered '):
            parts = line.split()
            done, total = int(parts[1]), int(parts[3])
            progress(done / max(1, total))
        elif line.startswith('DONE '):
            fields = dict(item.split('=') for item in line.split()[1:])
            result['synced_frames'] = int(fields['synced'])
            result['render_seconds'] = float(fields['seconds'])
            result['reused_frames'] = int(fields.get('reused', 0))

    run_worker(command, ROOT, control, on_line, label='LatentSync')
    return result


@contextmanager
def growing_preview(
    video: Path, normalized: Path, directory: Path, audio: Path, total_frames, on_preview
):
    """While the block renders, rebuild the review cut in the background (see `_grow_preview`), when
    `on_preview` is given and the source is small enough to decode every PREVIEW_EVERY seconds."""
    stop = threading.Event()
    grower = None
    if on_preview and min(frame_size(normalized)) <= PREVIEW_MAX_SOURCE:
        grower = threading.Thread(
            target=_grow_preview,
            args=(
                normalized,
                directory,
                audio,
                total_frames,
                frame_rate(video),
                stop,
                on_preview,
            ),
            name='preview',
            daemon=True,
        )
        grower.start()
    try:
        yield
    finally:
        stop.set()
        if grower:
            grower.join(timeout=120)


def _grow_preview(
    normalized: Path,
    directory: Path,
    audio: Path,
    total_frames: int,
    source_rate,
    stop: threading.Event,
    on_preview,
):
    """Every PREVIEW_EVERY seconds: the spans rendered so far (raw, unrestored) pasted into a 720p review cut."""
    manifest_path = directory / 'latentsync_cache' / 'spans.json'
    built = -1
    while not stop.wait(PREVIEW_EVERY):
        try:
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            ready = [s for s in manifest['spans'] if Path(s['file']).is_file()]
            if not ready or len(ready) == built or manifest.get('complete'):
                continue
            plan = [
                {
                    'first': s['first'],
                    'end': s['end'],
                    'raw': Path(s['file']),
                    'restored': Path(s['file']),
                }
                for s in ready
            ]
            partial = directory / 'review_video.part.mp4'
            finish.assemble(
                normalized,
                plan,
                audio,
                partial,
                total_frames,
                FPS,
                source_rate,
                None,
                None,
                preview_height=PREVIEW_HEIGHT,
            )
            partial.replace(directory / 'review_video.mp4')
            built = len(ready)
            if on_preview:
                on_preview(len(ready), len(manifest['spans']))
        except Exception as exc:  # noqa: BLE001  (the preview is a courtesy; the render goes on)
            logger.warning('Growing preview for %s skipped: %s', directory.name, exc)
