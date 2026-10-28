"""Wav2Lip, the CPU lip-sync engine (bundled in `Wav2Lip/`): the render source is rendered in chunks of
`wav2lip_chunk_seconds`, and a chunk is rendered again only when its own picture, voice or speaking-face
targets change; the chunks are then joined into one silent picture."""

import hashlib
import json
import math
import os

import numpy as np

from backend.config import ROOT, settings
from backend.media.video_input import run
from backend.rendering.video_tools import FPS, file_digest, frame_count, run_worker


def render_wav2lip(
    normalized,
    sync_audio,
    targets_path,
    output,
    directory,
    total_frames,
    checkpoint,
    enhance,
    control,
    progress,
) -> dict:
    """Lip-sync the render source into `output` (no sound) and return the render summary. Without
    `targets_path` Wav2Lip finds the faces itself (SFD) instead of following the studio's speaking targets.
    """
    checkpoint = checkpoint or settings.wav2lip_accurate_checkpoint
    if not checkpoint.is_file():
        checkpoint = settings.wav2lip_checkpoint
    enhance = bool(enhance and settings.esrgan_weights.is_file())
    chunk_frames = max(1, round(settings.wav2lip_chunk_seconds * FPS))
    count = math.ceil(total_frames / chunk_frames)
    pictures, reports = [], []
    targets = _target_arrays(targets_path)
    source_digest = (
        (directory / 'render_source.sha256').read_text()
        if (directory / 'render_source.sha256').is_file()
        else ''
    )
    for index, start_frame in enumerate(range(0, total_frames, chunk_frames)):
        if control:
            control.check()
        frames = min(chunk_frames, total_frames - start_frame)
        face = directory / f'lip_face_{index:04d}.mp4'
        sound = directory / f'lip_audio_{index:04d}.wav'
        synced = directory / f'lip_done_{index:04d}.mp4'
        report_path = directory / f'lip_report_{index:04d}.json'
        picture = directory / f'lip_picture_{index:04d}.mp4'
        progress(index / count)
        _cut_voice(sync_audio, start_frame, frames, sound)
        # A chunk is re-rendered only when its own picture, voice or speaking-face targets changed.
        stamp = {
            'video': source_digest,
            'targets': _slice_digest(targets, start_frame, start_frame + frames),
            'audio': file_digest(sound),
            'checkpoint': checkpoint.name,
            'enhance': enhance,
            'start': start_frame,
            'frames': frames,
        }
        report = _reusable_report(report_path, synced, stamp, frames)
        if report is None:
            _cut_picture(normalized, start_frame, frames, face)
            command = _wav2lip_command(
                checkpoint, face, sound, synced, report_path, targets_path, start_frame, enhance
            )
            env = {
                **os.environ,
                'NUMBA_DISABLE_JIT': '1',
                'OMP_NUM_THREADS': str(settings.wav2lip_threads),
            }

            def on_line(line, index=index, frames=frames):
                if line.startswith('Rendered '):
                    progress((index + min(int(line.split()[1]) / frames, 0.999)) / count)

            run_worker(command, ROOT / 'Wav2Lip', control, on_line, env=env, label='Wav2Lip')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report['frames'] != frames:
                raise RuntimeError(f'Lip-sync chunk {index + 1} lost frames')
            report['stamp'] = stamp
            report_path.write_text(json.dumps(report), encoding='utf-8')
        reports.append(report)
        _picture_only(synced, picture)
        pictures.append(picture)
    _join_pictures(pictures, directory / 'lip_chunks.txt', output)
    return {
        'synced_frames': sum(r['synced_frames'] for r in reports),
        'model': checkpoint.stem,
        'render_seconds': round(sum(r.get('render_seconds', 0) for r in reports), 2),
        'mouth_enhanced': enhance,
        'detector': 'studio targets' if targets_path else 'SFD',
    }


def _target_arrays(targets_path):
    """The per-frame target arrays a chunk's stamp covers, or None when Wav2Lip detects faces itself."""
    if not targets_path:
        return None
    with np.load(targets_path) as data:
        return {key: data[key] for key in ('bbox', 'lmk', 'track')}


def _slice_digest(arrays, start: int, stop: int) -> str:
    """Digest of the targets in frames [start, stop): an edit elsewhere in the video leaves it unchanged."""
    if arrays is None:
        return 'detector'
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(np.ascontiguousarray(arrays[key][start:stop]).tobytes())
    return digest.hexdigest()


def _reusable_report(report_path, synced, stamp: dict, frames: int) -> dict | None:
    """The chunk's earlier report, when it was rendered from the same stamp and its video is complete."""
    if report_path.is_file() and synced.is_file():
        cached = json.loads(report_path.read_text(encoding='utf-8'))
        if cached.get('stamp') == stamp and frame_count(synced) == frames:
            return cached
    return None


def _cut_voice(sync_audio, start_frame: int, frames: int, sound):
    """The chunk's voice (plus 0.2 s) as 16 kHz mono PCM, written bit-exactly so its digest is stable."""
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-ss',
        start_frame / FPS,
        '-i',
        sync_audio,
        '-t',
        frames / FPS + 0.2,
        '-ar',
        '16000',
        '-ac',
        '1',
        '-c:a',
        'pcm_s16le',
        '-map_metadata',
        '-1',
        '-fflags',
        '+bitexact',
        sound,
    )


def _cut_picture(normalized, start_frame: int, frames: int, face):
    """The chunk's frames of the render source, as the clip Wav2Lip draws on."""
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-ss',
        start_frame / FPS,
        '-i',
        normalized,
        '-frames:v',
        frames,
        '-an',
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '14',
        face,
    )


def _wav2lip_command(
    checkpoint, face, sound, synced, report_path, targets_path, start_frame: int, enhance: bool
) -> list[str]:
    """The `Wav2Lip/inference.py` call for one chunk."""
    command = [
        settings.wav2lip_python,
        '-u',
        'inference.py',
        '--checkpoint_path',
        str(checkpoint),
        '--face',
        str(face),
        '--audio',
        str(sound),
        '--outfile',
        str(synced),
        '--report',
        str(report_path),
        '--threads',
        str(settings.wav2lip_threads),
        '--face_det_interval',
        str(settings.wav2lip_face_interval),
        '--runtime',
        settings.lip_sync_runtime,
        '--openvino_device',
        settings.openvino_device,
        '--resize_factor',
        str(settings.wav2lip_resize_factor),
    ]
    if targets_path:
        command += ['--targets', str(targets_path), '--target_offset', str(start_frame)]
    if enhance:
        command += ['--enhance_weights', str(settings.esrgan_weights)]
    return command


def _picture_only(synced, picture):
    """Drop per-chunk AAC padding before concatenating so later chunks stay aligned with the speech."""
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        synced,
        '-map',
        '0:v:0',
        '-an',
        '-c:v',
        'copy',
        picture,
    )


def _join_pictures(pictures, listing, output):
    """Concatenate the chunk pictures into `output` without re-encoding."""
    listing.write_text('\n'.join(f"file '{path.name}'" for path in pictures), encoding='utf-8')
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-f',
        'concat',
        '-safe',
        '0',
        '-i',
        listing,
        '-c:v',
        'copy',
        output,
    )
