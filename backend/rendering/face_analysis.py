"""Face analysis of the render source: every face in every frame (InsightFace), hands over mouths (MediaPipe)
and how likely each face is to be speaking (Light-ASD).

Each step runs a model worker in its own environment (`backend/workers/`) and is cached next to the job under
a digest of its inputs, so a later pass (or the early background pass, `lip_sync.precompute_faces`) finds the
result on disk and skips the work.
"""

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from backend.config import ROOT, settings
from backend.media.video_input import run
from backend.rendering.video_tools import file_digest, frame_count, run_worker

logger = logging.getLogger(__name__)


def face_analysis_available() -> bool:
    """True when the face worker's Python and the InsightFace models are installed."""
    return (
        bool(settings.latentsync_python)
        and (settings.insightface_root / 'models' / 'buffalo_l').is_dir()
    )


def asd_available() -> bool:
    """True when face analysis and the Light-ASD weights are installed."""
    return face_analysis_available() and (settings.asd_dir / settings.asd_weights).is_file()


def speech_audio(directory: Path) -> Path:
    """The job's speech: the separated vocals stem when there is one (cleaner), else the source soundtrack."""
    return (
        directory / 'vocals.wav'
        if (directory / 'vocals.wav').is_file()
        else directory / 'source_audio.wav'
    )


def analyse_faces(normalized: Path, directory: Path, control=None, progress=None) -> Path | None:
    """Per-frame faces for the normalised video (cached by video content)."""
    if not face_analysis_available():
        return None
    output = directory / 'faces.npz'
    stamp = directory / 'faces.sha256'
    video_digest = file_digest(normalized)
    digest = f"{video_digest}:{settings.face_det_size}:{settings.face_stride}:g2"  # g2: faces carry gender and age
    if output.is_file() and stamp.is_file():
        # Reuse an analysis of the same video at the same detector size and the same or a finer stride
        # (stamps written before strides existed were per-frame at 640).
        cached = (stamp.read_text().split(':') + ['640', '1', ''])[:4]
        if (
            cached[0] == video_digest
            and cached[1] == str(settings.face_det_size)
            and cached[2].isdigit()
            and int(cached[2]) <= settings.face_stride
            and cached[3] == 'g2'
        ):
            return (
                output  # same video, detector and stride, and the analysis carries gender and age
            )

    total = frame_count(normalized)
    # On CPU the models are bound by per-call overhead, not cores: several workers on parts of the video are faster.
    workers = max(1, min(settings.face_workers, total // 250 or 1))
    bounds = [round(total * k / workers) for k in range(workers + 1)]
    done = [0] * workers
    parts = [directory / f'faces_part{k}.npz' for k in range(workers)]

    def run_part(k):
        def on_line(line):
            if progress and line.startswith('PROGRESS '):
                done[k] = int(line.split()[1])
                progress(sum(done) / max(1, total))

        command = [
            settings.latentsync_python,
            '-u',
            str(ROOT / 'backend' / 'workers' / 'face_worker.py'),
            str(normalized),
            str(parts[k]),
            '--root',
            str(settings.insightface_root),
            '--device',
            settings.device,
            '--det-size',
            str(settings.face_det_size),
            '--stride',
            str(settings.face_stride),
            '--threads',
            str(max(2, (os.cpu_count() or 4) // workers)),
            '--first',
            str(bounds[k]),
            '--last',
            str(bounds[k + 1] if k + 1 < workers else -1),
        ]
        run_worker(command, ROOT, control, on_line, label='Face analysis')

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(run_part, k) for k in range(workers)]:
            future.result()
    _merge_faces(parts, output)
    for part in parts:
        part.unlink(missing_ok=True)
    stamp.write_text(digest)
    return output


def _merge_faces(parts: list[Path], output: Path):
    """Join the workers' consecutive parts of the video into one faces file."""
    loaded = [dict(np.load(part)) for part in parts]
    if len(loaded) == 1:
        merged = loaded[0]
    else:
        merged = {key: loaded[0][key] for key in ('fps', 'width', 'height')}
        merged['frame_count'] = loaded[-1]['frame_count']
        merged['first'] = 0
        for key in (
            'thumbs',
            'frame',
            'bbox',
            'score',
            'kps',
            'lmk',
            'emb',
            'mouth',
            'gender',
            'age',
        ):
            if all(key in part for part in loaded):
                merged[key] = np.concatenate([part[key] for part in loaded])
    np.savez_compressed(output, **merged)


def detect_hands(normalized: Path, directory: Path, control=None, frames=None) -> Path | None:
    """Hand boxes per frame (MediaPipe), cached by the render source; None when the detector is unavailable.
    `frames`: the frames that have a face; the others are never lip-synced, so they are not searched.
    """
    output, stamp = directory / 'hands.npz', directory / 'hands.sha256'
    digest = file_digest(normalized) + ':h1'
    if output.is_file() and stamp.is_file() and stamp.read_text() == digest:
        return output
    if not Path(settings.hands_python).is_file():
        logger.warning(
            'Hand detection skipped: %s is missing (run scripts/setup_models.sh)',
            settings.hands_python,
        )
        return None
    command = [
        settings.hands_python,
        '-u',
        str(ROOT / 'backend' / 'workers' / 'hands_worker.py'),
        str(normalized),
        str(output),
    ]
    if frames is not None:
        wanted = directory / 'hands_frames.npy'
        np.save(wanted, np.asarray(sorted(set(int(f) for f in frames)), dtype=np.int64))
        command += ['--frames', str(wanted)]
    try:
        run_worker(command, ROOT, control, lambda line: None, label='Hand detection')
    except Exception as exc:  # noqa: BLE001  (no hands data: nothing is excluded, as before)
        logger.warning('Hand detection unavailable: %s', exc)
        return None
    stamp.write_text(digest)
    return output


def detect_active_speakers(
    normalized: Path,
    directory: Path,
    faces_path: Path,
    track_of: np.ndarray,
    control=None,
    progress=None,
) -> np.ndarray | None:
    """Light-ASD score per face detection (>= 0: speaking), cached; None if unavailable."""
    if not asd_available():
        return None
    speech = speech_audio(directory)
    if not speech.is_file():
        return None
    audio16k = directory / 'asd_audio16k.wav'
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        speech,
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'pcm_s16le',
        '-map_metadata',
        '-1',
        '-fflags',
        '+bitexact',
        audio16k,
    )
    tracks_path = directory / 'face_tracks.npz'
    np.savez_compressed(tracks_path, track=track_of)
    output, stamp = directory / 'asd.npz', directory / 'asd.sha256'
    digest = hashlib.sha256(
        f"{(directory / 'faces.sha256').read_text()}:{file_digest(audio16k)}:{settings.asd_weights}:".encode()
        + np.ascontiguousarray(track_of).tobytes()
    ).hexdigest()
    if not (output.is_file() and stamp.is_file() and stamp.read_text() == digest):

        def on_line(line):
            if progress and line.startswith('PROGRESS '):
                _, done, total = line.split()
                progress(0.8 * int(done) / max(1, int(total)))
            elif progress and line.startswith('Scored '):
                parts = line.split()
                progress(0.8 + 0.2 * int(parts[1]) / max(1, int(parts[3])))

        command = [
            settings.latentsync_python,
            '-u',
            str(ROOT / 'backend' / 'workers' / 'asd_worker.py'),
            str(normalized),
            str(audio16k),
            str(faces_path),
            str(tracks_path),
            str(output),
            '--repo',
            str(settings.asd_dir),
            '--weights',
            settings.asd_weights,
            '--device',
            settings.device,
        ]
        run_worker(command, ROOT, control, on_line, label='Active speaker detection')
        stamp.write_text(digest)
    with np.load(output) as data:
        scores = data['score']
    return scores if len(scores) == len(track_of) else None
