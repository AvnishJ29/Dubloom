"""Vocal separation (Demucs) so the background bed carries music and ambience only.

Ducking the original mix leaves the original voices faintly audible under the
dub. Removing the vocal stem instead gives a clean bed with no echo of the
source speech. Runs as a subprocess so a cancelled job can stop it.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from backend.config import ROOT, settings
from backend.media.video_input import run

logger = logging.getLogger(__name__)


def separation_available() -> bool:
    if settings.separation_backend == 'roformer':
        return Path(settings.separate_python).is_file()
    try:
        import demucs  # noqa: F401

        return True
    except ImportError:
        return False


def separation_stamp() -> str:
    """What produced the cached stems; a different backend or model rebuilds them."""
    return f"{settings.separation_backend}:{settings.roformer_model if settings.separation_backend == 'roformer' else settings.separation_model}"


def extract_full_audio(video: Path, directory: Path) -> Path:
    """Full-quality stereo audio of the source (cached per job)."""
    full = directory / 'source_audio_full.wav'
    if not (full.is_file() and full.stat().st_size > 1000):
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-i',
            video,
            '-vn',
            '-ac',
            '2',
            '-ar',
            '44100',
            '-c:a',
            'pcm_s16le',
            full,
        )
    return full


def _cuda_torch() -> bool:
    """Demucs runs on the GPU when the studio's torch is a CUDA build and a GPU is configured."""
    if settings.device != 'cuda':
        return False
    try:
        return '+cpu' not in importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        return False


def separate_background(video: Path, directory: Path, control=None, progress=None) -> Path:
    """Return a 44.1 kHz stereo WAV of the source without vocals (cached per job)."""
    output = directory / 'background.wav'
    stamp = directory / 'separation.txt'
    if (
        output.is_file()
        and output.stat().st_size > 1000
        and stamp.is_file()
        and stamp.read_text() == separation_stamp()
    ):
        return output
    full = extract_full_audio(video, directory)
    stems = directory / 'stems'
    shutil.rmtree(stems, ignore_errors=True)
    device = 'cuda' if _cuda_torch() else 'cpu'
    if settings.separation_backend == 'roformer':
        command = [
            str(settings.separate_python),
            '-u',
            str(ROOT / 'backend' / 'workers' / 'separate_worker.py'),
            str(full),
            str(stems),
            '--model',
            settings.roformer_model,
            '--model-dir',
            str(settings.separator_models),
        ]
    else:
        command = [
            sys.executable,
            '-m',
            'demucs.separate',
            '--two-stems',
            'vocals',
            '-n',
            settings.separation_model,
            '-d',
            device,
            '-j',
            str(settings.separation_jobs if device == 'cpu' else 0),
            '-o',
            str(stems),
            str(full),
        ]
    env = {
        'OMP_NUM_THREADS': str(settings.separation_threads),
        'PATH': '/usr/bin:/bin',
        'HOME': str(Path.home()),
    }
    env.update(
        {
            key: os.environ[key]
            for key in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')
            if key in os.environ
        }
    )
    from backend.jobs.gpus import pool

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=pool.env(env)
    )
    if control:
        control.register(process)
    tail = []
    try:
        for line in process.stdout:
            for piece in line.replace('\r', '\n').split('\n'):
                piece = piece.strip()
                if not piece:
                    continue
                tail = (tail + [piece])[-15:]
                match = re.search(r'(\d+)%\|', piece)
                if match and progress:
                    progress(int(match.group(1)) / 100)
        code = process.wait()
    finally:
        if control:
            control.unregister(process)
    if control:
        control.check()
    if code:
        raise RuntimeError('Vocal separation failed: ' + ' | '.join(tail)[-600:])
    produced = sorted(stems.rglob('*.wav'))

    def stem_named(
        *words,
    ):  # audio-separator names stems after the model's own labels: (Instrumental), (other), (vocals)...
        return next((p for p in produced if any(f'({w})' in p.name.lower() for w in words)), None)

    stem = next(stems.rglob('no_vocals.wav'), None) or stem_named(
        'instrumental', 'other', 'no vocals'
    )
    if stem is None:
        raise RuntimeError('Vocal separation produced no background stem')
    shutil.move(str(stem), str(output))
    vocals = next(stems.rglob('vocals.wav'), None) or stem_named('vocals')
    if vocals is not None:
        # Clean speech without music: the best source for voice-cloning references.
        shutil.move(str(vocals), str(directory / 'vocals.wav'))
    shutil.rmtree(stems, ignore_errors=True)
    stamp.write_text(separation_stamp())
    return output
