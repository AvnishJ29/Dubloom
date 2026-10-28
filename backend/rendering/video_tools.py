"""Helpers every lip-sync step shares: the 25 fps render source, the content digests that key the render
caches, frame facts from ffprobe, and running a worker process while streaming its output."""

import hashlib
import json
import subprocess
from pathlib import Path

from backend.media.video_input import run

FPS = 25  # the render source, face analysis and both lip-sync engines all work at this frame rate


def file_digest(path: Path) -> str:
    """SHA-256 of a file, remembered next to it: hashing a multi-GB 4K render source took 80 s per stage pass."""
    stat = path.stat()
    sidecar = path.with_name(path.name + '.digest')
    key = f"{stat.st_size}:{stat.st_mtime_ns}:"
    if sidecar.is_file():
        cached = sidecar.read_text()
        if cached.startswith(key):
            return cached[len(key) :]
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        sidecar.write_text(key + digest)
    except OSError:
        pass
    return digest


def frame_size(path: Path) -> tuple[int, int]:
    """(width, height) of a video's first picture stream; zeros when ffprobe reports none."""
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height',
            '-of',
            'json',
            path,
        )
    )
    stream = (probe.get('streams') or [{}])[0]
    return int(stream.get('width') or 0), int(stream.get('height') or 0)


def frame_count(path: Path) -> int:
    """Frames of a video: from the container's own count (instant); decoded and counted only when it is missing.
    Counting a 4K render source by decoding it cost 84 s on every lip-sync pass."""
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=nb_frames',
            '-of',
            'json',
            path,
        )
    )
    declared = (probe.get('streams') or [{}])[0].get('nb_frames')
    if declared and str(declared).isdigit() and int(declared) > 0:
        return int(declared)
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-count_frames',
            '-show_entries',
            'stream=nb_read_frames',
            '-of',
            'json',
            path,
        )
    )
    return int(probe['streams'][0]['nb_read_frames'])


def frame_rate(path: Path, text: bool = False):
    """The source's frame rate as a number, or as ffmpeg's exact fraction (e.g. 24000/1001)."""
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=r_frame_rate,avg_frame_rate',
            '-of',
            'json',
            path,
        )
    )
    stream = probe['streams'][0] if probe.get('streams') else {}
    fraction = stream.get('r_frame_rate') or '0/1'
    numerator, denominator = (float(x) for x in fraction.split('/'))
    if not denominator or numerator / denominator > 100:  # variable-rate sources: use the average
        fraction = stream.get('avg_frame_rate') or '0/1'
        numerator, denominator = (float(x) for x in fraction.split('/'))
    if text:
        return fraction
    return numerator / denominator if denominator else 0.0


def run_worker(command, cwd, control, on_line, env=None, label='worker'):
    """Run a model worker in its own process, passing each line it prints to `on_line`.

    The process sees the GPU slot held by the calling thread, if any (`backend.jobs.gpus`), and is registered
    with `control`, so cancelling the job stops it. A non-zero exit raises with the worker's last lines.
    """
    from backend.jobs.gpus import pool

    with subprocess.Popen(
        command,
        cwd=cwd,
        env=pool.env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        if control:
            control.register(process)
        tail = []
        try:
            for line in process.stdout:
                tail = (tail + [line])[-25:]
                on_line(line)
            code = process.wait()
        finally:
            if control:
                control.unregister(process)
        if control:
            control.check()
        if code:
            relevant = [
                line for line in tail if 'Warning' not in line and 'warnings.warn' not in line
            ]
            raise RuntimeError(f'{label} failed: ' + ''.join(relevant)[-1600:])


def prepare_render_source(video: Path, directory: Path) -> Path:
    """The 25 fps picture every lip-sync step works on; rebuilt only when the source changes."""
    normalized = directory / 'render_source.mp4'
    source_digest = directory / 'render_source.sha256'
    video_digest = file_digest(video) + ':q8'
    if not (
        normalized.is_file()
        and source_digest.is_file()
        and source_digest.read_text() == video_digest
    ):
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-i',
            video,
            '-an',
            '-vf',
            f'fps={FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-crf',
            '8',
            '-pix_fmt',
            'yuv420p',
            normalized,
        )  # near-lossless: these pixels end up in the dub
        source_digest.write_text(video_digest)
    return normalized
