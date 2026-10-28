"""Finish a LatentSync render from its cached spans: restore each span once (cached next to it), then paste the spans
into the picture and encode the dub exactly once.

Before, every render wrote the whole video four times (LatentSync, GFPGAN, Real-ESRGAN, export) and restored every
synced frame again, so a one-line edit cost as much as a full render. Now a span is restored only when its raw frames
or the restore settings change, and the single encode is the only whole-video pass."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from backend.config import ROOT, settings
from backend.rendering.mouth_enhance import MIN_FACE_WIDTH, enhance_lips
from backend.media.video_input import run

logger = logging.getLogger(__name__)
GFPGAN_MIN = 60  # the GFPGAN worker's floor (restore_worker.py --min-face)
GOP = 25  # a keyframe every second: later edits can be stitched at second boundaries


def restore_stamp(gfpgan: bool, esrgan: bool) -> str:
    split = settings.restore_split_px if gfpgan else 0
    return f"g{GFPGAN_MIN}-{split}t{settings.restore_temporal:g}e{int(esrgan)}"


def plan_restoration(manifest: dict, targets: Path, gfpgan: bool, esrgan: bool) -> list[dict]:
    """Per span: which passes it needs and the cached restored file (if already made). Pure, so it is unit-tested."""
    with np.load(targets) as data:
        track, bbox = data['track'], data['bbox']
    widths = np.where(track >= 0, bbox[:, 2] - bbox[:, 0], -1.0)
    split = settings.restore_split_px if gfpgan else 0
    stamp = restore_stamp(gfpgan, esrgan)
    plan = []
    for span in manifest['spans']:
        first, end, raw = span['first'], span['end'], Path(span['file'])
        inside = widths[first:end]
        needs_gfpgan = gfpgan and bool(((inside >= GFPGAN_MIN) & (inside < split)).any())
        needs_esrgan = esrgan and bool((inside >= max(split, MIN_FACE_WIDTH)).any())
        restored = (
            raw.with_name(f"{raw.stem}.{stamp}.mp4") if (needs_gfpgan or needs_esrgan) else raw
        )
        plan.append(
            {
                'first': first,
                'end': end,
                'raw': raw,
                'restored': restored,
                'gfpgan': needs_gfpgan,
                'esrgan': needs_esrgan,
                'cached': restored == raw or restored.is_file(),
            }
        )
    return plan


def restore_spans(
    manifest: dict, targets: Path, control=None, progress=None
) -> tuple[list[dict], dict]:
    """Restore every span that is not cached yet. Returns the plan and a report for `mouth_enhanced`."""
    gfpgan = (
        settings.lip_restore == 'gfpgan'
        and Path(settings.restore_python).is_file()
        and settings.gfpgan_weights.is_file()
    )
    esrgan = settings.esrgan_weights.is_file()
    plan = plan_restoration(manifest, targets, gfpgan, esrgan)
    todo = [span for span in plan if not span['cached']]
    report = {
        'applied': False,
        'model': 'GFPGAN v1.4' if gfpgan else ('Real-ESRGAN x2' if esrgan else None),
        'spans': len(plan),
        'spans_reused': len(plan) - len(todo),
        'temporal': settings.restore_temporal,
        'max_face': settings.restore_split_px if gfpgan else 0,
    }
    if not gfpgan and not esrgan:
        report['reason'] = 'no restoration model installed'
        return plan, report
    if not todo:
        report['applied'] = any(span['restored'] != span['raw'] for span in plan)
        return plan, report
    started = time.time()
    # ---- GFPGAN: one worker process over every span that needs it (model loaded once) ----
    gfpgan_jobs = [span for span in todo if span['gfpgan']]
    stage_one = {}
    if gfpgan_jobs:
        jobs = []
        for span in gfpgan_jobs:
            out = (
                span['restored']
                if not span['esrgan']
                else span['restored'].with_suffix('.gfpgan.mp4')
            )
            stage_one[id(span)] = out
            jobs.append({'input': str(span['raw']), 'output': str(out), 'first': span['first']})
        # several worker processes side by side (GFPGAN is light on the GPU; the frame loop is the cost), longest spans first
        jobs.sort(
            key=lambda job: -next(
                s['end'] - s['first'] for s in gfpgan_jobs if str(s['raw']) == job['input']
            )
        )
        workers = max(1, min(settings.restore_workers, len(jobs)))
        batches = [jobs[k::workers] for k in range(workers)]
        total = sum(span['end'] - span['first'] for span in gfpgan_jobs)
        counted = {'frames': 0, 'done': [0] * workers}
        lock = threading.Lock()
        from backend.rendering.lip_sync import _stream

        def run_batch(k):
            batch = Path(manifest['spans'][0]['file']).parent / f'restore_batch_{k}.json'
            batch.write_text(json.dumps(batches[k]))

            def on_line(line):
                if line.startswith('Restored '):
                    with lock:
                        counted['done'][k] = int(line.split()[1])
                        if progress and total:
                            progress(min(0.99, sum(counted['done']) / total))
                elif line.startswith('DONE '):
                    fields = dict(item.split('=') for item in line.split()[1:])
                    with lock:
                        counted['frames'] += int(fields['restored'])

            command = [
                settings.restore_python,
                '-u',
                str(ROOT / 'backend' / 'workers' / 'restore_worker.py'),
                '--batch',
                str(batch),
                '--targets-file',
                str(targets),
                '--weights',
                str(settings.gfpgan_weights),
                '--max-face',
                str(settings.restore_split_px),
                '--temporal',
                str(settings.restore_temporal),
            ]
            _stream(command, ROOT, control, on_line, label='Face restoration')

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(run_batch, k) for k in range(workers)]:
                future.result()
        report.update(
            frames=counted['frames'], seconds=round(time.time() - started, 1), workers=workers
        )
    # ---- Real-ESRGAN sharpening of the close-ups, spans side by side in threads (the model is shared) ----
    sharpened, started_esrgan = 0, time.time()

    def sharpen(span):
        source = stage_one.get(id(span), span['raw'])
        result = enhance_lips(
            Path(source),
            targets,
            span['restored'],
            settings.device,
            min_face=max(settings.restore_split_px if gfpgan else 0, MIN_FACE_WIDTH),
            first=span['first'],
        )
        if source != span['raw']:
            Path(source).unlink(missing_ok=True)
        return int(result.get('frames') or 0)

    esrgan_spans = [span for span in todo if span['esrgan']]
    if esrgan_spans:
        with ThreadPoolExecutor(
            max_workers=max(1, min(settings.restore_workers, len(esrgan_spans)))
        ) as pool:
            for future in [pool.submit(sharpen, span) for span in esrgan_spans]:
                sharpened += future.result()
                if control:
                    control.check()
    for span in todo:
        if not span['restored'].is_file():
            raise RuntimeError(
                f"restoration produced no file for frames {span['first']}-{span['end']}"
            )
    if any(span['esrgan'] for span in todo):
        report['esrgan'] = {
            'applied': sharpened > 0,
            'model': 'Real-ESRGAN x2',
            'frames': sharpened,
            'seconds': round(time.time() - started_esrgan, 1),
        }
    report['applied'] = any(span['restored'] != span['raw'] for span in plan)
    report['seconds'] = round(time.time() - started, 1)
    return plan, report


def assemble(
    render_source: Path,
    plan: list[dict],
    audio: Path,
    final: Path,
    total_frames: int,
    fps: float,
    source_rate: float | None,
    control=None,
    progress=None,
    preview_height: int | None = None,
) -> None:
    """Paste the restored spans into the picture and write the finished dub in one encode.
    With `preview_height` the output is a quick lighter rendition (the growing preview while a render runs).
    """
    capture = cv2.VideoCapture(str(render_source))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    filters = [f'fps={source_rate}'] if source_rate and abs(source_rate - fps) > 0.01 else []
    if preview_height and min(width, height) > preview_height:
        filters.append(
            f'scale=-2:{preview_height}' if height <= width else f'scale={preview_height}:-2'
        )
    picture = ['-vf', ','.join(filters)] if filters else []
    quality = (
        ['-preset', 'veryfast', '-crf', '23']
        if preview_height
        else ['-preset', 'medium', '-crf', '12']
    )
    command = [
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',
        '-s',
        f'{width}x{height}',
        '-r',
        str(fps),
        '-i',
        '-',
        '-i',
        str(audio),
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        *picture,
        '-c:v',
        'libx264',
        *quality,
        '-g',
        str(GOP),
        '-keyint_min',
        str(GOP),
        '-sc_threshold',
        '0',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-ar',
        '48000',
        '-af',
        'apad',
        '-t',
        str(total_frames / fps),
        '-movflags',
        '+faststart',
        str(final),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    if control:
        control.register(writer)
    spans = sorted(plan, key=lambda span: span['first'])
    position = 0
    try:
        for span in spans:
            while position < span['first']:  # untouched picture straight from the render source
                ok, frame = capture.read()
                if not ok:
                    break
                writer.stdin.write(frame.tobytes())
                position += 1
            reader = cv2.VideoCapture(str(span['restored']))
            for _ in range(span['end'] - span['first']):
                ok, frame = reader.read()
                if not ok:
                    raise RuntimeError(f"restored span {span['restored'].name} is short")
                writer.stdin.write(frame.tobytes())
                position += 1
            reader.release()
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            if progress:
                progress(min(0.99, position / max(1, total_frames)))
            if control:
                control.check()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.stdin.write(frame.tobytes())
            position += 1
    finally:
        capture.release()
        writer.stdin.close()
        code = writer.wait()
        if control:
            control.unregister(writer)
    if code:
        raise RuntimeError('ffmpeg could not write the finished dub')
