"""Source stage: fetch or reuse the video, normalise it, and extract its audio."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.config import settings
from backend.jobs.context import DubRun
from backend.jobs.store import store
from backend.media.video_input import (
    SourceError,
    download_video,
    duration,
    extract_audio,
    make_poster,
    prepare_source,
    probe,
    run,
)

logger = logging.getLogger(__name__)

SECTION_SOURCE_MAX_SECONDS = 2 * 3600  # a section dub may come from a video up to this long


def fetch_source(dub: DubRun):
    """Prepare `source.mp4`, enforce the length limit, save a poster frame and extract the audio."""
    dub.mark('source', 'Fetching source', dub.span('source', 0))
    video = dub.video = acquire_source(
        dub.job,
        dub.directory,
        lambda label, fraction: dub.mark('source', label, dub.span('source', fraction)),
    )
    info = probe(video)
    dub.video_duration = info['duration']
    if dub.video_duration > settings.max_video_minutes * 60:
        raise RuntimeError(f"Video exceeds the {settings.max_video_minutes}-minute limit")
    poster = dub.directory / 'poster.jpg'
    if not poster.is_file():
        try:
            make_poster(video, poster, min(1.0, dub.video_duration / 3))
        except RuntimeError:
            logger.warning('Poster frame failed for %s', dub.job_id)
    dub.mark(
        'audio',
        'Extracting source audio',
        dub.span('audio', 0),
        duration=round(dub.video_duration, 2),
        width=info['width'],
        height=info['height'],
        source_url_ready=True,
        poster_url=f'/api/jobs/{dub.job_id}/poster' if poster.is_file() else None,
        source_video_url=f'/api/jobs/{dub.job_id}/source',
    )
    dub.source_audio = dub.directory / "source_audio.wav"
    if not (dub.resume and dub.source_audio.is_file()):
        extract_audio(video, dub.source_audio)


def acquire_source(job: dict, directory: Path, report) -> Path:
    """Reuse a saved source, normalise an upload, or download from the URL.

    `report(label, fraction)` shows progress. A clip range keeps only that part of the video.
    """
    saved = directory / 'source.mp4'
    if saved.is_file():
        return saved
    clip = (job.get('clip_start'), job.get('clip_end')) if job.get('clip_end') else None
    max_seconds = settings.max_video_minutes * 60
    if clip:
        # the limit applies to the part; the whole video may be longer
        max_seconds = max(max_seconds, SECTION_SOURCE_MAX_SECONDS)
    info = None
    if job.get('source_type') == 'upload':
        uploads = sorted(directory.glob('upload.*'))
        if not uploads:
            raise SourceError('The uploaded file is no longer available; create the project again')
        raw = uploads[0]
        report('Preparing uploaded video', 0.4)
    else:
        info = download_video(
            job['source_url'],
            directory,
            max_seconds,
            settings.max_upload_mb * 1024 * 1024,
            lambda fraction: report(f'Downloading video {int(fraction * 100)}%', fraction * 0.7),
            section=(float(clip[0]), float(clip[1])) if clip else None,
        )
        raw = info.path
        if info.title and not job.get('title'):
            title = info.title[:120]
            if clip:
                title += f" ({_clock(clip[0])}–{_clock(clip[1])})"
            store.save(job['id'], title=title)
        if info.meta:
            store.save(job['id'], source_meta=info.meta)
        report('Preparing video', 0.75)
    prepare_source(raw, saved, settings.max_video_height)
    if clip and not (info is not None and info.sectioned):  # a range download holds only the part
        report('Cutting the part to dub', 0.9)
        cut_section(saved, clip[0], clip[1])
    return saved


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def cut_section(video: Path, start: float, end: float):
    """Keep only [start, end] of the prepared source, re-encoded so the cut lands exactly on the frame."""
    total = duration(video)
    end = min(end, total)
    if end - start < 1:
        raise SourceError(
            f"The part to dub ({_clock(start)}–{_clock(end)}) lies outside the video ({_clock(total)} long)"
        )
    cut = video.with_name('source_cut.mp4')
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-ss',
        f'{start:.3f}',
        '-i',
        video,
        '-t',
        f'{end - start:.3f}',
        '-c:v',
        'libx264',
        '-preset',
        'faster',
        '-crf',
        '16',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-movflags',
        '+faststart',
        cut,
    )
    cut.replace(video)
