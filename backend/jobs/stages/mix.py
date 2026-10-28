"""Mix stage: place every voiced line on the timeline over the kept background."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.jobs.context import DubRun
from backend.jobs.control import CancelledError
from backend.media.audio_merge import assemble_timeline
from backend.media.separation import extract_full_audio
from backend.media.timeline import placement_notes

logger = logging.getLogger(__name__)


def mix_dub(dub: DubRun) -> Path:
    """Build `dubbed_audio.wav` (voices over the background); returns its path."""
    bed = _background_bed(dub)
    dub.mark(
        'mix',
        'Placing lines on the timeline',
        dub.span('mix', 0),
        segments=[piece.public() for piece in dub.translated],
        background_mode=dub.background_mode,
    )
    dub.add_notes(*dub.background_notes)
    dubbed_audio = assemble_timeline(
        dub.segments, dub.video_duration, dub.directory, bed, dub.background_mode
    )
    dub.timing_warnings = placement_notes(dub.segments)
    return dubbed_audio


def _background_bed(dub: DubRun) -> Path | None:
    """The audio under the voices: the separated background, the ducked original mix, or none.

    A failed separation falls back to ducking the original mix, with a note for the owner.
    """
    bed = None
    if dub.separation is not None:
        if not dub.separation.done():
            dub.mark(
                'mix',
                'Waiting for vocal separation',
                dub.span('mix', 0),
                segments=[piece.public() for piece in dub.translated],
            )
        try:
            bed = dub.separation.result()
        except CancelledError:
            raise
        except Exception as exc:
            dub.control.check()
            logger.warning('Vocal separation failed for %s: %s', dub.job_id, exc)
            dub.background_mode = 'duck'
            dub.background_notes.append(
                f'Vocal separation failed ({str(exc)[:80]}); the original mix was ducked instead.'
            )
            dub.save(background={'mode': 'duck', 'status': 'ready', 'progress': 1})
    if dub.background_mode == 'duck':
        bed = extract_full_audio(dub.video, dub.directory)
    return bed
