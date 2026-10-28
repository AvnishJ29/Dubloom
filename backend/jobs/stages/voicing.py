"""Voice stage: speak every translated line with Bodhan Indic-Speak and fit it to its original timing.

Unchanged lines reuse their saved clips (voicing is paid). New lines are first rewritten to their slot
length (`prefit_lines`, cheap text work), then voiced, then any line still too long or short is
rewritten and re-voiced (`fit_lines`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.config import settings
from backend.jobs.context import DubRun
from backend.jobs.control import CancelledError, run_segment_stage
from backend.media.separation import extract_full_audio
from backend.providers import openrouter
from backend.speech.fitting import fit_lines, prefit_lines
from backend.speech.openrouter_tts import OPENROUTER_LANGUAGES
from backend.speech.styles import normalise_style
from backend.voices.engine import VoiceState, voice_references

logger = logging.getLogger(__name__)


def voice_script(dub: DubRun):
    """Voice every translated line that has no matching saved clip, then fit all lines to their slots."""
    translated = dub.translated
    dub.mark(
        'voice',
        'Generating voices',
        dub.span('voice', 0),
        segments=[piece.public() for piece in translated],
    )
    # Assign voices before concurrent requests so completion order never
    # changes a speaker's voice or the final timeline order.
    for segment in translated:
        segment.voice = dub.speaker_voices[segment.speaker]
    voice_mode = (
        'clone'
        if dub.job.get('tts_engine') == 'clone'
        and openrouter.configured()
        and dub.language in OPENROUTER_LANGUAGES
        else 'auto'
    )
    references = _clone_references(dub) if voice_mode == 'clone' else {}
    voices = VoiceState(
        dub.client, dub.language, dub.speaker_voices, voice_mode, references, dub.usage
    )

    def voice_segment(_, task):
        index, segment = task
        speech = dub.directory / f"tts_{index:04d}.wav"
        segment.tts_engine = voices.render(
            segment.speaker, segment.translation, speech, style_of(dub, segment)
        )
        segment.audio_path = str(speech)
        segment.voiced_text = segment.translation
        segment.voiced_style = style_of(dub, segment)
        segment.state = 'voiced'

    previous_engine = (dub.previous or {}).get('tts_engine_used') or ''
    unvoiced = _reuse_saved_clips(dub, voices, voice_mode, previous_engine)
    # Size new lines to their slots before voicing them (text rewrites are cheap, re-voicing is paid).
    if unvoiced and settings.timing_fit:
        dub.mark('voice', 'Fitting lines to their timing', dub.span('voice', 0))
        prefit_lines(
            [segment for _, segment in unvoiced],
            dub.language,
            dub.usage,
            dub.source_language,
            context=translated,
        )
    voiced_count = len(translated) - len(unvoiced)

    def voiced_progress(count):
        done = voiced_count + count
        dub.mark(
            'voice',
            f'Voicing {done} of {len(translated)}',
            dub.span('voice', 0.8 * done / len(translated)),
            segments=[piece.public() for piece in translated],
        )

    if voiced_count:
        voiced_progress(0)
    run_segment_stage(unvoiced, voice_segment, dub.workers, voiced_progress, dub.control)
    tts_used = voices.engine or previous_engine or 'bodhan'
    # One engine per project: if Bodhan dropped out mid-way, re-voice the other lines with the engine in use.
    mixed = [
        (index, segment)
        for index, segment in enumerate(translated)
        if segment.tts_engine and segment.tts_engine != tts_used
    ]
    if mixed:
        dub.mark(
            'voice',
            f'Re-voicing {len(mixed)} lines for a consistent voice',
            dub.span('voice', 0.82),
        )
        run_segment_stage(mixed, voice_segment, dub.workers, lambda count: None, dub.control)
    fit = _fit_to_timing(dub, voices)
    dub.save(
        tts_engine_used=tts_used,
        speaker_engine_voices=voices.labels(),
        fit_rounds=fit.rounds,
        fit_summary={
            'in_band': fit.in_band,
            'total': fit.total,
            'rewritten': fit.rewritten,
            'band': [settings.fit_min_speed, settings.fit_max_speed],
        },
        usage=dub.usage.as_dict(),
    )


def style_of(dub: DubRun, segment) -> str:
    """The delivery style for a line: the owner's direction for it, then the speaker's or the
    project's style, then the emotion the reviewer heard in the original."""
    if segment.style_override:
        return normalise_style(segment.style_override)
    if segment.speaker in dub.speaker_styles:
        return dub.speaker_styles[segment.speaker]
    return dub.delivery_style if dub.delivery_style != 'neutral' else normalise_style(segment.style)


def _clone_references(dub: DubRun) -> dict:
    """Voice samples per speaker for cloning, cut from the clean vocals when separation finishes."""
    clean = None
    if dub.separation is not None:
        dub.mark('voice', 'Waiting for clean voice samples', dub.span('voice', 0))
        try:
            dub.separation.result()
            clean = (
                dub.directory / 'vocals.wav' if (dub.directory / 'vocals.wav').is_file() else None
            )
        except CancelledError:
            raise
        except Exception as exc:
            logger.warning('Separation failed before cloning for %s: %s', dub.job_id, exc)
    return voice_references(
        dub.directory,
        dub.segments,
        dub.speakers,
        clean or extract_full_audio(dub.video, dub.directory),
    )


def _reuse_saved_clips(dub: DubRun, voices: VoiceState, voice_mode: str, previous_engine: str):
    """Keep each saved clip whose words, voice, style and engine still match; return the lines to voice
    as (index, segment) pairs."""
    unvoiced = []
    for index, segment in enumerate(dub.translated):
        speech = dub.directory / f"tts_{index:04d}.wav"
        saved = dub.cached_segments.get((segment.start, segment.end, segment.speaker), {})
        voiced_text = saved.get('voiced_text') or saved.get('translation')
        cached_engine = saved.get('tts_engine') or previous_engine
        mode_ok = (cached_engine == 'clone') == (voice_mode == 'clone')
        if (
            dub.resume
            and voiced_text == segment.translation
            and saved.get('voice') == segment.voice
            and (saved.get('voiced_style') or 'neutral') == style_of(dub, segment)
            and mode_ok
            and speech.is_file()
            and speech.stat().st_size > 44
        ):
            segment.audio_path = str(speech)
            segment.voiced_text = segment.translation
            segment.voiced_style = style_of(dub, segment)
            segment.tts_engine = cached_engine or 'bodhan'
            segment.state = 'voiced'
            if voices.engine is None and cached_engine:
                voices.engine = cached_engine
        else:
            unvoiced.append((index, segment))
    return unvoiced


def _fit_to_timing(dub: DubRun, voices: VoiceState):
    """Fit every line to its original slot (0.9x - 1.1x), rewriting and re-voicing as needed."""

    def render_line(segment):
        segment.tts_engine = voices.render(
            segment.speaker, segment.translation, Path(segment.audio_path), style_of(dub, segment)
        )
        segment.voiced_text = segment.translation

    def fit_progress(round_index, count):
        dub.mark(
            'voice',
            f'Fitting {count} lines to their timing (round {round_index + 1})',
            dub.span('voice', 0.86 + 0.04 * round_index),
        )

    return fit_lines(
        dub.translated,
        dub.directory,
        dub.language,
        render_line,
        dub.workers,
        run_segment_stage,
        dub.control,
        fit_progress,
        dub.usage,
        dub.source_language,
    )
