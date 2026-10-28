"""Script stages: transcribe every line with Bodhan ASR, translate it, and let the script editor polish it.

In streaming mode each line is translated and polished the moment it is heard, so the script fills
in line by line; otherwise translation runs as its own pass after transcription.
"""

from __future__ import annotations

import logging

from backend.config import settings
from backend.jobs.context import DubRun
from backend.jobs.control import run_segment_stage
from backend.media.video_input import cut_audio
from backend.models import LANGUAGES, Segment
from backend.providers import openrouter
from backend.providers.bodhan_api import BodhanRateLimitError
from backend.speech.asr import (
    DegenerateTranscript,
    degenerate,
    guess_source_language,
    script_mismatch,
    transcribe,
)
from backend.speech.fitting import DEFAULT_SPEAKING_RATE, SPEAKING_RATE
from backend.speech.llm_translation import translate_lines
from backend.speech.script_supervisor import check_transcripts, post_edit, profile_content
from backend.speech.translation import translate
from backend.voices.assignment import overlapping_turns, voice_reuse_warnings
from backend.voices.library import speaking_rate

logger = logging.getLogger(__name__)


def understand_content(dub: DubRun):
    """Settle the source language and the content profile (genre, tone) the editor and casting use."""
    if not dub.source_language:
        # the reviewer wrote the lines independently: Latin letters mean English speech, and Bodhan ASR needs that hint
        # (without it, English comes back as Telugu or Tamil letters spelling the sounds)
        dub.source_language = guess_source_language([s.reference_text for s in dub.segments])
        if dub.source_language:
            dub.save(source_language=dub.source_language)
            dub.job['source_language'] = dub.source_language
    dub.content = dub.job.get('script_profile') or {}
    if not dub.content and settings.script_supervision:
        heard = [
            Segment(s.start, s.end, s.speaker, text=s.reference_text)
            for s in dub.segments
            if s.reference_text
        ]
        # Without the reviewer's transcripts, the translate stage profiles from Bodhan's instead:
        # a profile from the title alone would stand in for one made from the dialogue.
        if heard:
            dub.content = profile_content(
                heard,
                dub.job.get('source_meta') or {},
                dub.job.get('title') or '',
                dub.source_language,
                dub.usage,
                target=LANGUAGES.get(dub.language, dub.language),
            )
            if dub.content:
                dub.save(script_profile=dub.content)


def transcribe_script(dub: DubRun):
    """Transcribe every line not already heard; in streaming mode also translate and polish it."""
    segments = dub.segments
    dub.mark(
        'transcribe',
        'Transcribing speech',
        dub.span('transcribe', 0),
        diarization_mode=dub.diarization_mode,
        speaker_count=len(dub.speakers),
        speaker_profiles=dub.profiles,
        speaker_voices=dub.speaker_voices,
        review_warnings=overlapping_turns(segments) + voice_reuse_warnings(dub.speaker_voices),
    )
    asr_state: dict = {}
    # Transcribe from the dialogue stem when there is one: the words without the music.
    dialogue = dub.directory / 'dialogue16k.wav'
    asr_audio = dialogue if dialogue.is_file() else dub.source_audio

    def transcribe_segment(index, segment):
        snippet = dub.directory / f"source_segment_{index:04d}.wav"
        cut_audio(asr_audio, segment.start, segment.end, snippet)
        try:
            segment.text = transcribe(
                dub.client, snippet, dub.source_language, dub.usage, asr_state
            )
            segment.text_source = ''
            if segment.reference_text and script_mismatch(segment.text, segment.reference_text):
                # English sounds spelled in an Indic script: the reviewer's words are the speech, Bodhan's are not
                logger.warning(
                    'Script mismatch at %.1fs in %s: %r vs %r; using the reviewer transcript',
                    segment.start,
                    dub.job_id,
                    segment.text[:30],
                    segment.reference_text[:30],
                )
                segment.asr_text, segment.text, segment.text_source = (
                    segment.text,
                    segment.reference_text,
                    'review',
                )
        except DegenerateTranscript as exc:
            # Bodhan looped on one token. The reviewer heard the same slot independently: use its words, and say so.
            if not segment.reference_text:
                raise RuntimeError(
                    f"Bodhan returned unusable text for the line at {segment.start:.1f}s and no second "
                    "transcript exists; try again later"
                ) from exc
            logger.warning(
                'Using the reviewer transcript for %s at %.1fs: Bodhan returned %r',
                dub.job_id,
                segment.start,
                str(exc)[:30],
            )
            segment.text, segment.text_source = segment.reference_text, 'review'
        finally:
            snippet.unlink(missing_ok=True)
        segment.state = 'heard'
        if dub.streaming and segment.text and not segment.translation:
            _translate_as_heard(dub, segment)

    remaining = [
        segment for segment in segments if not segment.text or degenerate(segment.text)
    ]  # junk cached text is redone
    cached_count = len(segments) - len(remaining)
    total = len(segments)

    def transcribed(count):
        done = cached_count + count
        dub.mark(
            'transcribe',
            f"{'Hearing and translating' if dub.streaming else 'Transcribing'} {done} of {total}",
            dub.span('transcribe', done / total),
            segments=[piece.public() for piece in segments if piece.text],
        )  # every line, so the script grows as it is heard

    if cached_count:
        transcribed(0)
    run_segment_stage(remaining, transcribe_segment, dub.workers, transcribed, dub.control)
    if remaining and settings.script_supervision and settings.transcript_check:
        dub.mark('transcribe', 'Checking the transcript', dub.span('transcribe', 1))
        check_transcripts(segments, dub.source_language, dub.usage, only=remaining)
    dub.save(
        asr_engine_used=asr_state.get('engine') or dub.job.get('asr_engine_used') or 'bodhan',
        usage=dub.usage.as_dict(),
    )
    dub.spoken = [segment for segment in segments if segment.text]
    if not dub.spoken:
        raise RuntimeError("Bodhan ASR returned no speech for this video")


def _translate_as_heard(dub: DubRun, segment: Segment):
    """Streaming: translate one line just heard, then let the editor polish it with the lines so far as context."""
    segment.translation = translate(
        dub.client, segment.text, dub.language, dub.script, dub.source_language, dub.usage
    )
    segment.state = 'translated'
    if settings.script_supervision and segment.translation:
        # the lines up to this one (by position in the script: a resumed run transcribes only some)
        position = next(i for i, piece in enumerate(dub.segments) if piece is segment)
        so_far = [
            piece for piece in dub.segments[: position + 1] if piece.text and piece.translation
        ]
        post_edit(
            so_far,
            dub.language,
            dub.source_language,
            dub.script,
            dub.usage,
            only=[segment],
            content=dub.content,
            targets={
                len(so_far) - 1: length_targets([segment], dub.speaker_voices, dub.language)[0]
            },
        )


def translate_script(dub: DubRun):
    """Translate every heard line not yet translated (all of them unless streaming), then edit the new ones."""
    spoken = dub.spoken
    dub.mark(
        'translate',
        'Translating speech',
        dub.span('translate', 0),
        segments=[piece.public() for piece in spoken],
    )

    def translate_segment(index, segment):
        segment.translation = translate(
            dub.client, segment.text, dub.language, dub.script, dub.source_language, dub.usage
        )
        segment.state = 'translated'

    untranslated = [segment for segment in spoken if not segment.translation]
    translated_count = len(spoken) - len(untranslated)

    def translated_progress(count):
        done = translated_count + count
        dub.mark(
            'translate',
            f'Translating {done} of {len(spoken)}',
            dub.span('translate', done / len(spoken)),
            segments=[piece.public() for piece in spoken],
        )

    if translated_count:
        translated_progress(0)
    translation_engine = 'bodhan'
    try:
        run_segment_stage(
            untranslated, translate_segment, dub.workers, translated_progress, dub.control
        )
    except BodhanRateLimitError as exc:
        pending = [segment for segment in untranslated if not segment.translation]
        if not openrouter.covers_bodhan() or not pending:
            raise
        logger.warning(
            'Bodhan translation unavailable for %s (%s); OpenRouter translates the rest',
            dub.job_id,
            exc,
        )
        _translate_with_openrouter(dub, pending, translated_count)
        translation_engine = 'openrouter'
        translated_progress(len(pending))
    if untranslated and settings.script_supervision:
        if not dub.content:
            dub.mark('translate', 'Reading the content', dub.span('translate', 0.95))
            dub.content = profile_content(
                spoken,
                dub.job.get('source_meta') or {},
                dub.job.get('title') or '',
                dub.source_language,
                dub.usage,
                target=LANGUAGES.get(dub.language, dub.language),
            )
            if dub.content:
                dub.save(script_profile=dub.content)
        dub.mark('translate', 'Editing the translation', dub.span('translate', 1))
        post_edit(
            spoken,
            dub.language,
            dub.source_language,
            dub.script,
            dub.usage,
            only=untranslated,
            content=dub.content,
            targets=length_targets(spoken, dub.speaker_voices, dub.language),
        )
    dub.save(translation_engine_used=translation_engine, usage=dub.usage.as_dict())
    dub.translated = [segment for segment in spoken if segment.translation]
    if not dub.translated:
        raise RuntimeError("Bodhan translation returned no text")


def _translate_with_openrouter(dub: DubRun, pending: list[Segment], translated_count: int):
    """When Bodhan translation is rate-limited out, translate the remaining lines scene by scene with Gemini."""
    lines = [
        {
            'id': index,
            'speaker': segment.speaker,
            'text': segment.text,
            'seconds': segment.end - segment.start,
        }
        for index, segment in enumerate(pending)
    ]
    total_spoken = len(dub.spoken)
    results, _ = translate_lines(
        lines,
        dub.language,
        dub.profiles,
        lambda done, total: dub.mark(
            'translate',
            f'Translating scene {done} of {total}',
            dub.span('translate', (translated_count + len(pending) * done / total) / total_spoken),
        ),
        script=dub.script,
        usage=dub.usage,
    )
    for index, segment in enumerate(pending):
        segment.translation = results[index]


def length_targets(lines, speaker_voices: dict, language: str) -> dict:
    """{index: characters} each line should have to fill its slot at its cast voice's pace (the editor writes to it)."""
    out = {}
    for index, segment in enumerate(lines):
        voice = (speaker_voices or {}).get(segment.speaker, "")
        rate = (
            speaking_rate(voice, SPEAKING_RATE.get(language, DEFAULT_SPEAKING_RATE))
            if voice
            else SPEAKING_RATE.get(language, DEFAULT_SPEAKING_RATE)
        )
        out[index] = max(4, round(rate * max(0.05, segment.end - segment.start)))
    return out
