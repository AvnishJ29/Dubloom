"""Speaker stage: who speaks when, who each speaker is, and which studio voice plays them.

Diarization runs in an isolated worker (DiariZen, or the sherpa-onnx fallback). For DiariZen, Gemini
then re-hears the turns (`speaker_review`), the turns are refined into dubbing slots, and gender
and voice profiles are attached. The result is cached in `speaker_turns.json`.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from backend.config import ROOT, settings
from backend.jobs.context import DubRun
from backend.jobs.control import CancelledError, heavy_stage
from backend.jobs.gpus import pool
from backend.jobs.stages.background import separate_in_background
from backend.media.separation import separation_available
from backend.media.video_input import extract_audio
from backend.models import Segment
from backend.rendering.lip_sync import speaker_face_evidence
from backend.speakers.diarization import refine_turns, speaker_turns
from backend.speakers.gender import classify_genders, decide_gender, split_mixed_speakers
from backend.speakers.speaker_profiles import speaker_profiles
from backend.speakers.speaker_review import review_speakers
from backend.voices.assignment import assign_voices

logger = logging.getLogger(__name__)

TURNS_VERSION = 6  # bump when diarization post-processing changes, so resumed jobs re-run it
FACE_EVIDENCE_WAIT = 300  # seconds to wait for the background face pass before voicing
# What the script view shows must survive a resume: a re-render rebuilds the lines from the speaker turns.
KEPT_ON_RESUME = (
    'machine_translation',
    'asr_text',
    'reference_text',
    'text_source',
    'style_override',
    'take',
    'source_edited',
    'condensed',
    'ai_edited',
    'history',
)


def find_speakers(dub: DubRun):
    """Find who speaks when (or reuse the saved turns) and rebuild the script lines from the turns."""
    dub.mark('speakers', 'Finding speakers', dub.span('speakers', 0))
    result = _saved_turns(dub)
    if result is None:
        dialogue = _dialogue_audio(dub)
        with heavy_stage(dub.control, dub.job_id, 'Waiting for a free analysis slot'):
            dub.control.check()
            result = _diarize(dub, dialogue)
            _describe_speakers(dub, result, dialogue)
            dub.turns_path.write_text(json.dumps(result), encoding='utf-8')
    dub.turns = result
    dub.segments = _segments(result['segments'])
    dub.diarization_mode = result['mode']
    dub.profiles = result.get('profiles') or {}
    if not dub.segments:
        raise RuntimeError("No speech was detected in the source video")
    dub.cached_segments = {
        (item['start'], item['end'], item['speaker']): item
        for item in (dub.previous or {}).get('segments', [])
    }
    for segment in dub.segments:
        restore_line(
            segment, dub.cached_segments.get((segment.start, segment.end, segment.speaker), {})
        )
    dub.speakers = list(dict.fromkeys(segment.speaker for segment in dub.segments))


def cast_voices(dub: DubRun):
    """Pick a studio voice for every speaker from their gender, age, the genre and the owner's choices."""
    dub.voice_reasons = {}
    dub.speaker_voices = assign_voices(
        dub.speakers,
        dub.language,
        dub.voice,
        dub.profiles,
        dub.overrides,
        genre=dub.content.get('genre'),
        reasons=dub.voice_reasons,
        preferred_voices=dub.job.get('voices') or None,
    )
    dub.save(voice_reasons=dub.voice_reasons)


def match_faces_to_voices(dub: DubRun):
    """Let the faces on screen correct the speakers: a person seen settles their gender, and one face
    speaking both halves of a voice split undoes the split. Voices are re-cast when anything changed.
    """
    evidence = _face_evidence(dub)
    changed = _undo_false_splits(dub, evidence)
    changed = _settle_genders(dub, evidence) or changed
    if changed:
        dub.speakers = list(dict.fromkeys(segment.speaker for segment in dub.segments))
        cast_voices(dub)
        dub.turns['segments'] = [s.public() for s in dub.segments]
        dub.turns['profiles'] = dub.profiles
        dub.turns_path.write_text(json.dumps(dub.turns), encoding='utf-8')
        dub.save(
            speaker_profiles=dub.profiles,
            speaker_voices=dub.speaker_voices,
            voice_reasons=dub.voice_reasons,
            speaker_count=len(dub.speakers),
            segments=[piece.public() for piece in dub.segments],
        )
    elif evidence:
        dub.save(speaker_profiles=dub.profiles)


def restore_line(segment: Segment, saved: dict) -> Segment:
    """A line rebuilt from the speaker turns gets back what an earlier run saved for it: its words, voice, edits and history."""
    segment.text = saved.get('text') or ''
    segment.translation = saved.get('translation') or ''
    segment.voice = saved.get('voice') or ''
    segment.voiced_text = saved.get('voiced_text') or ''
    segment.voiced_style = saved.get('voiced_style') or ''
    segment.tts_engine = saved.get('tts_engine') or ''
    segment.edited = bool(saved.get('edited'))
    for key in KEPT_ON_RESUME:
        if saved.get(key):
            setattr(segment, key, saved[key])
    return segment


def pitch_reader(audio: Path):
    """A function (start, end) -> median pitch in Hz over the original mix, or None when unvoiced."""
    import numpy as np
    from backend.speakers.speaker_profiles import median_pitch

    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(audio), '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768

    def pitch(start: float, end: float):
        return median_pitch(samples[int(start * 16000) : int(end * 16000)])[0]

    return pitch


def _segments(items: list[dict]) -> list[Segment]:
    return [
        Segment(
            **{key: value for key, value in item.items() if key in Segment.__dataclass_fields__}
        )
        for item in items
    ]


def _saved_turns(dub: DubRun) -> dict | None:
    """The turns an earlier run found, when it used the same speaker count and post-processing."""
    if not (dub.resume and dub.turns_path.is_file()):
        return None
    try:
        result = json.loads(dub.turns_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get('num_speakers') != dub.num_speakers or result.get('version') != TURNS_VERSION:
        return None
    return result


def _dialogue_audio(dub: DubRun) -> Path:
    """The audio to diarize: the dialogue stem when separation is available, because music and effects
    confuse diarization in films and trailers; otherwise the full mix."""
    if not (
        settings.diarize_dialogue_stem
        and separation_available()
        and dub.video_duration <= settings.separation_max_minutes * 60
    ):
        return dub.source_audio
    dub.mark('speakers', 'Separating dialogue from music', dub.span('speakers', 0))
    try:
        separate_in_background(dub.job_id, dub.directory, dub.video, dub.control)
        if (dub.directory / 'vocals.wav').is_file():
            dialogue = dub.directory / 'dialogue16k.wav'
            extract_audio(dub.directory / 'vocals.wav', dialogue)
            return dialogue
    except CancelledError:
        raise
    except Exception as exc:
        dub.control.check()
        logger.warning(
            'Dialogue separation failed for %s (%s); diarizing the full mix', dub.job_id, exc
        )
    return dub.source_audio


def _diarize(dub: DubRun, dialogue: Path) -> dict:
    """Speaker turns as {'mode', 'segments', ...}: the diarization worker, then the speaker review."""
    raw_path = dub.directory / 'speaker_turns_raw.json'
    command = _diarization_command(dub, dialogue, raw_path)
    if not command:
        segments, mode = speaker_turns(dub.source_audio, num_speakers=dub.num_speakers)
        return {'mode': mode, 'segments': [segment.public() for segment in segments]}
    _run_diarization_worker(dub, command)
    result = json.loads(raw_path.read_text(encoding='utf-8'))
    raw_turns = [Segment(t['start'], t['end'], t['speaker']) for t in result['segments']]
    if dialogue != dub.source_audio:
        result['mode'] += ' on the dialogue stem'
    if 'DiariZen' in result.get('mode', ''):
        reviewed = None
        if settings.speaker_review and raw_turns:
            dub.mark('speakers', 'Checking who says each line', dub.span('speakers', 0.9))
            reviewed = review_speakers(
                raw_turns, dialogue, dub.directory / 'review', dub.source_language, dub.usage
            )
        if reviewed:
            raw_turns, result['review'] = reviewed
            result['mode'] += ' + speaker review'
        slots = refine_turns(raw_turns, dialogue)
        _carry_styles(slots, raw_turns)
        result['segments'] = [t.public() for t in slots]
    return result


def _diarization_command(dub: DubRun, dialogue: Path, raw_path: Path) -> list[str] | None:
    """The worker command for the configured backend; None means the in-process fallback."""
    if settings.diarization_backend == 'diarizen' and settings.diarizen_python:
        return [
            settings.diarizen_python,
            '-u',
            str(ROOT / 'backend' / 'workers' / 'diarizen_worker.py'),
            str(dialogue),
            str(raw_path),
            '--model',
            str(settings.diarizen_model),
            '--embedding',
            str(settings.diarizen_embedding),
            '--speakers',
            str(dub.num_speakers or 0),
            '--device',
            settings.device,
        ]
    if settings.diarization_backend in {'sherpa', 'diarizen'}:
        return [
            sys.executable,
            '-u',
            '-m',
            'backend.workers.diarization_worker',
            str(dub.source_audio),
            str(raw_path),
            str(dub.num_speakers or 0),
        ]
    return None


def _run_diarization_worker(dub: DubRun, command: list[str]):
    """Run the worker on this slot's GPU, relaying its `PROGRESS done total` lines to the job."""
    with subprocess.Popen(
        command,
        cwd=settings.jobs_dir.parent,
        env=pool.env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        dub.control.register(process)
        tail = []
        try:
            for line in process.stdout:
                tail = (tail + [line])[-20:]
                if line.startswith('PROGRESS '):
                    _, done, total = line.split()
                    if int(total):
                        dub.mark(
                            'speakers',
                            f'Finding speakers {done}/{total}',
                            dub.span('speakers', int(done) / int(total)),
                        )
            code = process.wait()
        finally:
            dub.control.unregister(process)
        dub.control.check()
        if code:
            raise RuntimeError('Diarization worker failed: ' + ''.join(tail)[-1000:])


def _describe_speakers(dub: DubRun, result: dict, dialogue: Path):
    """Attach gender, voice profile and name to each speaker, splitting a voice that holds two people."""
    turns = _segments(result['segments'])
    try:
        classified = classify_genders(dub.source_audio, turns, dub.directory, dub.control)
        heard_genders = {
            s: (v or {}).get('gender')
            for s, v in (((result.get('review') or {}).get('voices')) or {}).items()
        }
        split = split_mixed_speakers(
            turns, classified, heard_genders, pitch_of=pitch_reader(dub.source_audio)
        )
        if split:
            logger.info('Split mixed speakers for %s: %s', dub.job_id, split)
            result['segments'] = [t.public() for t in turns]
            result['split_speakers'] = split
    except Exception:
        logger.exception('Gender classification failed for %s', dub.job_id)
        classified = {}
    try:
        result['profiles'] = speaker_profiles(dialogue, turns)
    except Exception:
        logger.exception('Speaker profiling failed for %s', dub.job_id)
        result['profiles'] = {}
    reviewed_voices = (result.get('review') or {}).get('voices') or {}
    for new_id, old_id in (result.get('split_speakers') or {}).items():
        gender = (classified.get(new_id) or {}).get('gender')
        result['profiles'].setdefault(new_id, {})['name'] = (
            f"Second voice ({gender})" if gender else "Second voice"
        )
        result['profiles'][new_id]['split_from'] = old_id
    for speaker, profile in result['profiles'].items():
        heard = (reviewed_voices.get(speaker) or {}).get('gender')
        profile.update(decide_gender(profile, classified.get(speaker), heard))
    for speaker, voice_info in reviewed_voices.items():
        if voice_info.get('name'):
            result['profiles'].setdefault(speaker, {})['name'] = voice_info['name']
    result['num_speakers'] = dub.num_speakers
    result['version'] = TURNS_VERSION


def _carry_styles(slots: list[Segment], turns: list[Segment]):
    """Give every final slot the delivery style of the reviewed turn it overlaps most, and the review
    transcripts of all the turns it covers (the second opinion for the transcript check)."""
    for slot in slots:
        best, best_overlap, texts = None, 0.0, []
        for turn in sorted(turns, key=lambda t: t.start):
            overlap = min(slot.end, turn.end) - max(slot.start, turn.start)
            if turn.speaker != slot.speaker or overlap <= 0:
                continue
            if turn.reference_text:
                texts.append(turn.reference_text)
            if overlap > best_overlap:
                best, best_overlap = turn, overlap
        if best is not None and best.style:
            slot.style = best.style
        slot.reference_text = " ".join(texts)


def _face_evidence(dub: DubRun) -> dict:
    """Per speaker, the face seen while they talk (identity, apparent gender); {} when unavailable."""
    if dub.faces_ahead is not None and not dub.faces_ahead.done():
        dub.mark('voice', 'Checking faces against voices', dub.span('voice', 0))
    try:
        if dub.faces_ahead is not None:
            dub.faces_ahead.result(timeout=FACE_EVIDENCE_WAIT)
        return speaker_face_evidence(
            dub.video,
            dub.directory,
            [(s.start, s.end, s.speaker) for s in dub.translated],
            dub.control,
        )
    except CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001  (faces are an improvement, never a requirement)
        logger.warning(
            'Face evidence unavailable for %s: %s',
            dub.job_id,
            str(exc)
            or f'{type(exc).__name__}: the face analysis took longer than {FACE_EVIDENCE_WAIT} s; voices kept as cast',
        )
        return {}


def _undo_false_splits(dub: DubRun, evidence: dict) -> bool:
    """Merge a split-off voice back into its origin when one face speaks both."""
    changed = False
    profiles = dub.profiles
    for split_id, origin in list((dub.turns.get('split_speakers') or {}).items()):
        identity = evidence.get(split_id, {}).get('identity')
        same_face = identity is not None and identity == evidence.get(origin, {}).get('identity')
        if same_face and split_id in profiles:
            logger.info(
                'Merging %s back into %s for %s: one face speaks both', split_id, origin, dub.job_id
            )
            for segment in dub.segments:
                if segment.speaker == split_id:
                    segment.speaker = origin
            merged = profiles.pop(split_id, {})
            profiles[origin]['speech_seconds'] = round(
                (profiles[origin].get('speech_seconds') or 0) + (merged.get('speech_seconds') or 0),
                2,
            )
            profiles[origin]['turns'] = (profiles[origin].get('turns') or 0) + (
                merged.get('turns') or 0
            )
            dub.turns['split_speakers'].pop(split_id, None)
            changed = True
    return changed


def _settle_genders(dub: DubRun, evidence: dict) -> bool:
    """Weigh the face's apparent gender with the voice's; True when any speaker's gender changed."""
    changed = False
    reviewed_voices = (dub.turns.get('review') or {}).get('voices') or {}
    for speaker, profile in dub.profiles.items():
        seen = evidence.get(speaker)
        if seen and seen.get('face_gender'):
            before = profile.get('gender')
            heard = reviewed_voices.get(speaker, {}).get('gender')
            classified_turns = (
                profile.get('gender_votes')
                and {
                    'gender': profile['gender'],
                    'confidence': profile.get('gender_confidence', 0),
                }
            ) or None
            profile.update(decide_gender(profile, classified_turns, heard, seen))
            profile['face_gender'] = seen['face_gender']
            changed = changed or profile.get('gender') != before
    return changed
