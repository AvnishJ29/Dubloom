"""Speaker gender for voice casting: a dedicated audio classifier first, the reviewer's ear and the pitch
estimate as tie-breakers. Only the cast voice depends on it; speaker labels never change."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from backend.config import ROOT, settings
from backend.rendering.video_tools import file_digest, run_worker

logger = logging.getLogger(__name__)

CLASSIFIER_MIN = 0.6  # below this the classifier abstains and the other two signals decide
CLASSIFIER_WEIGHT = 3.0  # a confident classifier outweighs reviewer (0.9) + pitch (<= 1.0) together
REVIEWER_WEIGHT = 0.9


SPLIT_SURE = 0.6  # a turn at least this sure votes for its own gender; less sure turns follow their nearest sure neighbour
SPLIT_PEAK = 0.8  # each side needs one turn this sure...
SPLIT_MIN_SECONDS = 1.0  # ... and this much speech before a cluster is treated as two people
SPLIT_PITCH_RATIO = 1.2  # ... and the female side must sit at least this much higher in pitch: one person keeps one register


def classify_genders(audio: Path, turns, directory: Path, control=None) -> dict[str, dict]:
    """{speaker: {female, male, child, gender, confidence, seconds, turns: [...]}} from the classifier.
    Judge on the original mix: the separated vocal stem distorts timbre enough to flip verdicts."""
    if not settings.gender_model:
        return {}
    spans: dict[str, list] = {}
    for turn in turns:
        spans.setdefault(turn.speaker, []).append([round(turn.start, 3), round(turn.end, 3)])
    turns_path = directory / 'gender_turns.json'
    turns_path.write_text(json.dumps(spans, sort_keys=True), encoding='utf-8')
    output, stamp = directory / 'gender.json', directory / 'gender.sha256'
    digest = hashlib.sha256(
        f"{file_digest(audio)}:{turns_path.read_text()}:{settings.gender_model}".encode()
    ).hexdigest()
    if not (output.is_file() and stamp.is_file() and stamp.read_text() == digest):
        command = [
            settings.latentsync_python,
            '-u',
            str(ROOT / 'backend' / 'workers' / 'gender_worker.py'),
            str(audio),
            str(turns_path),
            str(output),
            '--model',
            settings.gender_model,
            '--device',
            settings.device,
        ]
        run_worker(command, ROOT, control, lambda line: None, label='Gender classifier')
        stamp.write_text(digest)
    result = json.loads(output.read_text(encoding='utf-8'))
    for (
        verdict
    ) in result.values():  # the pooled verdict is replaced by the duration-weighted turn vote
        gender, confidence = _vote(verdict.get('turns') or [])
        if gender:
            verdict['gender'], verdict['confidence'] = gender, confidence
    return result


def _vote(turns: list[dict]) -> tuple[str | None, float]:
    """The gender with the most confidence-weighted seconds over the turns, and its share of the total."""
    seconds = {'female': 0.0, 'male': 0.0}
    for turn in turns:
        seconds[turn['gender']] += (turn['end'] - turn['start']) * turn['confidence']
    total = sum(seconds.values())
    if not total:
        return None, 0.0
    gender = max(seconds, key=seconds.get)
    return gender, round(seconds[gender] / total, 4)


def split_mixed_speakers(
    turns, classified: dict[str, dict], heard: dict[str, str] | None = None, pitch_of=None
) -> dict[str, str]:
    """Diarization sometimes merges two people. When a speaker's turns consistently split by gender (each side
    with a sure turn and at least a second of speech), the side with less speech becomes a new speaker '<id>b'.
    Turns that are not sure follow the nearest sure turn in time, since one person tends to speak in stretches.
    `heard`: the reviewer's gender per speaker; that side keeps the original id and name when it exists.
    `pitch_of(start, end)`: median pitch of a span; when given, a split also needs the female side to sit
    SPLIT_PITCH_RATIO above the male side, since a classifier that wavers on one ambiguous voice gives
    both "sides" the same pitch. Mutates the turns' speaker ids; returns {new: old}."""
    created = {}
    for speaker, verdict in list(classified.items()):
        votes = sorted(verdict.get('turns') or [], key=lambda t: t['start'])
        if len(votes) < 2:
            continue
        sure = [t for t in votes if t['confidence'] >= SPLIT_SURE]
        side = {
            'female': [t for t in sure if t['gender'] == 'female'],
            'male': [t for t in sure if t['gender'] == 'male'],
        }
        seconds = {g: sum(t['end'] - t['start'] for t in ts) for g, ts in side.items()}
        if any(
            seconds[g] < SPLIT_MIN_SECONDS
            or not any(t['confidence'] >= SPLIT_PEAK for t in side[g])
            for g in side
        ):
            continue
        if pitch_of is not None:
            pitches = {
                g: sorted(p for p in (pitch_of(t['start'], t['end']) for t in side[g]) if p)
                for g in side
            }
            if pitches['female'] and pitches['male']:
                median = lambda xs: xs[len(xs) // 2]
                if median(pitches['female']) < SPLIT_PITCH_RATIO * median(pitches['male']):
                    logger.info(
                        'Not splitting %s: both sides share one register (%d vs %d Hz)',
                        speaker,
                        median(pitches['female']),
                        median(pitches['male']),
                    )
                    verdict['register_note'] = 'one voice with an ambiguous register'
                    continue
        for turn in votes:  # unsure turns follow their nearest sure neighbour
            if turn['confidence'] < SPLIT_SURE:
                nearest = min(
                    sure,
                    key=lambda t: min(abs(t['start'] - turn['end']), abs(turn['start'] - t['end'])),
                )
                turn['gender'] = nearest['gender']
        for turn in votes:
            seconds_all = {
                g: sum(t['end'] - t['start'] for t in votes if t['gender'] == g) for g in side
            }
        minority = min(seconds_all, key=seconds_all.get)
        if (heard or {}).get(
            speaker
        ) in side:  # the reviewer named this voice: keep its id for that gender
            minority = 'male' if heard[speaker] == 'female' else 'female'
        moved = [t for t in votes if t['gender'] == minority]
        new_id = f"{speaker}b"
        for turn in turns:
            if turn.speaker == speaker and any(
                abs(turn.start - t['start']) < 0.05 and abs(turn.end - t['end']) < 0.05
                for t in moved
            ):
                turn.speaker = new_id
        kept = [t for t in votes if t['gender'] != minority]
        verdict['turns'] = kept
        verdict['gender'], verdict['confidence'] = _vote(kept)
        classified[new_id] = {
            'gender': minority,
            'confidence': _vote(moved)[1],
            'turns': moved,
            'split_from': speaker,
        }
        created[new_id] = speaker
    return created


FACE_WEIGHT = 2.5  # the speaking face's own look, when InsightFace is sure across its frames
FACE_MIN_SHARE = 0.75


def decide_gender(
    profile: dict, classifier: dict | None, heard: str | None, seen: dict | None = None
) -> dict:
    """Weighted vote of classifier, reviewer, pitch and (once faces are known) the speaking face's look.
    Returns the profile fields to set."""
    votes = {'male': 0.0, 'female': 0.0}
    source = []
    if (
        seen
        and seen.get('face_gender') in votes
        and (seen.get('face_gender_share') or 0) >= FACE_MIN_SHARE
    ):
        votes[seen['face_gender']] += FACE_WEIGHT * seen['face_gender_share']
        source.append('face')
    if (
        classifier
        and classifier.get('gender') in votes
        and classifier.get('confidence', 0) >= CLASSIFIER_MIN
    ):
        votes[classifier['gender']] += CLASSIFIER_WEIGHT * classifier['confidence']
        source.append('classifier')
    if heard in votes:
        votes[heard] += REVIEWER_WEIGHT
        source.append('review')
    if profile.get('gender') in votes:
        votes[profile['gender']] += min(1.0, float(profile.get('gender_confidence') or 0))
        source.append('pitch')
    total = sum(votes.values())
    if not total:
        return {'gender': 'unknown', 'gender_confidence': 0.0, 'gender_source': 'none'}
    gender = max(votes, key=votes.get)
    return {
        'gender': gender,
        'gender_confidence': round(votes[gender] / total, 2),
        'gender_source': '+'.join(source),
        'gender_votes': {k: round(v, 2) for k, v in votes.items()},
    }
