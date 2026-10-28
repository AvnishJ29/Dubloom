"""Stable, distinct Bodhan voice assignments for any number of speakers.

Voices are chosen in first-appearance order from a language-priority pool. A
per-speaker gender estimate (see speaker_profiles) prefers a matching voice,
explicit overrides always win, and once the pool is exhausted voices repeat
consistently while the reuse is reported.
"""

from backend.models import VOICES, VOICE_CATALOG, VOICE_GENDERS
from backend.voices.library import LIBRARY, cast


def voice_pool(language: str) -> list[str]:
    if language not in VOICES:
        raise ValueError('Unsupported speech language')
    priorities = [language]
    for group in VOICE_CATALOG.get('priority_groups', []):
        if language in group:
            priorities.extend(group)
    priorities.extend(VOICES)
    return list(dict.fromkeys(voice for code in priorities for voice in VOICES[code]))


def voice_details(language: str) -> list[dict]:
    details = []
    for name in voice_pool(language):
        card = LIBRARY.get(name, {})
        details.append(
            {
                'name': name,
                'gender': card.get('gender') or VOICE_GENDERS.get(name, 'unknown'),
                'language': card.get('language', ''),
                'pitch_hz': card.get('pitch_hz'),
                'description': card.get('description', ''),
                'best_for': card.get('best_for', []),
            }
        )
    return details


def assign_voices(
    speakers,
    language: str,
    preferred_voice: str | None = None,
    profiles: dict | None = None,
    overrides: dict | None = None,
    genre: str | None = None,
    reasons: dict | None = None,
    preferred_voices: list[str] | None = None,
) -> dict[str, str]:
    """Use the preferred voice for the first speaker, then distinct voices.
    With the voice library present, casting scores language, genre fit and register (see voice_library);
    `reasons`, if given, is filled with one line per speaker."""
    speakers = list(dict.fromkeys(speakers))
    voices = voice_pool(language)
    if preferred_voice and preferred_voice not in voices:
        raise ValueError('Unknown Bodhan voice')
    if LIBRARY and all(name in LIBRARY for name in voices):
        chosen, why = cast(
            speakers,
            language,
            profiles,
            genre,
            preferred_voices or preferred_voice,
            overrides,
            pool=voices,
        )
        if reasons is not None:
            reasons.update(why)
        return chosen
    if preferred_voice:
        if preferred_voice not in voices:
            raise ValueError('Unknown Bodhan voice')
        voices.remove(preferred_voice)
        voices.insert(0, preferred_voice)
    if not voices:
        raise ValueError('No TTS voices are configured for this provider')
    assignments: dict[str, str] = {}
    used: set[str] = set()
    for speaker, voice in (overrides or {}).items():
        if speaker in speakers and voice in voices:
            assignments[speaker] = voice
            used.add(voice)
    for index, speaker in enumerate(speakers):
        if speaker in assignments:
            continue
        available = [voice for voice in voices if voice not in used]
        chosen = None
        if index == 0 and preferred_voice and preferred_voice not in used:
            chosen = preferred_voice
        elif available:
            wanted = (profiles or {}).get(speaker, {}).get('gender')
            if wanted in {'male', 'female'}:
                matching = [voice for voice in available if VOICE_GENDERS.get(voice) == wanted]
                chosen = matching[0] if matching else available[0]
            else:
                chosen = available[0]
        else:
            chosen = voices[len(assignments) % len(voices)]
        assignments[speaker] = chosen
        used.add(chosen)
    return {speaker: assignments[speaker] for speaker in speakers}


def voice_reuse_warnings(assignments: dict[str, str]) -> list[str]:
    """Report provider capacity without imposing a limit on speaker count."""
    unique_count = len(set(assignments.values()))
    if len(assignments) > unique_count:
        return [
            f'{len(assignments)} speakers share {unique_count} available TTS voices. '
            'Speaker labels and timing remain separate; some synthetic voices repeat.'
        ]
    return []


def overlapping_turns(segments, minimum_overlap=0.15) -> list[str]:
    """Flag meaningful cross-speaker overlaps; segmentation jitter is ignored."""
    warnings = []
    ordered = sorted(segments, key=lambda segment: segment.start)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if second.start >= first.end:
                break
            overlap = min(first.end, second.end) - second.start
            if first.speaker != second.speaker and overlap > minimum_overlap:
                warnings.append(
                    f'Overlapping speakers at {second.start:.1f}s ({overlap:.1f}s). Speech is not isolated; transcription and dubbing need review.'
                )
    return warnings
