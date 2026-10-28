"""Speaker review: who really says each line, as a dubbing supervisor would decide.

Acoustic diarization alone fails on films and trailers: filtered or shouted
voices of one character land in different clusters, and short lines of two
characters get merged. Here every raw turn is transcribed (OpenRouter, cheap),
then an audio-capable model listens to the dialogue stem with the turn list
and assigns each turn to a consistent voice, using timbre and context (names,
who answers whom). It also gives each voice a name and gender (for casting),
the emotion of each line (for the delivery style) and flags turns that are
not speech. Bodhan is not used here; it transcribes the final lines only.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.providers import openrouter
from backend.config import settings
from backend.speech.llm_asr import transcribe_clip
from backend.models import Segment

logger = logging.getLogger(__name__)

EMOTION_TO_STYLE = {
    "neutral": "neutral",
    "angry": "angry",
    "sad": "sad",
    "fearful": "fearful",
    "surprised": "surprised",
    "happy": "happy",
    "disgusted": "disgusted",
    "narration": "audiobook",
}

REVIEW_PROMPT = """You are the dubbing supervisor for a {kind}. You hear its dialogue track (music removed).
Below are the speech turns an automatic diarization system found: id, time in seconds, its automatic speaker label, and a rough transcript.
Automatic labels are often wrong in films and trailers: one character can be split over several labels (voice effects, shouting, distance), and short lines of different characters can share a label.
Assign every turn to a voice so that one person always has the same voice id and different people never share one. Decide from the timbre you hear and from context (names, who addresses or answers whom). Keep the automatic grouping where it is right.
For each voice give a short name and gender (male, female or unknown). Use a character's name only when the dialogue itself establishes it (they name themselves or are clearly addressed by it in this audio); never guess from outside knowledge. Otherwise use a plain description such as "Narrator", "Man 1" or "Woman 2".
For each turn give the emotion of its delivery: one of neutral, angry, sad, fearful, surprised, happy, disgusted, narration (calm voice-over storytelling). Mark "speech": false for turns that are not words (breath, scream, music, effect).
Respond with JSON only:
{{"voices": [{{"id": "V1", "name": "...", "gender": "female"}}], "turns": [{{"id": 0, "voice": "V1", "emotion": "sad", "speech": true}}]}}"""


def _clip(audio: Path, start: float, end: float, target: Path) -> Path:
    """Cut [start, end] of `audio`, with 0.1 s margins, to a 16 kHz mono file at `target`."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, start - 0.1):.3f}",
            "-to",
            f"{end + 0.1:.3f}",
            "-i",
            str(audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ],
        check=True,
    )
    return target


def _transcripts(
    turns: list[Segment], audio: Path, work: Path, language: str | None, usage
) -> list[str]:
    """Transcript of every turn, transcribed in parallel; "" where transcription failed."""
    work.mkdir(exist_ok=True)

    def one(index: int) -> str:
        turn = turns[index]
        try:
            return transcribe_clip(
                _clip(audio, turn.start, turn.end, work / f"turn_{index:03d}.wav"),
                language,
                {},
                usage,
            )
        except Exception as exc:  # a missing transcript only weakens the review
            logger.warning("Review transcript of turn %s failed: %s", index, exc)
            return ""

    with ThreadPoolExecutor(max(1, settings.model_request_workers)) as pool:
        return list(pool.map(one, range(len(turns))))


def _parse(content: str) -> dict:
    """The JSON object in a model reply, ignoring code fences and surrounding text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    return json.loads(text[text.find("{") : text.rfind("}") + 1])


REVIEW_RUNS = 3  # independent reviews; turns are grouped where a majority agrees
MALE_PITCH_HZ = (
    145.0  # a voice whose median pitch is below this is male, whatever the reviewer heard
)


def _pitches(turns: list[Segment], audio: Path) -> list[float | None]:
    """Median voiced F0 of each turn (None when whispered or unvoiced)."""
    try:
        import librosa
        import numpy as np

        samples, rate = librosa.load(str(audio), sr=16000)
    except Exception as exc:
        logger.warning("Pitch analysis unavailable: %s", exc)
        return [None] * len(turns)
    result = []
    for turn in turns:
        clip = samples[int(turn.start * rate) : int(turn.end * rate)]
        if len(clip) < rate // 5:
            result.append(None)
            continue
        f0, voiced, _ = librosa.pyin(clip, fmin=65, fmax=500, sr=rate, frame_length=1024)
        values = f0[voiced & np.isfinite(f0)]
        result.append(float(np.median(values)) if len(values) >= 5 else None)
    return result


def _one_review(prompt: str, data: str, usage) -> dict | None:
    """One reviewer pass over the dialogue audio (`data`: base64 MP3).

    Returns {"voices": {voice id: voice}, "turns": {turn id: verdict}}, or None when the request
    or its JSON fails.
    """
    try:
        content = openrouter.chat(
            settings.openrouter_asr_chat_model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}},
                    ],
                }
            ],
            usage=usage,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        answer = _parse(content)
    except Exception as exc:
        logger.warning("Speaker review run failed: %s", exc)
        return None
    voices = {
        str(v.get("id")): v for v in answer.get("voices", []) if isinstance(v, dict) and v.get("id")
    }
    turns = {}
    for item in answer.get("turns", []):
        try:
            turns[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return {"voices": voices, "turns": turns}


def _majority(values: list[str], default: str) -> str:
    """The most common non-empty value (ties go to the one seen first), or `default` when there is none."""
    values = [v for v in values if v]
    return max(dict.fromkeys(values), key=values.count) if values else default


def review_speakers(
    turns: list[Segment],
    audio: Path,
    work: Path,
    source_language: str | None = None,
    usage=None,
    kind: str = "video",
) -> tuple[list[Segment], dict] | None:
    """Raw turns -> (relabelled turns without non-speech, report with voices). None if the review is unavailable."""
    if not turns or not openrouter.configured() or not settings.openrouter_asr_chat_model:
        return None
    texts = _transcripts(turns, audio, work, source_language, usage)
    pitches = _pitches(turns, audio)
    listing = "\n".join(
        f"{i} | {t.start:.2f}-{t.end:.2f} | {t.speaker} | "
        f"{f'{pitches[i]:.0f} Hz' if pitches[i] else 'unvoiced'} | {texts[i] or '(no words recognised)'}"
        for i, t in enumerate(turns)
    )
    prompt = (
        REVIEW_PROMPT.format(kind=kind)
        + "\nThe measured median pitch of each turn is given: typical adult male "
        "speech is 85-155 Hz and female 165-255 Hz, while shouting raises both a lot."
        "\n\nTurns (id | seconds | automatic label | pitch | transcript):\n" + listing
    )
    compressed = work / "dialogue_review.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(audio),
            "-ac",
            "1",
            "-b:a",
            "48k",
            str(compressed),
        ],
        check=True,
    )
    data = base64.b64encode(compressed.read_bytes()).decode()
    with ThreadPoolExecutor(REVIEW_RUNS) as pool:
        runs = [
            run
            for run in pool.map(lambda _: _one_review(prompt, data, usage), range(REVIEW_RUNS))
            if run
        ]
    if not runs:
        return None

    # Non-speech turns: dropped when a majority of runs say so and at most one word was recognised.
    keep = [
        index
        for index in range(len(turns))
        if not (
            sum(run["turns"].get(index, {}).get("speech") is False for run in runs) * 2 > len(runs)
            and len(texts[index].split()) <= 1
        )
    ]

    # Group turns that a majority of runs put in the same voice (a turn a run skipped keeps its automatic label).
    def voice_of(run, index):
        return str(run["turns"].get(index, {}).get("voice") or f"auto:{turns[index].speaker}")

    # Average-linkage clustering of the agreement matrix: two groups join only when, on average, a majority
    # of reviews put their turns together (pairwise votes alone could chain unrelated turns).
    import numpy as np

    agree = np.array(
        [
            [sum(voice_of(run, a) == voice_of(run, b) for run in runs) / len(runs) for b in keep]
            for a in keep
        ],
        dtype=float,
    )
    groups = {position: [position] for position in range(len(keep))}
    link = agree.copy()
    np.fill_diagonal(link, -1.0)
    while len(groups) > 1:
        a, b = np.unravel_index(int(np.argmax(link)), link.shape)
        if link[a, b] <= 0.5:
            break
        a, b = min(a, b), max(a, b)
        size_a, size_b = len(groups[a]), len(groups[b])
        merged = (link[a] * size_a + link[b] * size_b) / (size_a + size_b)
        link[a, :], link[:, a] = merged, merged
        link[a, a] = -1.0
        link[b, :], link[:, b] = -1.0, -1.0
        groups[a] += groups.pop(b)
    group_of = {keep[position]: min(members) for members in groups.values() for position in members}

    def find(index):
        return group_of[index]

    order, reviewed, members = {}, [], {}
    for index in keep:
        speaker = order.setdefault(find(index), f"SPEAKER_{len(order):02d}")
        members.setdefault(speaker, []).append(index)
        segment = Segment(turns[index].start, turns[index].end, speaker)
        emotion = _majority(
            [str(run["turns"].get(index, {}).get("emotion", "")).lower() for run in runs], "neutral"
        )
        segment.style = EMOTION_TO_STYLE.get(emotion, "neutral")
        segment.reference_text = texts[index]
        reviewed.append(segment)

    voices = {}
    for speaker, indices in members.items():
        names, genders = [], []
        for run in runs:
            for index in indices:
                info = run["voices"].get(voice_of(run, index), {})
                names.append(str(info.get("name") or ""))
                genders.append(str(info.get("gender") or "").lower())
        voiced = [pitches[i] for i in indices if pitches[i]]
        pitch = float(sorted(voiced)[len(voiced) // 2]) if voiced else None
        gender = _majority([g for g in genders if g in {"male", "female"}], "unknown")
        if pitch is not None and pitch < MALE_PITCH_HZ:
            gender = "male"
        voices[speaker] = {
            "name": _majority(names, ""),
            "gender": gender,
            "pitch_hz": round(pitch, 1) if pitch else None,
        }
    report = {
        "model": settings.openrouter_asr_chat_model,
        "runs": len(runs),
        "automatic_speakers": len({t.speaker for t in turns}),
        "reviewed_speakers": len(order),
        "dropped_turns": len(turns) - len(keep),
        "voices": voices,
        "transcripts": texts,
        "pitches": [round(p, 1) if p else None for p in pitches],
    }
    return reviewed, report
