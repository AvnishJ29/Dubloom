"""Find speaker turns before transcription."""

from pathlib import Path
import wave

import numpy as np

from backend.models import Segment
from backend.speakers.speaker_embeddings import diarize
from backend.config import settings

MAX_SEGMENT_SECONDS = 25.0  # Bodhan ASR accepts at most 30 seconds per request.
MIN_SPLIT_SECONDS = 10.0  # long turns are cut at the quietest moment after this point
MERGE_GAP_SECONDS = 0.5
MIN_TURN_SECONDS = 0.25


def _load_samples(audio: Path):
    try:
        with wave.open(str(audio), 'rb') as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                return None
            return (
                np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(
                    np.float32
                )
                / 32768
            )
    except (OSError, wave.Error, EOFError):
        return None


def split_points(start: float, end: float, samples, rate: int = 16000) -> list[float]:
    """Cut a long turn at pauses: the quietest 200 ms window in the last part of each piece."""
    points = []
    cursor = start
    while end - cursor > MAX_SEGMENT_SECONDS:
        low, high = cursor + MIN_SPLIT_SECONDS, cursor + MAX_SEGMENT_SECONDS
        cut = high
        if samples is not None:
            frame = int(0.2 * rate)
            a, b = int(low * rate), min(int(high * rate), len(samples) - frame)
            if b - a > frame:
                region = samples[a:b]
                windows = np.lib.stride_tricks.sliding_window_view(region, frame)[:: frame // 4]
                energy = np.sqrt(np.mean(windows**2, axis=1))
                # Prefer later cuts slightly so pieces stay long; weigh energy by position.
                positions = np.linspace(0, 1, len(energy))
                score = energy * (1.4 - 0.4 * positions)
                best = int(np.argmin(score))
                cut = (a + best * (frame // 4) + frame / 2) / rate
        points.append(round(cut, 3))
        cursor = cut
    return points


def merge_turns(turns: list[Segment]) -> list[Segment]:
    """Join same-speaker turns separated by tiny pauses; drop unusable blips."""
    merged: list[Segment] = []
    for turn in sorted(turns, key=lambda item: item.start):
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end <= MERGE_GAP_SECONDS
            and turn.end - merged[-1].start <= MAX_SEGMENT_SECONDS
        ):
            merged[-1].end = max(merged[-1].end, turn.end)
        elif turn.end - turn.start >= MIN_TURN_SECONDS:
            merged.append(Segment(turn.start, turn.end, turn.speaker))
    return merged


CONTAINED_DROP_SECONDS = (
    1.0  # a turn inside another speaker's turn and shorter than this is a backchannel
)
CROSSTALK_SECONDS = 1.0  # partial overlaps up to this long are split at their midpoint
SLOT_MERGE_GAP = 0.3  # same-speaker turns closer than this form one dubbing slot
SHORT_SLOT_SECONDS = 1.5  # a slot shorter than this joins its same-speaker neighbour ...
SHORT_SLOT_GAP = 0.8  # ... when the pause between them is below this


def make_exclusive(turns: list[Segment]) -> list[Segment]:
    """Resolve overlaps so every moment belongs to one speaker, except genuine simultaneous speech.

    Short interjections inside another speaker's turn are dropped, and short
    crosstalk at a speaker change is split at its midpoint. Long overlaps stay:
    both people really talk at once and both lines are dubbed.
    """
    ordered = sorted(
        (Segment(t.start, t.end, t.speaker) for t in turns if t.end - t.start >= 0.2),
        key=lambda t: (t.start, -(t.end - t.start)),
    )
    kept: list[Segment] = []
    for turn in ordered:
        drop = False
        for other in kept:
            if other.speaker == turn.speaker or other.end <= turn.start or turn.end <= other.start:
                continue
            inside = other.start <= turn.start and turn.end <= other.end
            if inside and turn.end - turn.start < CONTAINED_DROP_SECONDS:
                drop = True
                break
            overlap = min(other.end, turn.end) - max(other.start, turn.start)
            if not inside and 0 < overlap <= CROSSTALK_SECONDS:
                middle = round((max(other.start, turn.start) + min(other.end, turn.end)) / 2, 3)
                if other.start <= turn.start:
                    other.end, turn.start = middle, middle
                else:
                    turn.end, other.start = middle, middle
        if not drop and turn.end - turn.start >= 0.2:
            kept.append(turn)
    return sorted((t for t in kept if t.end - t.start >= 0.2), key=lambda t: t.start)


def build_slots(turns: list[Segment]) -> list[Segment]:
    """Join same-speaker turns separated by tiny pauses; keep real pauses as separate slots."""
    slots: list[Segment] = []
    for turn in sorted(turns, key=lambda t: t.start):
        if slots and slots[-1].speaker == turn.speaker:
            previous = slots[-1]
            gap = turn.start - previous.end
            short = min(previous.end - previous.start, turn.end - turn.start) < SHORT_SLOT_SECONDS
            fits = turn.end - previous.start <= MAX_SEGMENT_SECONDS
            if fits and (gap < SLOT_MERGE_GAP or (short and gap < SHORT_SLOT_GAP)):
                previous.end = max(previous.end, turn.end)
                continue
        slots.append(Segment(turn.start, turn.end, turn.speaker))
    return slots


def refine_turns(turns: list[Segment], audio: Path) -> list[Segment]:
    """Raw diarization turns -> dubbing slots: exclusive, pause-aware, tight to speech, at most 25 s."""
    from backend.speakers.local_diarization import snap_boundaries

    samples = _load_samples(audio)
    slots = build_slots(make_exclusive(turns))
    if samples is not None:
        slots = snap_boundaries(slots, samples, pad_start=0.04, pad_end=0.08, reach=0.2)
    pieces = []
    for slot in slots:
        start = slot.start
        for cut in split_points(slot.start, slot.end, samples) + [slot.end]:
            if cut - start > 0.05:
                pieces.append(Segment(round(start, 3), round(cut, 3), slot.speaker))
            start = cut
    return pieces


def speaker_turns(
    audio: Path, progress=None, num_speakers: int | None = None
) -> tuple[list[Segment], str]:
    """Diarize `audio` in-process with the sherpa or speechbrain backend.

    Returns (turns, description of the method). Turns are merged (merge_turns) and long ones cut
    at pauses (split_points); times are in seconds. Any backend failure is raised as RuntimeError.
    """
    try:
        if settings.diarization_backend == 'sherpa':
            from backend.speakers.local_diarization import diarize as local_diarize

            turns = local_diarize(audio, progress=progress, num_speakers=num_speakers)
            mode = 'Pyannote segmentation 3.0 + CAM++ (local ONNX)'
        elif settings.diarization_backend == 'speechbrain':
            turns = diarize(audio)
            mode = 'Silero VAD + SpeechBrain ECAPA diarization'
        else:
            raise ValueError('Unknown DIARIZATION_BACKEND; use sherpa or speechbrain')
    except Exception as exc:
        raise RuntimeError(
            f'Speaker diarization failed ({settings.diarization_backend}): {exc}'
        ) from exc

    samples = _load_samples(audio)
    pieces = []
    for turn in merge_turns(turns):
        start = turn.start
        for cut in split_points(turn.start, turn.end, samples) + [turn.end]:
            if cut - start > 0.05:
                pieces.append(Segment(round(start, 3), round(cut, 3), turn.speaker))
            start = cut
    return pieces, mode
