"""Exact placement of dubbed lines.

Every dubbed line occupies exactly the slot of the original line: it starts
when the original speaker starts and ends when they stop. The clip is
time-stretched (pitch preserved) by speed = clip / slot. The fitting stage
rewrites and re-voices lines until that speed lies in [FIT_MIN_SPEED,
FIT_MAX_SPEED] (0.9x - 1.1x by default); a line still outside the band after
every rewrite is stretched within the hard limits below and reported, never
moved into a neighbouring slot.
"""

from __future__ import annotations

from dataclasses import dataclass

HARD_MIN_SPEED = 0.8  # natural mode: slower than this sounds dragged, keep 0.8x and leave the rest of the slot silent
HARD_MAX_SPEED = (
    1.35  # natural mode: faster than this is hard to follow, keep 1.35x and trim the tail
)
PACE_MIN = 0.85  # exact mode: the pace band that still sounds like the same speaker
PACE_MAX = 1.15
PACE_HARD_MAX = 1.3  # beyond this a too-long line is trimmed rather than rushed
GAP_MARGIN = 0.15  # seconds kept free before the next line when a long line borrows from the gap


@dataclass
class Placement:
    """The final start, tempo, occupied duration, and compromises for one line."""

    index: int
    start: float
    speed: float  # tempo applied to the natural clip
    room: float  # seconds of audio placed (== slot unless the hard limits applied)
    ratio: float  # natural clip / slot: what the line needed
    trimmed: bool = False
    borrowed: float = 0.0  # seconds taken from the silence after the slot (exact mode)


SHORT_TAIL_SECONDS = (
    0.25  # a too-short line keeps the 0.9x floor if this little silence (or 5 %) is left
)


def plan_timeline(
    lines: list[tuple[float, float, float]],
    total_duration: float | None = None,
    min_speed: float = 0.9,
    exact: bool = False,
    pace_min: float = PACE_MIN,
    pace_max: float = PACE_MAX,
) -> list[Placement]:
    """`lines`: (slot_start, slot_end, natural_clip_seconds); result in timeline order.

    `exact`: a line starts with the original speaker and is stretched to end with them, but only within
    [pace_min, pace_max] so it still sounds like the same person. A line that would need more speed first
    borrows from the silence after its slot (up to GAP_MARGIN before the next line), then is trimmed at
    PACE_HARD_MAX; a line that would need to drag ends early instead. The natural mode keeps the older
    0.8x-1.35x limits without borrowing."""
    order = sorted(range(len(lines)), key=lambda i: lines[i][0])
    placements = []
    for position, i in enumerate(order):
        start, end, clip = lines[i]
        slot = max(0.05, end - start)
        ratio = clip / slot if slot else 1.0
        if exact:
            if pace_min <= ratio <= pace_max:
                placements.append(
                    Placement(i, round(start, 3), round(ratio, 4), round(slot, 3), round(ratio, 3))
                )
                continue
            if ratio < pace_min:  # too short: keep the pace, end a little early
                speed = pace_min
                placements.append(
                    Placement(i, round(start, 3), speed, round(clip / speed, 3), round(ratio, 3))
                )
                continue
            next_start = (
                lines[order[position + 1]][0]
                if position + 1 < len(order)
                else (total_duration or end + 2.0)
            )
            limit = max(slot, next_start - start - GAP_MARGIN)
            if clip / pace_max <= limit:  # borrow silence after the slot instead of rushing
                speed, room = pace_max, clip / pace_max
            else:
                speed = min(clip / limit, PACE_HARD_MAX)
                room = min(limit, clip / speed)
            placements.append(
                Placement(
                    i,
                    round(start, 3),
                    round(speed, 4),
                    round(room, 3),
                    round(ratio, 3),
                    trimmed=bool(clip / speed > limit + 1e-6),
                    borrowed=float(round(max(0.0, room - slot), 3)),
                )
            )
            continue
        speed = min(max(ratio, HARD_MIN_SPEED), HARD_MAX_SPEED)
        if ratio < min_speed and slot - clip / min_speed <= max(SHORT_TAIL_SECONDS, 0.05 * slot):
            speed = min_speed  # slowing further would sound dragged; a short pause ends the line instead
        room = min(slot, clip / speed)
        placements.append(
            Placement(
                i,
                round(start, 3),
                round(speed, 4),
                round(room, 3),
                round(ratio, 3),
                trimmed=bool(ratio > HARD_MAX_SPEED),
            )
        )
    return placements


def ratio_in_band(ratio: float, low: float, high: float) -> bool:
    return low - 1e-6 <= ratio <= high + 1e-6


def dubbed_slots(segments: list[dict]) -> list[tuple[float, float, str]]:
    """The frames whose mouth must follow the dub: the original speech and the placed audio together.
    A line that borrows the pause after it keeps moving its lips while the dub still speaks; a line that ends early
    is still redrawn to the end of the original speech, where the dub is silent and the mouth must close (left as
    filmed, it would keep saying the original words). Older jobs without placement metadata keep the turn bounds.
    """
    return [
        (
            (
                min(line["start"], line["dub_start"])
                if line.get("dub_start") is not None
                else line["start"]
            ),
            max(line["end"], line["dub_end"]) if line.get("dub_end") is not None else line["end"],
            line["speaker"],
        )
        for line in segments
        if line.get("translation")
    ]


def placement_notes(segments) -> list[str]:
    """Describe what the mix actually did, rather than predicting from raw TTS."""
    notes = []
    for line in segments:
        who = f"{line.speaker.replace('_', ' ').title()} at {line.start:.1f}s"
        if line.timing_trimmed:
            notes.append(
                f"{who}: audio was cut at the end at {line.timing_ratio:.2f}x speed. Shorten this line and re-render to keep every word."
            )
        elif line.timing_borrowed > 0.01:
            notes.append(
                f"{who}: uses {line.timing_borrowed:.2f}s of the following pause at {line.timing_ratio:.2f}x speed."
            )
        elif line.dub_end is not None and line.end - line.dub_end > SHORT_TAIL_SECONDS:
            notes.append(
                f"{who}: ends {line.end - line.dub_end:.2f}s early to preserve a natural pace."
            )
        elif line.timing_ratio > PACE_MAX:
            notes.append(
                f"{who}: sped up to {line.timing_ratio:.2f}x. A shorter script would sound more natural."
            )
    return notes
