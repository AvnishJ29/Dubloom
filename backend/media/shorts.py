"""Shorts: a span of a finished dub as a vertical 9:16 clip, with the dubbed lines burned in when asked.

The crop follows the speaking face when face analysis is available (one steady window for the whole short,
no panning), otherwise the centre; 'blur' keeps the full width over a blurred background instead."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.config import ROOT

FONT_DIR = ROOT / "fonts"
# The bundled font for each script (Unicode block), so burned-in lines never show as empty boxes.
# Every one of these fonts also has Latin letters, for the English words in code-mixed lines.
SCRIPT_FONTS = (
    (0x0600, 0x06FF, "Noto Naskh Arabic"),  # Urdu, Kashmiri, Sindhi
    (0x0900, 0x097F, "Noto Sans Devanagari"),  # Hindi, Marathi, Nepali, Sanskrit, ...
    (0x0980, 0x09FF, "Noto Sans Bengali"),  # Bengali, Assamese, Manipuri
    (0x0A00, 0x0A7F, "Noto Sans Gurmukhi"),
    (0x0A80, 0x0AFF, "Noto Sans Gujarati"),
    (0x0B00, 0x0B7F, "Noto Sans Oriya"),
    (0x0B80, 0x0BFF, "Noto Sans Tamil"),
    (0x0C00, 0x0C7F, "Noto Sans Telugu"),
    (0x0C80, 0x0CFF, "Noto Sans Kannada"),
    (0x0D00, 0x0D7F, "Noto Sans Malayalam"),
    (0x1C50, 0x1C7F, "Noto Sans Ol Chiki"),  # Santali
    (0xABC0, 0xABFF, "Noto Sans Meetei Mayek"),  # Manipuri
)
DEFAULT_FONT = "Noto Sans Devanagari"  # English-only lines
MAX_SECONDS = 180.0
OUT_W, OUT_H = 1080, 1920


def subtitle_font(text: str) -> str:
    """The bundled font for the script most of `text` is written in."""
    scripts = Counter(
        font for char in text for low, high, font in SCRIPT_FONTS if low <= ord(char) <= high
    )
    return scripts.most_common(1)[0][0] if scripts else DEFAULT_FONT


def _probe(path: Path) -> tuple[int, int, float]:
    """(width, height, duration in seconds) of the first video stream."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(out)
    return (
        int(data["streams"][0]["width"]),
        int(data["streams"][0]["height"]),
        float(data["format"]["duration"]),
    )


def _srt_time(seconds: float) -> str:
    """SRT timestamp (HH:MM:SS,mmm) for a time in seconds."""
    ms = int(round(max(0.0, seconds) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


CUE_CHARS = 64  # about two subtitle rows on a phone


def cues(text: str, limit: int = CUE_CHARS) -> list[str]:
    """Split a long line into readable cues at punctuation, then at spaces, never mid-word."""
    words, pieces, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
        if current.endswith((",", "।", ".", "?", "!", ";")) and len(current) > limit * 0.45:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)
    return pieces


def write_subtitles(segments: list[dict], start: float, end: float, path: Path) -> int:
    """SRT of the dubbed lines inside [start, end], shifted to the short's own clock, long lines split into
    cues that share the line's time by character count. Returns the cue count."""
    lines = []
    for segment in segments:
        a, b = (
            segment.get("dub_start") if segment.get("dub_start") is not None else segment["start"]
        ), segment.get("dub_end") or segment["end"]
        text = (segment.get("translation") or "").strip()
        if not text or b <= start or a >= end:
            continue
        pieces = cues(text)
        total = sum(len(p) for p in pieces) or 1
        at = a
        for piece in pieces:
            span = (b - a) * len(piece) / total
            cue_start, cue_end = max(at, start), min(at + span, end)
            at += span
            if cue_end - cue_start < 0.2:
                continue
            lines.append(
                f"{len(lines) + 1}\n{_srt_time(cue_start - start)} --> {_srt_time(cue_end - start)}\n{piece}\n"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def face_crop_x(
    directory: Path, start: float, end: float, width: int, crop_w: int, fps: float = 25.0
) -> int | None:
    """Horizontal crop offset that keeps the largest face of the span in view; None without face data."""
    faces = directory / "faces.npz"
    if not faces.is_file():
        return None
    with np.load(faces) as data:
        frames, boxes = data["frame"], data["bbox"]
    inside = (frames >= int(start * fps)) & (frames < int(end * fps))
    if not inside.any():
        return None
    centres = []
    for frame in np.unique(frames[inside]):
        rows = np.where(inside & (frames == frame))[0]
        biggest = rows[
            np.argmax((boxes[rows, 2] - boxes[rows, 0]) * (boxes[rows, 3] - boxes[rows, 1]))
        ]
        centres.append((boxes[biggest, 0] + boxes[biggest, 2]) / 2)
    centre = float(np.median(centres))
    return int(min(max(0, centre - crop_w / 2), width - crop_w))


def make_short(
    job: dict,
    directory: Path,
    start: float,
    end: float,
    layout: str = "face",
    subtitles: bool = False,
) -> dict:
    """Render shorts/short_<n>.mp4 and return its record (the caller appends it to job['shorts'])."""
    source = directory / "dubbed_video.mp4"
    if not source.is_file():
        raise ValueError("Render the dub first")
    width, height, duration = _probe(source)
    start, end = max(0.0, float(start)), min(float(end), duration)
    if end - start < 1.0:
        raise ValueError("A short needs at least one second")
    if end - start > MAX_SECONDS:
        raise ValueError(f"A short can be at most {int(MAX_SECONDS)} seconds")
    number = max((s["number"] for s in job.get("shorts") or []), default=0) + 1
    folder = directory / "shorts"
    folder.mkdir(exist_ok=True)
    output = folder / f"short_{number}.mp4"
    crop_w = min(width, int(height * OUT_W / OUT_H) // 2 * 2)
    filters = []
    if layout == "blur":
        filters.append(
            f"split[bg][fg];[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},gblur=sigma=30[bg];"
            f"[fg]scale={OUT_W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        x = face_crop_x(directory, start, end, width, crop_w) if layout == "face" else None
        if x is None:
            x = (width - crop_w) // 2
        filters.append(f"crop={crop_w}:{height}:{x}:0,scale={OUT_W}:{OUT_H}")
    lines = 0
    if subtitles:
        srt = folder / f"short_{number}.srt"
        lines = write_subtitles(job.get("segments") or [], start, end, srt)
        if lines:
            font = subtitle_font(srt.read_text(encoding="utf-8"))
            # Encoding=-1 makes libass take each line's direction from its text (Urdu reads right to left)
            style = f"FontName={font},FontSize=15,Bold=1,Outline=1.5,Shadow=0,MarginV=64,Alignment=2,Encoding=-1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000"
            filters.append(
                f"subtitles={srt.as_posix()}:fontsdir={FONT_DIR.as_posix()}:force_style='{style}'"
            )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.3f}",
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "faster",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return {
        "number": number,
        "start": round(start, 2),
        "end": round(end, 2),
        "layout": layout,
        "subtitles": bool(lines),
        "lines": lines,
        "file": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
