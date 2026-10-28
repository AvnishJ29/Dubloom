"""Fit synthesized lines to speaker turns and build the final audio track.

Every clip is placed at its source turn. A clip that runs long may borrow the
silence before the next turn (never overlapping it) before it is sped up; a
clip that runs short keeps a natural pace instead of being stretched. The
optional background bed keeps music and ambience from the original audio,
ducked while the dub is speaking, and the mix is loudness-normalised.
"""

from functools import lru_cache
from pathlib import Path
import subprocess

from backend.media.audio_norm import level_clip
from backend.config import settings
from backend.media.timeline import plan_timeline
from backend.media.video_input import duration, run
from backend.models import Segment

SILENCE_TRIM = (
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,areverse,"
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.12,areverse"
)


@lru_cache(maxsize=1)
def rubberband_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20
        )
        return "rubberband" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def _tempo_filters(speed: float) -> list[str]:
    if abs(speed - 1.0) < 0.005:
        return []
    if rubberband_available():
        # Pitch-preserving, formant-preserving stretch: cleaner than atempo on speech.
        return [f"rubberband=tempo={speed:.4f}:transients=crisp:formant=preserved:pitchq=quality"]
    filters = []
    while speed > 2:
        filters.append("atempo=2")
        speed /= 2
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed *= 2
    filters.append(f"atempo={speed:.6f}")
    return filters


def trimmed_clip(source: Path, directory: Path, index: int | None = None) -> Path:
    """The clip without its leading/trailing silence, cached per source clip (not per list position)."""
    target = (
        directory / f"trim_{source.stem}_agc.wav"
    )  # _agc: levelled with the AGC recipe, older trims are rebuilt
    if not (target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns):
        run("ffmpeg", "-y", "-v", "error", "-i", source, "-af", SILENCE_TRIM, "-ac", "1", target)
        level_clip(target)  # cached clips from older jobs are levelled here, at mix time
    return target


def natural_durations(segments, directory: Path) -> dict[int, float]:
    """Trimmed clip length per segment index (segments with audio only)."""
    result = {}
    for index, segment in enumerate(segments):
        if segment.audio_path and Path(segment.audio_path).is_file():
            result[index] = duration(trimmed_clip(Path(segment.audio_path), directory, index))
    return result


LINE_FADE_IN, LINE_FADE_OUT = 0.02, 0.05  # seconds
DUCK_FALL, DUCK_RISE = (
    0.25,
    0.6,
)  # seconds the bed takes to dip before a line and to come back after it


def duck_envelope(
    intervals: list[tuple[float, float]],
    full: float,
    duck: float,
    fall: float = DUCK_FALL,
    rise: float = DUCK_RISE,
) -> str:
    """ffmpeg `volume` expression: `full` away from speech, `duck` during it, with linear ramps at every edge
    instead of a step (a step at each line start and end is heard as the music cutting in and out).
    """
    if not intervals:
        return f"{full}"
    ramps = [
        f"clip((t-{a - fall:.3f})/{fall},0,1)*clip(({b + rise:.3f}-t)/{rise},0,1)"
        for a, b in intervals
    ]
    return f"{full}-{full - duck:.4f}*min(1,{'+'.join(ramps)})"


def speech_intervals(segments, pad: float = 0.15) -> list[tuple[float, float]]:
    """Union of all diarized turns; used for ducking the background bed."""
    spans = sorted((max(0.0, s.start - pad), s.end + pad) for s in segments)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def assemble_timeline(
    segments: list[Segment],
    total_duration: float,
    directory: Path,
    background: Path | None = None,
    background_mode: str = "duck",
) -> Path:
    """Place every utterance on a silent timeline using the elastic plan."""
    ordered = sorted((s for s in segments if s.end > s.start), key=lambda s: s.start)
    voiced = [(index, segment) for index, segment in enumerate(ordered) if segment.audio_path]
    durations = {}
    trimmed = {}
    for index, segment in voiced:
        trimmed[index] = trimmed_clip(Path(segment.audio_path), directory, index)
        durations[index] = duration(trimmed[index])
    plan = plan_timeline(
        [(ordered[i].start, ordered[i].end, durations[i]) for i, _ in voiced],
        total_duration,
        settings.fit_min_speed,
        exact=settings.timing_mode == "exact",
        pace_min=settings.pace_min,
        pace_max=settings.pace_max,
    )
    by_index = {placement.index: placement for placement in plan}
    prepared = []
    for position, (index, segment) in enumerate(voiced):
        placement = by_index[position]
        target = directory / f"fitted_{index:04d}.wav"
        segment.timing_ratio = round(placement.speed, 3)
        segment.fit_ratio = round(placement.ratio, 3)
        segment.timing_trimmed = placement.trimmed
        segment.timing_borrowed = placement.borrowed
        segment.dub_start = placement.start
        segment.dub_end = round(placement.start + placement.room, 3)
        # Short fades at both ends: the silence trim cuts at a threshold, which otherwise clicks.
        filters = _tempo_filters(placement.speed) + [
            f"atrim=duration={placement.room:.3f}",
            f"afade=t=in:d={LINE_FADE_IN}",
            f"afade=t=out:st={max(0.0, placement.room - LINE_FADE_OUT):.3f}:d={LINE_FADE_OUT}",
            "aresample=48000",
        ]
        run(
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            trimmed[index],
            "-af",
            ",".join(filters),
            "-ac",
            "1",
            target,
        )
        prepared.append((segment, target, placement))
    if not prepared:
        raise RuntimeError("No speech was transcribed from this video")
    output = directory / "dubbed_audio.wav"
    voice_only = directory / "dubbed_voice.wav"
    # A finite silence bed fixes the exact timeline length between speech turns.
    args = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=48000:cl=mono:d={total_duration:.3f}",
    ]
    for _, path, _ in prepared:
        args.extend(["-i", str(path)])
    filters = []
    labels = []
    for index, (segment, _, placement) in enumerate(prepared, 1):
        delay = round(placement.start * 1000)
        filters.append(f"[{index}:a]adelay={delay}|{delay}[s{index}]")
        labels.append(f"[s{index}]")
    if background is not None:
        bed_index = len(prepared) + 1
        args.extend(["-i", str(background)])
        placed = [
            Segment(placement.start, placement.start + placement.room, segment.speaker)
            for segment, _, placement in prepared
        ]
        if background_mode == "separate":
            full, duck = settings.background_level, settings.background_speech_level
        else:
            full, duck = settings.duck_level, settings.duck_speech_level
        expression = duck_envelope(speech_intervals(placed), full, duck)
        filters.append(
            f"[{bed_index}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume=volume='{expression}':eval=frame[bed]"
        )
        labels.append("[bed]")
    voice_labels = [label for label in labels if label != "[bed]"]
    inputs = "[0:a]" + "".join(voice_labels)
    # The clean voice timeline drives lip sync; the delivered stereo mix may add the background bed.
    filters.append(
        f"{inputs}amix=inputs={len(voice_labels) + 1}:normalize=0:duration=longest,"
        "asplit=2[voice_raw][voice_mix]"
    )
    filters.append("[voice_raw]alimiter=limit=0.95:latency=1[voice_out]")
    filters.append("[voice_mix]pan=stereo|c0=c0|c1=c0[voice_st]")
    if background is not None:
        filters.append(
            "[voice_st][bed]amix=inputs=2:normalize=0:duration=longest,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.95:latency=1[out]"
        )
    else:
        filters.append("[voice_st]loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.95:latency=1[out]")
    script = directory / "mix_filters.txt"
    script.write_text(";".join(filters), encoding="utf-8")
    args.extend(
        [
            "-filter_complex_script",
            str(script),
            "-map",
            "[voice_out]",
            "-t",
            f"{total_duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(voice_only),
            "-map",
            "[out]",
            "-t",
            f"{total_duration:.3f}",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    run(*args)
    return output
