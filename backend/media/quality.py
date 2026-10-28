"""The dub quality check: what the pipeline can measure about a finished (or running) dub, as pass / check / fix items.

It cannot judge naturalness; it catches the failure modes seen in real jobs (speech recognised in the wrong script,
symbol-only lines, voice clips that looped, lines rushed past the pace band, lines left without a face, a silent export)
so that none of them reaches the viewer unnoticed."""

from __future__ import annotations

import wave
from pathlib import Path

from backend.voices.library import speaking_rate

PACE_SOFT = 1.15  # the exact-mode band; beyond it a line was rushed or ran into the next pause
LONG_CLIP = (
    2.5  # a voice clip this many times longer than the voice's rate predicts looped on a token
)
INDIC_SCRIPT = (
    "as",
    "bn",
    "brx",
    "doi",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
)


def _item(area: str, label: str, status: str, note: str) -> dict:
    return {"area": area, "label": label, "status": status, "note": note}


def _clip_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() / max(1, reader.getframerate())
    except (wave.Error, EOFError, OSError):
        return 0.0


def _has_target_words(text: str, language: str) -> bool:
    """A translated line must carry at least one word of the target script; code-mixed lines keep English terms in
    Roman letters by design, so Roman letters outnumbering the native ones is normal."""
    if language == "en":
        return any(ch.isalpha() for ch in text)
    return any(
        "\u0900" <= ch <= "\u0dff"
        or "\u0600" <= ch <= "\u06ff"
        or "\u1c50" <= ch <= "\u1c7f"
        or "\uabc0" <= ch <= "\uabff"
        for ch in text
    )


def assess(job: dict, directory: Path | None = None) -> dict:
    """Checks for a job record. `directory` (the job folder) lets the voice clips be measured."""
    items: list[dict] = []
    lines = job.get("segments") or []
    language = job.get("language") or ""
    spoken = [line for line in lines if line.get("text")]
    if spoken:
        reviewer = [line for line in spoken if line.get("text_source") == "review"]
        items.append(
            _item(
                "script",
                "Speech recognised",
                "check" if reviewer else "ok",
                (
                    f"{len(reviewer)} of {len(spoken)} lines fell back to the reviewer's transcript"
                    if reviewer
                    else f"{len(spoken)} lines, all from Bodhan ASR"
                ),
            )
        )
        missing = [line for line in spoken if not line.get("translation")]
        symbols = [
            line
            for line in spoken
            if line.get("translation") and not _has_target_words(line["translation"], language)
        ]
        if missing:
            items.append(
                _item("script", "Translation", "fix", f"{len(missing)} lines have no translation")
            )
        elif symbols:
            items.append(
                _item(
                    "script",
                    "Translation",
                    "check",
                    f"{len(symbols)} lines carry only symbols or Roman tokens, no {language} words",
                )
            )
        else:
            items.append(
                _item("script", "Translation", "ok", "every line has words in the target language")
            )
    voiced = [line for line in lines if line.get("voiced_text")]
    if voiced:
        long_clips = []
        if directory is not None:
            for index, line in enumerate(lines):
                if not line.get("voiced_text"):
                    continue
                seconds = _clip_seconds(directory / f"tts_{index:04d}.wav")
                expected = len(line["voiced_text"]) / max(
                    6.0,
                    speaking_rate(
                        line.get("voice")
                        or (job.get("speaker_voices") or {}).get(line.get("speaker"), ""),
                        12.0,
                    ),
                )
                if (
                    seconds
                    and len(line["voiced_text"]) >= 8
                    and seconds > max(LONG_CLIP * expected, 1.5)
                ):
                    long_clips.append(index)
        items.append(
            _item(
                "voice",
                "Voice clips",
                "fix" if long_clips else "ok",
                (
                    f"lines {', '.join(str(i + 1) for i in long_clips)} ran far longer than the voice's own pace: the engine looped"
                    if long_clips
                    else f"{len(voiced)} lines voiced at the voice's own pace"
                ),
            )
        )
        rushed = [line for line in voiced if (line.get("timing_ratio") or 1) > PACE_SOFT + 0.001]
        fit = job.get("fit_summary") or {}
        items.append(
            _item(
                "voice",
                "Timing",
                "check" if rushed else "ok",
                (
                    f"{len(rushed)} lines had to play faster than {PACE_SOFT}x or run into the next pause"
                    if rushed
                    else (
                        f"{fit.get('in_band', len(voiced))} of {fit.get('total', len(voiced))} lines fit their slot at a natural pace"
                        if fit
                        else "every line fits its slot"
                    )
                ),
            )
        )
    lip = job.get("lip_sync") or {}
    if lip and lip.get("engine") != "none":
        synced, total = lip.get("lines_lip_synced"), len(lines)
        off = lip.get("off_screen_speakers") or []
        coverage = lip.get("coverage_percent") or 0
        if synced is not None and total:
            # low coverage is usually the clip (voice-over, narration over footage); a fix only when nothing was synced at all
            status = (
                "ok"
                if synced == total or (off and coverage >= 60)
                else (
                    "fix"
                    if synced == 0 and (lip.get("speech_frames") or 0) > 0 and not off
                    else "check"
                )
            )
            note = f"{synced} of {total} lines lip-synced, {coverage}% of on-screen speech"
            if off:
                note += f"; {len(off)} speaker(s) are never on screen"
            items.append(_item("lips", "Lip sync", status, note))
        hands = lip.get("hand_covered_frames") or 0
        if hands:
            items.append(
                _item(
                    "lips",
                    "Occlusion",
                    "ok",
                    f"{hands} frames with a hand over the mouth were left as filmed",
                )
            )
        export = lip.get("export") or {}
        if export:
            peak = export.get("audio_peak_db")
            items.append(
                _item(
                    "export",
                    "Export",
                    "ok" if peak is not None and peak > -20 else "fix",
                    (
                        f"{export.get('width')}x{export.get('height')}, {export.get('duration', 0):.1f} s, audio peak {peak} dB"
                        if peak is not None
                        else "no audio measured"
                    ),
                )
            )
    elif lip.get("engine") == "none":
        items.append(_item("lips", "Lip sync", "ok", "voice only, as requested"))
    worst = "ok"
    for item in items:
        if item["status"] == "fix":
            worst = "fix"
        elif item["status"] == "check" and worst == "ok":
            worst = "check"
    return {"verdict": worst, "items": items}
