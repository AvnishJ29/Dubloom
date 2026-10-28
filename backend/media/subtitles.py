"""Subtitle export for the source transcript and the translated dub."""


def _timestamp(seconds: float, separator: str = ",") -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    whole, fraction = divmod(rest, 1)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(whole):02d}{separator}{int(round(fraction * 1000)):03d}"


def build_subtitles(
    segments: list[dict], track: str = "target", fmt: str = "srt", with_speaker: bool = True
) -> str:
    """Build source or translated SRT/WebVTT cues in source-timeline order."""
    key = "translation" if track == "target" else "text"
    lines = ["WEBVTT", ""] if fmt == "vtt" else []
    separator = "." if fmt == "vtt" else ","
    number = 0
    for segment in sorted(segments, key=lambda item: item.get("start", 0)):
        text = (segment.get(key) or "").strip()
        if not text:
            continue
        number += 1
        start, end = segment.get("start", 0), segment.get("end", 0)
        if end <= start:
            end = start + 1.0
        label = segment.get("speaker", "").replace("_", " ").title()
        body = f"{label}: {text}" if with_speaker and label else text
        if fmt != "vtt":
            lines.append(str(number))
        lines.append(f"{_timestamp(start, separator)} --> {_timestamp(end, separator)}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
