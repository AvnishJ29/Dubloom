"""Validate the delivered MP4, including decoded audio rather than headers alone."""

import json
import math
from pathlib import Path
import re
import subprocess
from backend.media.video_input import run


def validate_video(path: Path, expected_duration: float) -> dict:
    """Check export codecs, duration alignment, and decoded audio audibility."""
    probe = json.loads(
        run('ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path)
    )
    video = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    audio = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
    if not video or not audio or video['codec_name'] != 'h264' or audio['codec_name'] != 'aac':
        raise RuntimeError('Export must contain H.264 video and an AAC audio track')
    seconds = float(probe['format']['duration'])
    if abs(seconds - expected_duration) > 0.15:
        raise RuntimeError('Export duration differs from the source timeline')
    if abs(float(video.get('duration', seconds)) - float(audio.get('duration', seconds))) > 0.15:
        raise RuntimeError('Export audio and video durations do not match')
    result = subprocess.run(
        ['ffmpeg', '-v', 'info', '-i', str(path), '-vn', '-af', 'volumedetect', '-f', 'null', '-'],
        capture_output=True,
        text=True,
    )
    peak = re.search(r'max_volume: ([\-\w.]+) dB', result.stderr)
    if result.returncode or not peak:
        raise RuntimeError('Could not decode and measure exported audio')
    peak_db = float(peak.group(1))
    if not math.isfinite(peak_db) or peak_db < -65:
        raise RuntimeError('Exported speech is silent or inaudibly quiet')
    return {
        'duration': seconds,
        'video_codec': 'h264',
        'audio_codec': 'aac',
        'audio_peak_db': peak_db,
        'width': video['width'],
        'height': video['height'],
    }
