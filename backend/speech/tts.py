"""Bodhan Indic-Speak text-to-speech.

The API reads a sentence or two per request (about 30 seconds of speech) and
does not split longer text, so a line is chunked at sentence boundaries and the
WAV pieces are joined. The language and an optional delivery style go in the
`instructions` JSON string, exactly as documented.
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import wave
from pathlib import Path

from backend.providers.bodhan_api import BodhanAPI
from backend.config import settings
from backend.media.audio_norm import level_clip
from backend.voices.library import speaking_rate

logger = logging.getLogger(__name__)
from backend.speech.styles import bodhan_style

SENTENCE_BREAK = re.compile(r"(?<=[\u0964\u0965.!?\u061f\u06d4])\s+")
SOFT_BREAK = re.compile(r"(?<=[,;:\u060c])\s+")
MAX_CHUNK_CHARS = 220
CHUNK_GAP_SECONDS = 0.14


def split_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS, max_sentences: int = 2) -> list[str]:
    """Split text into pieces of at most `max_sentences` sentences and about `max_chars` characters."""
    text = " ".join((text or "").split())
    if not text:
        return []
    pieces: list[str] = []
    for sentence in SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        # An over-long sentence: break at commas, then at spaces as a last resort.
        current = ""
        for part in SOFT_BREAK.split(sentence):
            while len(part) > max_chars:
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 3 else max_chars
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(part[:cut].strip())
                part = part[cut:].strip()
            if current and len(current) + 1 + len(part) > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            pieces.append(current)
    chunks: list[str] = []
    count = 0
    for piece in pieces:
        if chunks and count < max_sentences and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
            count += 1
        else:
            chunks.append(piece)
            count = 1
    return chunks


def join_wavs(blobs: list[bytes], output: Path, gap_seconds: float = CHUNK_GAP_SECONDS):
    """Concatenate WAV byte strings (same format) with a short pause between them."""
    try:
        frames, params = [], None
        for blob in blobs:
            with wave.open(io.BytesIO(blob), "rb") as stream:
                if params is None:
                    params = stream.getparams()
                elif (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (
                    params.framerate,
                    params.nchannels,
                    params.sampwidth,
                ):
                    raise wave.Error("mismatched formats")
                frames.append(stream.readframes(stream.getnframes()))
        silence = b"\x00" * (
            int(params.framerate * gap_seconds) * params.nchannels * params.sampwidth
        )
        with wave.open(str(output), "wb") as target:
            target.setnchannels(params.nchannels)
            target.setsampwidth(params.sampwidth)
            target.setframerate(params.framerate)
            for index, chunk in enumerate(frames):
                if index:
                    target.writeframes(silence)
                target.writeframes(chunk)
    except (wave.Error, EOFError, AttributeError):
        # Unusual WAV flavour: let ffmpeg normalise and join the pieces.
        parts = []
        for index, blob in enumerate(blobs):
            part = output.with_name(f"{output.stem}.part{index}.wav")
            part.write_bytes(blob)
            parts.append(part)
        listing = output.with_name(f"{output.stem}.parts.txt")
        listing.write_text("\n".join(f"file '{part.name}'" for part in parts), encoding="utf-8")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        for part in parts + [listing]:
            part.unlink(missing_ok=True)
        if result.returncode:
            raise RuntimeError("Could not join synthesized audio: " + result.stderr[-300:])


def clip_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() / max(1, reader.getframerate())
    except (wave.Error, EOFError, OSError):
        return 0.0


def synthesize(
    api: BodhanAPI,
    text: str,
    language: str,
    voice: str,
    output: Path,
    style: str | None = None,
    usage=None,
):
    """Synthesize Bodhan speech in chunks and write the resulting audio to output."""
    instructions = {"lang": language}
    native_style = bodhan_style(style)
    if native_style:
        instructions["style"] = native_style
    spoken = re.sub(
        r"(?<=[A-Za-z])(?=\d)", " ", text
    )  # W1 -> W 1: a letter and a digit are two spoken tokens
    chunks = split_for_tts(spoken)
    if not chunks:
        raise RuntimeError("There is no text to voice")

    def request(destination: Path):
        blobs = []
        for chunk in chunks:
            response = api.post(
                "/v1/audio/speech",
                settings.bodhan_tts_key,
                json={
                    "model": "indic-speak",
                    "input": chunk,
                    "voice": voice,
                    "instructions": json.dumps(instructions, ensure_ascii=False),
                },
            )
            if "audio" not in response.headers.get("content-type", ""):
                raise RuntimeError("Bodhan TTS did not return audio")
            blobs.append(response.content)
            if usage is not None:
                usage.add("bodhan_tts_characters", len(chunk))
        if len(blobs) == 1:
            destination.write_bytes(blobs[0])
            level_clip(
                destination
            )  # same loudness and peak for every line, whatever the voice or style
        else:
            join_wavs(blobs, destination)

    request(output)
    # A clip far longer than the voice's own speaking rate predicts is the engine looping on a token (seen on lines of
    # bare symbols: 10 s for 23 characters). One more take, keeping the shorter one.
    expected = len(spoken) / max(6.0, speaking_rate(voice, 12.0))
    seconds = clip_seconds(output)
    if len(spoken) >= 8 and seconds > max(2.5 * expected, 1.5):
        logger.warning(
            "Bodhan TTS clip for %r is %.1fs, about %.1fs expected: taking it again",
            spoken[:40],
            seconds,
            expected,
        )
        retry = output.with_name(output.stem + ".retry.wav")
        request(retry)
        if clip_seconds(retry) < seconds:
            retry.replace(output)
        else:
            retry.unlink(missing_ok=True)
