"""Voices served through OpenRouter: Gemini TTS stock voices, and the original speakers' cloned voices."""

from __future__ import annotations

import base64
import subprocess
import wave
from pathlib import Path

from backend.providers import openrouter
from backend.config import settings
from backend.speech.styles import style_prompt, style_tag
from backend.speech.tts import split_for_tts

GEMINI_VOICES = {
    "female": ["Kore", "Aoede", "Leda", "Zephyr", "Despina", "Callirrhoe", "Sulafat", "Erinome"],
    "male": ["Charon", "Puck", "Orus", "Iapetus", "Fenrir", "Enceladus", "Algieba", "Rasalgethi"],
}
OPENROUTER_LANGUAGES = {
    "hi",
    "bn",
    "ta",
    "te",
    "mr",
    "gu",
    "kn",
    "ml",
    "pa",
    "ur",
    "en",
    "ne",
    "as",
    "or",
}
PCM_RATE = 24000
GAP_SECONDS = 0.14


def _write_pcm(chunks: list[bytes], output: Path):
    silence = b"\x00" * (int(PCM_RATE * GAP_SECONDS) * 2)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(PCM_RATE)
        for index, chunk in enumerate(chunks):
            if index:
                target.writeframes(silence)
            target.writeframes(chunk[: len(chunk) - len(chunk) % 2])


def gemini_synthesize(text: str, voice: str, output: Path, style: str | None = None, usage=None):
    """Speak `text` with a Gemini TTS stock voice; writes a 24 kHz mono WAV to `output`.

    Long text is voiced in pieces joined by short pauses, and `style` becomes a spoken direction
    in the prompt. A voice the model rejects falls back to Kore. Raises RuntimeError when a piece
    comes back without audio.
    """
    direction = style_prompt(style)
    chunks = []
    for piece in split_for_tts(text, max_chars=420, max_sentences=4):
        prompt = f"Say the following in a {direction} tone: {piece}" if direction else piece
        body = {
            "model": settings.openrouter_tts_model,
            "input": prompt,
            "voice": voice,
            "response_format": "pcm",
        }
        try:
            audio, _ = openrouter.post_audio("/audio/speech", body)
        except openrouter.OpenRouterError as exc:
            if "voice" not in str(exc).lower():
                raise
            audio, _ = openrouter.post_audio("/audio/speech", {**body, "voice": "Kore"})
        if len(audio) < 2000:
            raise RuntimeError("Gemini TTS returned no audio")
        chunks.append(audio)
        if usage is not None:
            usage.add("openrouter_tts_characters", len(piece))
    _write_pcm(chunks, output)


def clone_synthesize(
    text: str,
    reference: Path,
    output: Path,
    style: str | None = None,
    reference_text: str | None = None,
    usage=None,
):
    """Speak `text` in the voice heard in `reference` (a short, clean WAV of the speaker)."""
    sample = "data:audio/wav;base64," + base64.b64encode(reference.read_bytes()).decode()
    references = [{"type": "input_audio", "input_audio": {"data": sample}}]
    if reference_text:
        references.append({"type": "text", "text": reference_text[:600]})
    tag = style_tag(style)
    parts = []
    for index, piece in enumerate(split_for_tts(text, max_chars=380, max_sentences=4)):
        body = {
            "model": settings.openrouter_clone_model,
            "input": f"{tag} {piece}" if tag else piece,
            "response_format": "mp3",
            "input_references": references,
        }
        audio, _ = openrouter.post_audio("/audio/speech", body)
        if len(audio) < 1000:
            raise RuntimeError("The cloning voice returned no audio")
        part = output.with_name(f"{output.stem}.clone{index}.mp3")
        part.write_bytes(audio)
        parts.append(part)
        if usage is not None:
            usage.add("openrouter_tts_characters", len(piece))
    listing = output.with_name(f"{output.stem}.clone.txt")
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
            str(PCM_RATE),
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
        raise RuntimeError("Could not decode cloned speech: " + result.stderr[-300:])


def build_reference(
    source: Path, start: float, end: float, output: Path, max_seconds: float = 14.0
) -> Path:
    """Cut a clean mono sample of one speaker for cloning."""
    length = max(1.0, min(max_seconds, end - start))
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(PCM_RATE),
            "-af",
            "highpass=f=70,loudnorm=I=-20:TP=-2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or not output.is_file():
        raise RuntimeError("Could not prepare the voice reference: " + result.stderr[-200:])
    return output
