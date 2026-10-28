"""Ungated diarization: Silero speech regions + SpeechBrain ECAPA embeddings."""

from functools import lru_cache
from pathlib import Path
import wave

import numpy as np

from backend.config import settings
from backend.models import Segment


@lru_cache(maxsize=1)
def _models():
    from silero_vad import load_silero_vad
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    cache = settings.jobs_dir / "models" / "speechbrain-ecapa"
    cache.parent.mkdir(parents=True, exist_ok=True)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache),
        local_strategy=LocalStrategy.COPY,
    )
    return load_silero_vad(), encoder


def _samples(audio: Path):
    import torch

    with wave.open(str(audio), "rb") as stream:
        rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
        pcm = stream.readframes(stream.getnframes())
    if (rate, channels, width) != (16000, 1, 2):
        raise RuntimeError("Speaker diarization requires 16 kHz, mono, 16-bit WAV")
    return torch.from_numpy(np.frombuffer(pcm, dtype="<i2").copy()).float().div_(32768)


def _windows(start: float, end: float):
    """Embedding windows of a speech region: the region itself up to 3.5 s, else 3 s every 1.5 s."""
    if end - start <= 3.5:
        return [(start, end)]
    width, step = 3.0, 1.5
    positions = []
    cursor = start
    while cursor + width < end:
        positions.append((cursor, cursor + width))
        cursor += step
    if not positions or end - positions[-1][1] > 0.25:
        positions.append((max(start, end - width), end))
    return positions


def diarize(audio: Path) -> list[Segment]:
    """Diarize a 16 kHz mono 16-bit WAV with Silero VAD and SpeechBrain ECAPA embeddings.

    Speech regions are embedded in 3 s windows (1.5 s hop); windows of at least 2 s are clustered
    (cosine distance 0.45) and shorter ones join the nearest cluster. Within a region the speaker
    changes halfway between window midpoints. Same-speaker turns less than 0.4 s apart are joined
    (up to 25 s). Labels are SPEAKER_00, SPEAKER_01, ... in order of appearance.
    """
    import torch
    from silero_vad import get_speech_timestamps
    from sklearn.cluster import AgglomerativeClustering

    vad, encoder = _models()
    samples = _samples(audio)
    regions = get_speech_timestamps(
        samples,
        vad,
        sampling_rate=16000,
        return_seconds=True,
        min_speech_duration_ms=350,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )
    spans = []
    vectors = []
    with torch.inference_mode():
        for region_index, region in enumerate(regions):
            for start, end in _windows(region["start"], region["end"]):
                clip = samples[round(start * 16000) : round(end * 16000)]
                if clip.numel() < 16000:
                    clip = torch.nn.functional.pad(clip, (0, 16000 - clip.numel()))
                embedding = encoder.encode_batch(clip.unsqueeze(0), normalize=True)
                vectors.append(embedding.detach().cpu().numpy().reshape(-1))
                spans.append((region_index, start, end))
    if not spans:
        return []
    reliable = [i for i, (_, start, end) in enumerate(spans) if end - start >= 2.0]
    if not reliable:
        reliable = list(range(len(spans)))
    if len(reliable) == 1:
        main_labels = [0]
    else:
        main_labels = (
            AgglomerativeClustering(
                n_clusters=None, distance_threshold=0.45, metric="cosine", linkage="average"
            )
            .fit_predict(np.stack([vectors[i] for i in reliable]))
            .tolist()
        )
    labels = [None] * len(spans)
    for index, label in zip(reliable, main_labels):
        labels[index] = label
    centroids = {
        label: np.mean([vectors[i] for i in reliable if labels[i] == label], axis=0)
        for label in set(main_labels)
    }
    for index in range(len(spans)):
        if labels[index] is None:
            # Very short utterances yield unstable embeddings. Attach them to the
            # closest established voice rather than inventing a new speaker.
            labels[index] = min(
                centroids,
                key=lambda label: 1
                - np.dot(vectors[index], centroids[label])
                / (np.linalg.norm(vectors[index]) * np.linalg.norm(centroids[label]) + 1e-8),
            )
    stable = {}
    labels = [stable.setdefault(label, len(stable)) for label in labels]
    turns = []
    for region_index, region in enumerate(regions):
        local = [
            (start, end, labels[i])
            for i, (idx, start, end) in enumerate(spans)
            if idx == region_index
        ]
        if not local:
            continue
        # Window midpoints yield one continuous speaker timeline within a speech region.
        cursor = region["start"]
        for first, second in zip(local, local[1:]):
            if first[2] != second[2]:
                boundary = (0.5 * (first[0] + first[1]) + 0.5 * (second[0] + second[1])) / 2
                turns.append(
                    Segment(round(cursor, 3), round(boundary, 3), f"SPEAKER_{first[2]:02d}")
                )
                cursor = boundary
        turns.append(
            Segment(round(cursor, 3), round(region["end"], 3), f"SPEAKER_{local[-1][2]:02d}")
        )
    merged = []
    for turn in turns:
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end < 0.4
            and turn.end - merged[-1].start <= 25
        ):
            merged[-1].end = turn.end
        elif turn.end - turn.start >= 0.15:
            merged.append(turn)
    return merged
