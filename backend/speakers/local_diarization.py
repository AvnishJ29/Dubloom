"""Local speaker segmentation and CAM++ clustering using sherpa-onnx.

After diarization two refinements make the turns usable as dubbing slots:
boundaries are snapped to the actual speech onset/offset (so a dubbed line
starts when the person starts talking), and speakers whose voice embeddings
are near-identical are merged (clustering sometimes splits one person into two).
"""

from functools import lru_cache
from pathlib import Path
import wave

import numpy as np

from backend.config import settings
from backend.models import Segment

RATE = 16000
MERGE_SIMILARITY = 0.72  # cosine between speaker centroids above which they are one person
ABSORB_SIMILARITY = 0.5  # a tiny speaker joins its closest speaker above this similarity
TINY_SHARE = 0.03  # ... when it holds less than 3 % of the speech


def _models_dir():
    return settings.jobs_dir / 'models' / 'sherpa'


@lru_cache(maxsize=4)
def _pipeline(num_clusters: int):
    """sherpa-onnx diarization pipeline for a cluster count (-1: decided by threshold)."""
    import sherpa_onnx

    folder = _models_dir()
    segmentation = folder / 'sherpa-onnx-pyannote-segmentation-3-0/model.onnx'
    embedding = folder / '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx'
    if not segmentation.is_file() or not embedding.is_file():
        raise RuntimeError(
            'Local diarization weights are missing; run `python -m backend.speakers.setup_local_models`'
        )
    threads = settings.diarization_threads
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)
            ),
            num_threads=threads,
            provider='cpu',
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding), num_threads=threads, provider='cpu'
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.3,
    )
    if not config.validate():
        raise RuntimeError('Invalid local diarization model configuration')
    return sherpa_onnx.OfflineSpeakerDiarization(config)


@lru_cache(maxsize=1)
def _embedder():
    import sherpa_onnx

    model = _models_dir() / '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx'
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=settings.diarization_threads, provider='cpu'
        )
    )


def load_samples(audio: Path) -> np.ndarray:
    """Float samples in [-1, 1) of a 16 kHz mono 16-bit WAV; raises ValueError for other formats."""
    with wave.open(str(audio), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (RATE, 1, 2):
            raise ValueError('Diarization needs a 16 kHz mono PCM WAV')
        return (
            np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32)
            / 32768
        )


def frame_energy(samples: np.ndarray, frame: int = 320) -> np.ndarray:
    """RMS per 20 ms frame."""
    count = len(samples) // frame
    if count == 0:
        return np.zeros(1, dtype=np.float32)
    return np.sqrt(np.mean(samples[: count * frame].reshape(count, frame) ** 2, axis=1))


def snap_boundaries(
    turns: list[Segment],
    samples: np.ndarray,
    pad_start: float = 0.06,
    pad_end: float = 0.10,
    reach: float = 0.35,
) -> list[Segment]:
    """Move each turn's start to the first speech frame near it and its end to the last one."""
    if not turns or samples.size < RATE:
        return turns
    frame = 320
    energy = frame_energy(samples, frame)
    speech = energy > max(0.004, 0.12 * np.percentile(energy, 90))
    seconds_per_frame = frame / RATE
    ordered = sorted(turns, key=lambda t: t.start)
    result = []
    for index, turn in enumerate(ordered):
        lo = max(0.0, turn.start - reach)
        hi = turn.start + 0.8
        if index:
            lo = max(lo, result[-1].end + 0.02)
        a, b = int(lo / seconds_per_frame), min(len(speech), int(hi / seconds_per_frame))
        start = turn.start
        if b > a:
            hits = np.flatnonzero(speech[a:b])
            if hits.size:
                start = max(lo, (a + hits[0]) * seconds_per_frame - pad_start)
        lo_e, hi_e = max(start + 0.2, turn.end - 0.8), turn.end + reach
        if index + 1 < len(ordered):
            hi_e = min(hi_e, ordered[index + 1].start - 0.02)
        a, b = int(lo_e / seconds_per_frame), min(len(speech), int(hi_e / seconds_per_frame))
        end = turn.end
        if b > a:
            hits = np.flatnonzero(speech[a:b])
            if hits.size:
                end = min(hi_e, (a + hits[-1] + 1) * seconds_per_frame + pad_end)
        end = max(end, start + 0.25)
        result.append(Segment(round(start, 3), round(end, 3), turn.speaker))
    return result


def _embedding(samples: np.ndarray, start: float, end: float):
    """CAM++ voice embedding of samples[start:end] (times in seconds)."""
    extractor = _embedder()
    stream = extractor.create_stream()
    stream.accept_waveform(RATE, samples[int(start * RATE) : int(end * RATE)])
    stream.input_finished()
    return np.asarray(extractor.compute(stream), dtype=np.float32)


def consolidate_speakers(
    turns: list[Segment], samples: np.ndarray, num_speakers: int | None = None, embed=_embedding
) -> list[Segment]:
    """Merge speaker labels whose voices are the same person; relabel in first-appearance order."""
    speakers = sorted({t.speaker for t in turns})
    if len(speakers) < 2 or (num_speakers and len(speakers) <= num_speakers):
        return _relabel(turns)
    vectors = {}
    for speaker in speakers:
        own = sorted(
            (t for t in turns if t.speaker == speaker and t.end - t.start >= 1.0),
            key=lambda t: t.end - t.start,
            reverse=True,
        )[:6]
        if own:
            vecs = [embed(samples, t.start, min(t.end, t.start + 12)) for t in own]
            centroid = np.mean(vecs, axis=0)
            vectors[speaker] = centroid / (np.linalg.norm(centroid) + 1e-8)
    share = {s: sum(t.end - t.start for t in turns if t.speaker == s) for s in speakers}
    total = sum(share.values()) or 1.0
    mapping = {s: s for s in speakers}

    def similarity(a, b):
        return float(vectors[a] @ vectors[b]) if a in vectors and b in vectors else -1.0

    changed = True
    while changed:
        changed = False
        labels = sorted({mapping[s] for s in speakers}, key=lambda s: -share[s])
        if num_speakers and len(labels) <= num_speakers:
            break
        best = None
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                sim = similarity(a, b)
                tiny = share[b] / total < TINY_SHARE or share[a] / total < TINY_SHARE
                if sim >= MERGE_SIMILARITY or (tiny and sim >= ABSORB_SIMILARITY):
                    if best is None or sim > best[0]:
                        best = (sim, a, b)
        if best:
            _, keep, drop = best
            if share[drop] > share[keep]:
                keep, drop = drop, keep
            for s in speakers:
                if mapping[s] == drop:
                    mapping[s] = keep
            share[keep] += share[drop]
            share[drop] = 0.0
            changed = True
    return _relabel([Segment(t.start, t.end, mapping[t.speaker]) for t in turns])


def _relabel(turns: list[Segment]) -> list[Segment]:
    """Rename speakers SPEAKER_00, SPEAKER_01, ... in order of first appearance; sorts by start."""
    labels = {}
    result = []
    for turn in sorted(turns, key=lambda t: t.start):
        labels.setdefault(turn.speaker, len(labels))
        result.append(Segment(turn.start, turn.end, f'SPEAKER_{labels[turn.speaker]:02d}'))
    return result


def diarize(audio: Path, progress=None, num_speakers: int | None = None) -> list[Segment]:
    """Diarize a 16 kHz mono WAV on the CPU with sherpa-onnx, then refine the turns.

    `num_speakers` fixes the cluster count when given; `progress(done, total)` receives
    sherpa-onnx's chunk progress. Same-voice speakers are merged and boundaries snapped to speech
    (see the module docstring); turns are labelled SPEAKER_00, SPEAKER_01, ... in order of
    appearance.
    """
    samples = load_samples(audio)
    clusters = int(num_speakers) if num_speakers and num_speakers > 0 else -1
    pipeline = _pipeline(clusters)
    if progress:
        # sherpa-onnx's progress callback must return an int (its value is ignored)
        result = pipeline.process(
            samples, callback=lambda done, total: (progress(done, total), 0)[1]
        )
    else:
        result = pipeline.process(samples)
    turns = [
        Segment(round(turn.start, 3), round(turn.end, 3), f'RAW_{turn.speaker}')
        for turn in result.sort_by_start_time()
    ]
    turns = consolidate_speakers(turns, samples, num_speakers)
    return snap_boundaries(turns, samples)
