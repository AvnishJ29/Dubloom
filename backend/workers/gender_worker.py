"""Speaker gender from the voice itself, with audeering's wav2vec2 age-gender model (female / male / child).

    python gender_worker.py AUDIO.wav TURNS.json OUT.json [--model REPO_OR_DIR] [--device cuda]

TURNS.json maps speaker -> [[start, end], ...] in seconds. Up to MAX_SECONDS of each speaker's longest turns
are scored in windows, and the window probabilities are averaged by length. Runs in the LatentSync venv.
"""

import argparse
import os
import json
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

SAMPLE_RATE = 16000
MAX_SECONDS = 60.0  # per speaker
WINDOW = 8.0  # seconds per model call
MIN_CLIP = 0.4
LABELS = ("female", "male", "child")


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        return torch.softmax(self.gender(hidden), dim=1)


def resolve(model: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:  # noqa: BLE001  (not cached yet: fetch once)
        return snapshot_download(model)


def load_model(model: str, device: str):
    path = resolve(model)
    processor = Wav2Vec2Processor.from_pretrained(path)
    net = AgeGenderModel.from_pretrained(path).to(device).eval()
    return processor, net


def load_audio(path: str) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate == SAMPLE_RATE:
        return samples.mean(axis=1)
    import subprocess

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


@torch.no_grad()
def gender_probs(processor, net, samples: np.ndarray, device: str) -> dict:
    """Length-weighted mean of the window probabilities over `samples` (16 kHz)."""
    total = np.zeros(3)
    weight = 0.0
    step = int(WINDOW * SAMPLE_RATE)
    for start in range(0, len(samples), step):
        piece = samples[start : start + step]
        if len(piece) < MIN_CLIP * SAMPLE_RATE:
            continue
        inputs = processor(piece, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(
            device
        )
        total += net(inputs)[0].cpu().numpy() * len(piece)
        weight += len(piece)
    if not weight:
        return {}
    probs = total / weight
    result = {label: round(float(p), 4) for label, p in zip(LABELS, probs)}
    adult = probs[0] + probs[1]
    result["gender"] = "female" if probs[0] >= probs[1] else "male"
    result["confidence"] = round(float(max(probs[0], probs[1]) / adult), 4) if adult > 0 else 0.0
    result["seconds"] = round(weight / SAMPLE_RATE, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("turns")
    parser.add_argument("output")
    parser.add_argument("--model", default="audeering/wav2vec2-large-robust-6-ft-age-gender")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (
        args.device == "cuda" and not torch.cuda.is_available()
    ):  # a wrong GPU slot must not become a silent CPU render
        raise SystemExit(
            f"cuda requested but no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    device = args.device
    processor, net = load_model(args.model, device)
    samples = load_audio(args.audio)
    turns = json.load(open(args.turns, encoding="utf-8"))
    result = {}
    for speaker, spans in turns.items():
        pieces, collected, per_turn = [], 0.0, []
        for start, end in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
            clip = samples[max(0, int(start * SAMPLE_RATE)) : max(0, int(end * SAMPLE_RATE))]
            if len(clip) < MIN_CLIP * SAMPLE_RATE:
                continue
            verdict = gender_probs(processor, net, clip, device)
            per_turn.append(
                {
                    "start": start,
                    "end": end,
                    "female": verdict["female"],
                    "male": verdict["male"],
                    "gender": verdict["gender"],
                    "confidence": verdict["confidence"],
                }
            )
            if collected < MAX_SECONDS:
                pieces.append(clip)
                collected += len(clip) / SAMPLE_RATE
        result[speaker] = (
            gender_probs(processor, net, np.concatenate(pieces), device) if pieces else {}
        )
        result[speaker]["turns"] = per_turn
        print(
            f"{speaker}: {result[speaker].get('gender')} {result[speaker].get('confidence')} over {len(per_turn)} turns",
            flush=True,
        )
    json.dump(result, open(args.output, "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
