"""Speaker diarization with DiariZen (WavLM-Large EEND + VBx clustering).

Runs inside .venv-diarizen (torch 2.1.1), so it imports nothing from the studio.

    python diarizen_worker.py AUDIO.wav OUT.json --model models/diarizen-v2 \
        --embedding models/pyannote-wespeaker-voxceleb-resnet34-LM/pytorch_model.bin [--speakers N] [--device cuda|cpu]

Writes {"mode": ..., "segments": [{"start", "end", "speaker"}, ...]} with raw,
possibly overlapping turns; the studio post-processes them into dubbing slots.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    """Diarize AUDIO with DiariZen; write raw turns to OUT.json, labelled by first appearance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--speakers", type=int, default=0)
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    if args.device == "cpu":
        torch.cuda.is_available = lambda: False  # the pipeline picks its device itself
    threads = max(1, (torch.get_num_threads() or 4))
    torch.set_num_threads(threads)
    from diarizen.pipelines.inference import DiariZenPipeline

    print("PROGRESS 0 3", flush=True)
    started = time.time()
    pipeline = DiariZenPipeline(
        diarizen_hub=Path(args.model).resolve(), embedding_model=str(Path(args.embedding).resolve())
    )
    if args.speakers > 0:
        pipeline.min_speakers = pipeline.max_speakers = args.speakers
    elif args.max_speakers > 0:
        pipeline.max_speakers = args.max_speakers
    print("PROGRESS 1 3", flush=True)
    annotation = pipeline(str(Path(args.audio).resolve()), sess_name="audio")
    print("PROGRESS 2 3", flush=True)
    labels = {}
    segments = []
    for turn, _, speaker in sorted(
        annotation.itertracks(yield_label=True), key=lambda item: item[0].start
    ):
        index = labels.setdefault(speaker, len(labels))
        segments.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": f"SPEAKER_{index:02d}",
            }
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output).write_text(
        json.dumps(
            {
                "mode": f"DiariZen WavLM-Large (v2) + VBx on {device}",
                "segments": segments,
                "seconds": round(time.time() - started, 1),
            }
        ),
        encoding="utf-8",
    )
    print("PROGRESS 3 3", flush=True)


if __name__ == "__main__":
    sys.exit(main())
