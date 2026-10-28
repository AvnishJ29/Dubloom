"""Vocal separation with a BS/Mel-RoFormer model through audio-separator (own venv: .venv-separate).

    python separate_worker.py INPUT.wav OUT_DIR --model model_bs_roformer_ep_317_sdr_12.9755.ckpt --model-dir models/separator

Writes OUT_DIR/<name>_(Vocals)_<model>.wav and OUT_DIR/<name>_(Instrumental)_<model>.wav; the model file is
downloaded into --model-dir on first use. Progress lines look like Demucs' so the studio can show them.
"""

import argparse
import logging
import os
import sys
import time


def main():
    """Split INPUT into vocal and instrumental WAVs in OUT_DIR."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("out_dir")
    parser.add_argument("--model", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    parser.add_argument("--model-dir", default="models/separator")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available() and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):  # a wrong GPU slot must not become a silent CPU run
        raise SystemExit(
            f"no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r})"
        )
    from audio_separator.separator import Separator

    started = time.time()
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=args.model_dir,
        output_dir=args.out_dir,
        output_format="WAV",
    )
    separator.load_model(model_filename=args.model)
    print(
        f"model loaded ({time.time() - started:.0f}s) device={'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    outputs = separator.separate(args.input)
    print(f"DONE seconds={time.time() - started:.1f} outputs={len(outputs)}", flush=True)
    for name in outputs:
        print("  ", name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
