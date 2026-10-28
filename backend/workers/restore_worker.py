"""Restore the redrawn lower face with GFPGAN v1.4 after LatentSync.

    python restore_worker.py LIPS.mp4 TARGETS.npz OUT.mp4 --weights models/gfpgan/GFPGANv1.4.pth [--min-face 60] [--max-face 200]

The model paints a 512 px face; GFPGAN gives it back real lip, teeth and skin detail. Only the lower face
of the redrawn frames is replaced (feathered), so eyes and everything else stay the original picture.
Runs in .venv-restore (GFPGAN's basicsr/facexlib stack must not touch the LatentSync environment).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch


def lower_face(bbox, width, height):
    """Lower-face box (x1, y1, x2, y2) of a face `bbox`: 10 % wider a side, clipped to the frame."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - 0.1 * w)),
        max(0, int(y1 + 0.45 * h)),
        min(width, int(x2 + 0.1 * w)),
        min(height, int(y2 + 0.12 * h)),
    )


def smooth_detail(frame, restored, memory, share, index, track_id, centre):
    """The restored frame with its added detail (restored - raw) blended with the previous frame's, shifted by the face's
    motion. Motion comes from the raw frame, only the hallucinated detail is smoothed, so lips move freely without flicker.
    """
    delta = restored.astype(np.float32) - frame.astype(np.float32)
    if (
        share > 0
        and memory["delta"] is not None
        and index == memory["index"] + 1
        and track_id == memory["track"]
    ):
        dx, dy = int(round(centre[0] - memory["centre"][0])), int(
            round(centre[1] - memory["centre"][1])
        )
        previous = np.roll(np.roll(memory["delta"], dy, axis=0), dx, axis=1)
        delta = (1 - share) * delta + share * previous
    memory.update(delta=delta if share > 0 else None, index=index, track=track_id, centre=centre)
    return frame.astype(np.float32) + delta


def restore_file(restorer, source, output, first, bbox, track, args):
    """Restore the lower faces of one video file whose frame 0 is frame `first` of the targets. Returns frames restored."""
    capture = cv2.VideoCapture(source)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "faster",
            "-crf",
            "8",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        stdin=subprocess.PIPE,
    )
    offset = done = 0
    memory = {
        "delta": None,
        "index": -9,
        "track": None,
        "centre": (0, 0),
    }  # last frame's restoration detail, for temporal smoothing
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index = first + offset
        face_px = bbox[index, 2] - bbox[index, 0] if index < len(track) else 0
        if (
            index < len(track)
            and track[index] >= 0
            and args.min_face <= face_px
            and (not args.max_face or face_px < args.max_face)
        ):
            x1, y1, x2, y2 = bbox[index]
            w, h = x2 - x1, y2 - y1
            cx1, cy1, cx2, cy2 = (
                max(0, int(x1 - 0.6 * w)),
                max(0, int(y1 - 0.6 * h)),
                min(width, int(x2 + 0.6 * w)),
                min(height, int(y2 + 0.6 * h)),
            )
            crop = frame[cy1:cy2, cx1:cx2]
            try:
                _, _, restored = restorer.enhance(
                    crop, has_aligned=False, only_center_face=True, paste_back=True
                )
            except Exception:  # noqa: BLE001  (no face found in the crop: keep the frame)
                restored = None
            if restored is not None and restored.shape == crop.shape:
                mx1, my1, mx2, my2 = lower_face(bbox[index], width, height)
                mask = np.zeros((height, width), np.float32)
                cv2.rectangle(
                    mask,
                    (mx1 + int(0.06 * (mx2 - mx1)), my1 + int(0.06 * (my2 - my1))),
                    (mx2 - int(0.06 * (mx2 - mx1)), my2 - int(0.06 * (my2 - my1))),
                    1,
                    -1,
                )
                mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, w * 0.04))[..., None]
                full = frame.copy()
                full[cy1:cy2, cx1:cx2] = restored
                full = smooth_detail(
                    frame,
                    full,
                    memory,
                    args.temporal,
                    index,
                    int(track[index]),
                    ((x1 + x2) / 2, (y1 + y2) / 2),
                )
                frame = np.clip(full * mask + frame.astype(np.float32) * (1 - mask), 0, 255).astype(
                    np.uint8
                )
                done += 1
        writer.stdin.write(frame.tobytes())
        offset += 1
    capture.release()
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError(f"ffmpeg could not write the restored video {output}")
    return done


def main():
    """Restore one LatentSync output, or every span file listed by --batch with one model load."""
    parser = argparse.ArgumentParser()
    parser.add_argument("lips", nargs="?")
    parser.add_argument("targets", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument(
        "--batch",
        help="JSON list of {input, output, first}: several span files restored with one model load",
    )
    parser.add_argument("--targets-file", help="targets.npz for --batch")
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--min-face",
        type=int,
        default=60,
        help="skip faces narrower than this: webcam insets of ~80 px still gain real lip detail",
    )
    parser.add_argument(
        "--max-face",
        type=int,
        default=0,
        help="skip faces this wide or wider (0 = no limit): close-ups keep their own texture",
    )
    parser.add_argument(
        "--temporal",
        type=float,
        default=0.0,
        help="0..0.9: share of the previous frame's restored detail carried over (less flicker)",
    )
    args = parser.parse_args()
    jobs = (
        json.load(open(args.batch))
        if args.batch
        else [{"input": args.lips, "output": args.output, "first": 0}]
    )
    targets = args.targets_file or args.targets
    if not jobs or not targets or any(not j.get("input") or not j.get("output") for j in jobs):
        parser.error("give LIPS TARGETS OUTPUT, or --batch jobs.json --targets-file targets.npz")
    from gfpgan import GFPGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):  # a wrong GPU slot must not become a silent CPU pass
        raise SystemExit(
            f"no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r})"
        )
    restorer = GFPGANer(
        model_path=args.weights,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=device,
    )
    data = np.load(targets)
    bbox, track = data["bbox"], data["track"]
    started = time.time()
    done = 0
    for number, job in enumerate(jobs, 1):
        done += restore_file(
            restorer, job["input"], job["output"], int(job.get("first", 0)), bbox, track, args
        )
        print(
            f"Restored {done} frames ({time.time() - started:.0f}s) file {number} of {len(jobs)}",
            flush=True,
        )
    print(f"DONE restored={done} files={len(jobs)} seconds={time.time() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
