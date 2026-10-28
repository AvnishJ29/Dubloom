"""Per-frame face analysis with InsightFace (SCRFD detector, 106-point landmarks, ArcFace identity).

Runs inside .venv-latentsync. For every frame of a (25 fps) video it records each
face's box, score, 5 key points, 106 landmarks, identity embedding and a small
grey mouth crop, plus a grey thumbnail of the whole frame for cut detection.

    python face_worker.py VIDEO.mp4 OUT.npz --root models/insightface --device cuda|cpu [--det-size 640]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

MOUTH_W, MOUTH_H = 48, 24
THUMB_W, THUMB_H = 32, 18


def mouth_crop(gray: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Axis-aligned crop around the mouth corners (key points 3 and 4), resized to 48x24."""
    left, right = kps[3], kps[4]
    center = (left + right) / 2
    width = max(8.0, float(np.linalg.norm(right - left)) * 1.7)
    height = width * 0.6
    x1, x2 = int(center[0] - width / 2), int(center[0] + width / 2)
    y1, y2 = int(center[1] - height * 0.45), int(center[1] + height * 0.55)
    h, w = gray.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((MOUTH_H, MOUTH_W), np.uint8)
    return cv2.resize(gray[y1:y2, x1:x2], (MOUTH_W, MOUTH_H), interpolation=cv2.INTER_AREA)


def main():
    """Analyse frames [--first, --last) of VIDEO and write one row per face to OUT.npz.

    Also stores a thumbnail of every frame read, `frame_count` (index after the last frame read)
    and `first`, so parallel workers' outputs can be merged.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="analyse every Nth frame and repeat it in between (CPU)",
    )
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 4))
    parser.add_argument(
        "--first",
        type=int,
        default=0,
        help="first frame to analyse (parallel workers split the video)",
    )
    parser.add_argument(
        "--last", type=int, default=-1, help="frame after the last one to analyse (-1: to the end)"
    )
    args = parser.parse_args()
    stride = max(1, args.stride)

    if args.device == "cuda":
        # onnxruntime-gpu loads the CUDA / cuDNN libraries that PyTorch installed (nvidia-* wheels).
        try:
            import torch  # noqa: F401
            import onnxruntime

            if hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()
        except Exception as exc:
            print(f"CUDA libraries for onnxruntime not preloaded: {exc}", flush=True)
    import onnxruntime
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    # The models run one after another: bounded pools that sleep instead of spin, or they thrash each other.
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = max(1, args.threads)
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    cv2.setNumThreads(2)
    app = FaceAnalysis(
        name="buffalo_l",
        root=args.root,
        providers=providers,
        sess_options=options,
        allowed_modules=["detection", "landmark_2d_106", "recognition", "genderage"],
    )
    app.prepare(ctx_id=0 if args.device == "cuda" else -1, det_size=(args.det_size, args.det_size))

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = {
        k: [] for k in ("frame", "bbox", "score", "kps", "lmk", "emb", "mouth", "gender", "age")
    }
    thumbs = []
    last_rows: list[int] = []
    index = 0
    started = time.time()
    while index < args.first:  # frames before this worker's share: demux only
        if not capture.grab():
            break
        index += 1
    while args.last < 0 or index < args.last:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumbs.append(cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA))
        if (index - args.first) % stride:
            # Between analysed frames, repeat the previous frame's faces (CPU speed mode).
            for key in rows:
                rows[key].extend(rows[key][i] if key != "frame" else index for i in last_rows)
            index += 1
            continue
        last_rows = []
        for face in app.get(frame):
            if (
                float(face.det_score) < args.min_score
                or getattr(face, "landmark_2d_106", None) is None
            ):
                continue
            last_rows.append(len(rows["frame"]))
            rows["frame"].append(index)
            rows["bbox"].append(face.bbox.astype(np.float32))
            rows["score"].append(float(face.det_score))
            rows["kps"].append(face.kps.astype(np.float32))
            rows["lmk"].append(face.landmark_2d_106.astype(np.float32))
            embedding = getattr(face, "normed_embedding", None)
            rows["emb"].append(
                (embedding if embedding is not None else np.zeros(512)).astype(np.float16)
            )
            rows["mouth"].append(mouth_crop(gray, face.kps))
            rows["gender"].append(
                int(face.gender) if getattr(face, "gender", None) is not None else -1
            )  # 1 male, 0 female
            rows["age"].append(int(face.age) if getattr(face, "age", None) is not None else -1)
        index += 1
        if index % 50 == 0:
            print(
                f"PROGRESS {index - args.first} {(args.last if args.last >= 0 else max(total, index)) - args.first}",
                flush=True,
            )
    capture.release()
    count = len(rows["frame"])
    np.savez_compressed(
        args.output,
        frame_count=index,
        first=args.first,
        fps=fps,
        width=width,
        height=height,
        thumbs=np.asarray(thumbs, np.uint8).reshape(-1, THUMB_H, THUMB_W),
        frame=np.asarray(rows["frame"], np.int32),
        bbox=np.asarray(rows["bbox"], np.float32).reshape(count, 4),
        score=np.asarray(rows["score"], np.float32),
        kps=np.asarray(rows["kps"], np.float32).reshape(count, 5, 2),
        lmk=np.asarray(rows["lmk"], np.float32).reshape(count, 106, 2),
        emb=np.asarray(rows["emb"], np.float16).reshape(count, 512),
        mouth=np.asarray(rows["mouth"], np.uint8).reshape(count, MOUTH_H, MOUTH_W),
        gender=np.asarray(rows["gender"], np.int8),
        age=np.asarray(rows["age"], np.int16),
    )
    print(f"PROGRESS {index - args.first} {index - args.first}", flush=True)
    print(f"faces {count} in {index - args.first} frames, {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
