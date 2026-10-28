"""Hands per frame with MediaPipe Hands, so a mouth covered by a hand is never redrawn.

    python hands_worker.py VIDEO_25FPS.mp4 OUT.npz [--stride 1]

Writes frame indices and hand boxes (x1, y1, x2, y2 in pixels) for every detected hand. Runs in its own venv (.venv-hands): MediaPipe's protobuf must not touch the LatentSync environment.
"""

import argparse
import sys
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--frames", help="npy of frame indices to search (frames with a face); others are skipped"
    )
    args = parser.parse_args()
    wanted = set(np.load(args.frames).tolist()) if args.frames else None
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    capture = cv2.VideoCapture(args.video)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    frames, boxes = [], []
    index, started = 0, time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % args.stride == 0 and (wanted is None or index in wanted):
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for hand in result.multi_hand_landmarks or []:
                xs = [lm.x * width for lm in hand.landmark]
                ys = [lm.y * height for lm in hand.landmark]
                frames.append(index)
                boxes.append(
                    [
                        max(0.0, min(xs)),
                        max(0.0, min(ys)),
                        min(width, max(xs)),
                        min(height, max(ys)),
                    ]
                )
        index += 1
        if index % 200 == 0:
            print(f"PROGRESS {index}", flush=True)
    capture.release()
    np.savez_compressed(
        args.output,
        frame_count=index,
        frame=np.asarray(frames, np.int32),
        bbox=np.asarray(boxes, np.float32).reshape(-1, 4),
        stride=args.stride,
    )
    print(f"hands {len(frames)} in {index} frames, {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
