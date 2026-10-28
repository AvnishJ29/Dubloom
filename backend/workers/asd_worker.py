"""Active speaker detection with Light-ASD (CVPR 2023, trained on AVA-ActiveSpeaker movies).

Runs inside .venv-latentsync. For every face track (detections linked by the
studio inside one shot) it scores each frame: is this face speaking the audio
heard at that moment? Preprocessing and scoring follow Light-ASD's own
inference (Columbia_test.py): median-smoothed boxes, the mouth-centred 112 px
grayscale crop at 25 fps, 13 MFCCs at 100 Hz from int16 audio, and scores
averaged over windows of 1-6 seconds. A score >= 0 means speaking.

    python asd_worker.py VIDEO_25FPS.mp4 AUDIO_16K.wav FACES.npz TRACKS.npz OUT.npz --repo third_party/Light-ASD
"""

import argparse
import math
import os
import sys
import time
import wave

import cv2
import numpy as np

MIN_TRACK = 10  # Light-ASD ignores shorter tracks
CROP_SCALE = 0.40
DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)  # seconds; "more reliable" setting of Light-ASD
FPS = 25


def read_int16(path):
    """int16 samples of a 16 kHz 16-bit WAV, channels averaged; exits on any other format."""
    with wave.open(path, "rb") as stream:
        if stream.getframerate() != 16000 or stream.getsampwidth() != 2:
            raise SystemExit("audio must be 16 kHz 16-bit PCM")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        if stream.getnchannels() > 1:
            samples = samples.reshape(-1, stream.getnchannels()).mean(axis=1).astype(np.int16)
    return samples


def main():
    """Score every face track and write OUT.npz.

    `score` holds one Light-ASD score per detection of FACES.npz (TRACKS.npz gives each
    detection's track); detections in tracks shorter than MIN_TRACK frames stay NaN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("audio")
    parser.add_argument("faces")
    parser.add_argument("tracks")
    parser.add_argument("output")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", default="weight/pretrain_AVA_CVPR.model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    video, audio_path, faces_path, tracks_path, output = (
        os.path.abspath(p) for p in (args.video, args.audio, args.faces, args.tracks, args.output)
    )
    os.chdir(args.repo)
    sys.path.insert(0, os.getcwd())
    import python_speech_features
    import torch
    from scipy import signal
    from model.Model import ASD_Model

    torch.set_grad_enabled(False)
    if (
        args.device == "cuda" and not torch.cuda.is_available()
    ):  # a wrong GPU slot must not become a silent CPU render
        raise SystemExit(
            f"cuda requested but no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    device = torch.device(args.device)
    model = ASD_Model()
    head = torch.nn.Linear(128, 2)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
    head.load_state_dict({k[10:]: v for k, v in state.items() if k.startswith("lossAV.FC.")})
    model.eval().to(device)
    head.eval().to(device)

    faces = np.load(faces_path)
    frames, boxes = faces["frame"].astype(np.int64), faces["bbox"].astype(np.float64)
    track_of = np.load(tracks_path)["track"]
    scores = np.full(len(frames), np.nan, dtype=np.float32)
    started = time.time()

    # ---- tracks: contiguous frames (missed detections interpolated), smoothed crop boxes ----
    plans = []
    for tid in np.unique(track_of[track_of >= 0]):
        rows = np.where(track_of == tid)[0]
        rows = rows[np.argsort(frames[rows], kind="stable")]
        rows = rows[np.concatenate([[True], np.diff(frames[rows]) > 0])]
        if len(rows) < MIN_TRACK:
            continue
        first, last = int(frames[rows[0]]), int(frames[rows[-1]])
        span = np.arange(first, last + 1)
        box = np.stack([np.interp(span, frames[rows], boxes[rows, k]) for k in range(4)], axis=1)
        size = signal.medfilt(
            np.maximum(box[:, 3] - box[:, 1], box[:, 2] - box[:, 0]) / 2, kernel_size=13
        )
        cy = signal.medfilt((box[:, 1] + box[:, 3]) / 2, kernel_size=13)
        cx = signal.medfilt((box[:, 0] + box[:, 2]) / 2, kernel_size=13)
        plans.append(
            {
                "rows": rows,
                "first": first,
                "last": last,
                "size": size,
                "cx": cx,
                "cy": cy,
                "crops": np.zeros((len(span), 112, 112), np.uint8),
            }
        )
    print(f"tracks {len(plans)} of {len(np.unique(track_of))} long enough", flush=True)
    if not plans:
        np.savez_compressed(output, score=scores)
        print("DONE tracks=0", flush=True)
        return

    # ---- one pass over the video: crop every tracked face ----
    by_frame = {}
    for index, plan in enumerate(plans):
        for frame in range(plan["first"], plan["last"] + 1):
            by_frame.setdefault(frame, []).append(index)
    capture = cv2.VideoCapture(video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or (max(by_frame) + 1)
    last_needed, frame_index = max(by_frame), 0
    while frame_index <= last_needed:
        ok, image = capture.read()
        if not ok:
            break
        for index in by_frame.get(frame_index, ()):
            plan = plans[index]
            k = frame_index - plan["first"]
            bs = float(plan["size"][k])
            bsi = int(bs * (1 + 2 * CROP_SCALE))
            padded = np.pad(
                image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110)
            )
            my, mx = plan["cy"][k] + bsi, plan["cx"][k] + bsi
            face = padded[
                int(my - bs) : int(my + bs * (1 + 2 * CROP_SCALE)),
                int(mx - bs * (1 + CROP_SCALE)) : int(mx + bs * (1 + CROP_SCALE)),
            ]
            if face.size == 0:
                continue
            face = cv2.resize(
                cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY), (224, 224)
            )
            plan["crops"][k] = face[56:168, 56:168]
        if frame_index % 250 == 0:
            print(f"PROGRESS {frame_index} {total}", flush=True)
        frame_index += 1
    capture.release()

    # ---- score each track ----
    audio = read_int16(audio_path)
    for number, plan in enumerate(plans, 1):
        start = int(round(plan["first"] / FPS * 16000))
        stop = int(round((plan["last"] + 1) / FPS * 16000))
        segment = audio[start:stop]
        if len(segment) < 1600:
            continue
        mfcc = python_speech_features.mfcc(segment, 16000, numcep=13, winlen=0.025, winstep=0.010)
        crops = plan["crops"]
        length = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100, crops.shape[0] / FPS)
        mfcc = mfcc[: int(round(length * 100))]
        crops = crops[: int(round(length * FPS))]
        if len(crops) == 0:
            continue
        runs = []
        for duration in DURATIONS:
            values = []
            for i in range(int(math.ceil(length / duration))):
                a = (
                    torch.FloatTensor(mfcc[i * duration * 100 : (i + 1) * duration * 100])
                    .unsqueeze(0)
                    .to(device)
                )
                v = (
                    torch.FloatTensor(crops[i * duration * FPS : (i + 1) * duration * FPS])
                    .unsqueeze(0)
                    .to(device)
                )
                if a.shape[1] < 4 or v.shape[1] == 0:
                    continue
                out = model.forward_audio_visual_backend(
                    model.forward_audio_frontend(a), model.forward_visual_frontend(v)
                )
                values.extend(head(out.reshape(-1, 128))[:, 1].float().cpu().numpy().tolist())
            runs.append(
                np.array(
                    values[: len(crops)]
                    + [values[-1] if values else 0.0] * (len(crops) - len(values))
                )
            )
        track_scores = np.mean(runs, axis=0)
        offsets = np.clip(frames[plan["rows"]] - plan["first"], 0, len(track_scores) - 1)
        scores[plan["rows"]] = track_scores[offsets]
        if number % 10 == 0:
            print(f"Scored {number} of {len(plans)} tracks", flush=True)
    np.savez_compressed(output, score=scores)
    scored = int(np.isfinite(scores).sum())
    print(
        f"DONE tracks={len(plans)} scored={scored} speaking={int((scores >= 0).sum())} "
        f"seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
