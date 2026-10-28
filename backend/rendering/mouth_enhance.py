"""Sharpen the redrawn lower face after LatentSync. The model paints a 512 px face crop; on close-ups the face is
larger than that, so the pasted region comes back softer than the picture around it. Real-ESRGAN (the same
weights the Wav2Lip path uses) restores edge detail in the mouth region of every redrawn frame."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from backend.config import ROOT, settings

MIN_FACE_WIDTH = (
    140  # per frame: below this the 512 px crop already exceeds the face's own resolution
)
FEATHER = 0.04  # of the crop width


def _load_model(device: str):
    """Real-ESRGAN (the compact network bundled with Wav2Lip) on `device`, and the torch module."""
    sys.path.insert(0, str(ROOT / "Wav2Lip"))
    import torch
    from enhance import SRVGGNetCompact

    model = SRVGGNetCompact()
    state = torch.load(str(settings.esrgan_weights), map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("params_ema") or state.get("params") or state, strict=True)
    return model.eval().to(device), torch


def mouth_box(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) of the region to sharpen: the lower face of `bbox` with a margin, inside the frame."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - 0.1 * w)),
        max(0, int(y1 + 0.45 * h)),
        min(width, int(x2 + 0.1 * w)),
        min(height, int(y2 + 0.12 * h)),
    )


def sharpen(model, torch, crop: np.ndarray) -> np.ndarray:
    """Real-ESRGAN x4 on a half-size copy, scaled back: detail restored, nothing invented at scale."""
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(small[..., ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model(tensor).clamp_(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255
    sr = np.ascontiguousarray(out.round().astype(np.uint8)[..., ::-1])
    return cv2.resize(sr, (w, h), interpolation=cv2.INTER_AREA)


def smooth_detail(frame, restored, memory, share, index, track_id, centre):
    """restored with its added detail (restored - raw) blended with the previous frame's, shifted by the face's motion:
    motion stays with the raw picture, only the sharpening is smoothed over time."""
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


def enhance_lips(
    lips: Path,
    targets: Path,
    output: Path,
    device: str = "cpu",
    progress=None,
    min_face: int = MIN_FACE_WIDTH,
    temporal: float = 0.0,
    first: int = 0,
) -> dict:
    """Rewrite `lips` (whose frame 0 is frame `first` of the targets) with the mouth region of every redrawn frame
    sharpened. Returns a small report."""
    data = np.load(targets)
    bbox, track = data["bbox"], data["track"]
    count = int(cv2.VideoCapture(str(lips)).get(cv2.CAP_PROP_FRAME_COUNT)) or len(track)
    redrawn = first + np.where(track[first : first + count] >= 0)[0]
    widths = bbox[redrawn, 2] - bbox[redrawn, 0] if len(redrawn) else np.zeros(0)
    large = int((widths >= min_face).sum())
    if not large:
        return {
            "applied": False,
            "reason": "faces small enough for the model's own resolution",
            "frames": 0,
        }
    model, torch = _load_model(device)
    capture = cv2.VideoCapture(str(lips))
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
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    index, done = first, 0
    memory = {
        "delta": None,
        "index": -9,
        "track": None,
        "centre": (0, 0),
    }  # last frame's added detail, for temporal smoothing
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index < len(track) and track[index] >= 0 and bbox[index, 2] - bbox[index, 0] >= min_face:
            x1, y1, x2, y2 = mouth_box(bbox[index], width, height)
            if x2 - x1 > 16 and y2 - y1 > 16:
                crop = frame[y1:y2, x1:x2]
                sharp = sharpen(model, torch, crop)
                mask = np.zeros(crop.shape[:2], np.float32)
                cv2.rectangle(
                    mask,
                    (int(crop.shape[1] * 0.08), int(crop.shape[0] * 0.08)),
                    (int(crop.shape[1] * 0.92), int(crop.shape[0] * 0.92)),
                    1,
                    -1,
                )
                mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, crop.shape[1] * FEATHER))[..., None]
                if temporal > 0:
                    full = frame.copy()
                    full[y1:y2, x1:x2] = sharp
                    bx1, by1, bx2, by2 = bbox[index]
                    full = smooth_detail(
                        frame,
                        full,
                        memory,
                        temporal,
                        index,
                        int(track[index]),
                        ((bx1 + bx2) / 2, (by1 + by2) / 2),
                    )
                    sharp = np.clip(full[y1:y2, x1:x2], 0, 255)
                frame[y1:y2, x1:x2] = (sharp * mask + crop * (1 - mask)).round().astype(np.uint8)
                done += 1
                if progress and done % 50 == 0:
                    progress(done / large)
        writer.stdin.write(frame.tobytes())
        index += 1
    capture.release()
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg could not write the enhanced lips")
    return {
        "applied": True,
        "model": "Real-ESRGAN x2",
        "frames": done,
        "of_redrawn": int(len(redrawn)),
        "median_face_px": int(np.median(widths)),
    }
