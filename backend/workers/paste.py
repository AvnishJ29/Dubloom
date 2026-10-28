"""Paste only the model's inpainted region back into the frame (numpy + OpenCV only, shared with tests)."""

import cv2
import numpy as np


def paste_generated_region(
    original, pasted, region, frame_to_crop, crop_w, crop_h, feather_share=0.025
):
    """`region`: uint8 mask on the aligned crop, 255 where the model generated pixels (the lower face).
    `frame_to_crop`: the 2x3 affine that produced the crop from the frame. Everything outside the region,
    feathered, keeps the original frame's pixels, so eyes, forehead and hair never lose resolution.
    """
    h, w = original.shape[:2]
    scaled = cv2.resize(region, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    crop_to_frame = cv2.invertAffineTransform(
        np.asarray(frame_to_crop, dtype=np.float64).reshape(2, 3)
    )
    mask = (
        cv2.warpAffine(scaled, crop_to_frame, (w, h), flags=cv2.INTER_LINEAR, borderValue=0).astype(
            np.float32
        )
        / 255.0
    )
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, crop_w * feather_share))[..., None]
    out = pasted.astype(np.float32) * mask + original.astype(np.float32) * (1 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)
