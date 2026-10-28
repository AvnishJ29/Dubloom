"""SFD localization; never borrow face boxes from unrelated frames."""
from collections import deque
import cv2
import numpy as np
from face_detection import FaceAlignment, LandmarksType


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BoxSmoother:
    """Average boxes over a short window so the mouth crop does not jitter.

    The window resets whenever the face disappears or the box jumps to a
    different face/shot, so smoothing never blends across scene cuts.
    """

    def __init__(self, window=5, min_iou=0.4):
        self.history = deque(maxlen=max(1, window))
        self.min_iou = min_iou

    def apply(self, box):
        if box is None:
            self.history.clear()
            return None
        if self.history and box_iou(self.history[-1], box) < self.min_iou:
            self.history.clear()
        self.history.append(tuple(box))
        mean = np.mean(np.asarray(self.history, dtype=np.float64), axis=0)
        return tuple(int(round(value)) for value in mean)


class FaceTracker:
    def __init__(self, device, max_width=480, interval=1):
        self.detector = FaceAlignment(LandmarksType._2D, device=device)
        self.max_width = max_width
        self.interval = max(1, interval)
        self.previous_gray = None
        self.previous_box = None
        self.frame_number = 0

    def detect(self, frames):
        if self.interval == 1:
            return self._detect(frames)
        results = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = None
            if self.frame_number % self.interval and self.previous_box is not None:
                box = self._track(gray)
            if box is None:
                box, reason = self._detect([frame])[0]
            else:
                reason = None
            results.append((box, reason))
            self.previous_gray, self.previous_box = gray, box
            self.frame_number += 1
        return results

    def _track(self, gray):
        """Track only reliable motion; scene cuts or weak tracks trigger SFD."""
        previous = self.previous_gray
        if previous is None or previous.shape != gray.shape:
            return None
        # A global cut check prevents carrying a face box into a different shot.
        a = cv2.resize(previous, (64, 64)).astype(np.float32)
        b = cv2.resize(gray, (64, 64)).astype(np.float32)
        if np.mean(np.abs(a - b)) > 25:
            return None
        x1, y1, x2, y2 = self.previous_box
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(previous, 60, 0.02, 5, mask=mask)
        if points is None or len(points) < 10:
            return None
        moved, valid, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if moved is None:
            return None
        back, backward_valid, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None)
        if back is None:
            return None
        good = (valid.ravel() == 1) & (backward_valid.ravel() == 1) & (np.linalg.norm(points - back, axis=2).ravel() < 1.5)
        if good.sum() < 10 or good.mean() < 0.7:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC)
        if transform is None or inliers.mean() < 0.7:
            return None
        scale = np.hypot(transform[0, 0], transform[0, 1])
        if not 0.9 < scale < 1.1 or abs(transform[0, 1]) > 0.12:
            return None
        corners = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
        tracked = cv2.transform(corners, transform)[0].reshape(-1)
        nx1, ny1, nx2, ny2 = tracked.astype(int)
        height, width = gray.shape
        if nx1 < 0 or ny1 < 0 or nx2 > width or ny2 > height or nx2 - nx1 < 24 or ny2 - ny1 < 24:
            return None
        return (nx1, ny1, nx2, ny2)

    def _choose(self, faces, scale):
        return choose_face(faces, self.previous_box, scale)

    def _detect(self, frames):
        height, width = frames[0].shape[:2]
        scale = min(1.0, self.max_width / width)
        reduced = [cv2.resize(frame, (round(width * scale), round(height * scale))) for frame in frames]
        images = np.asarray(reduced)[..., ::-1].copy()
        detections = self.detector.face_detector.detect_from_batch(images)
        results = []
        for detected in detections:
            faces = [face for face in detected if face[4] >= 0.8]
            chosen = self._choose(faces, scale)
            if chosen is None:
                results.append((None, 'multiple_faces' if faces else 'no_face'))
                continue
            x1, y1, x2, y2 = np.asarray(chosen[:4]) / scale
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2) + 10)
            if x2 - x1 < 24 or y2 - y1 < 24:
                results.append((None, 'small_face'))
            else:
                results.append(((x1, y1, x2, y2), None))
        return results


def _area(face):
    return max(0.0, face[2] - face[0]) * max(0.0, face[3] - face[1])


def choose_face(faces, previous_box=None, scale=1.0, dominance=1.6, min_iou=0.3):
    """Pick the speaker's face among several detections, or None when ambiguous.

    A face that continues the tracked box wins; otherwise a clearly dominant
    (largest) face wins, which ignores posters, busts and background people.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]
    if previous_box is not None:
        scaled = [tuple(np.asarray(face[:4]) / scale) for face in faces]
        overlaps = [box_iou(previous_box, box) for box in scaled]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= min_iou:
            return faces[best]
    ordered = sorted(faces, key=_area, reverse=True)
    if _area(ordered[0]) >= dominance * _area(ordered[1]):
        return ordered[0]
    return None


def mel_window(mel, frame_index, fps, size=16):
    """A correctly timed window for every frame, including the last frames."""
    start = min(int(frame_index * 80.0 / fps), mel.shape[1] - 1)
    window = mel[:, start:start + size]
    return np.pad(window, ((0, 0), (0, size - window.shape[1])), mode='edge')


def blend_mouth(original, predicted):
    """Keep original eyes/forehead; feather the generated lower face."""
    height, width = original.shape[:2]
    predicted = cv2.resize(predicted, (width, height), interpolation=cv2.INTER_CUBIC)
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    mask = np.clip((y - 0.40) / 0.15, 0, 1)
    mask = mask * np.clip(np.minimum(x, 1 - x) / 0.10, 0, 1)
    mask *= np.clip((1 - y) / 0.05, 0, 1)
    return np.clip(original * (1 - mask[..., None]) + predicted * mask[..., None], 0, 255).astype(np.uint8)
