"""Which face to lip-sync in every frame.

1. Shots: a large change of the frame thumbnail starts a new shot.
2. Tracks: detections are linked frame to frame by box overlap inside a shot.
3. Identities: tracks of the same person (ArcFace embeddings) are merged, so a
   person keeps one identity across cuts.
4. Active speaker: Light-ASD (audio-visual model trained on AVA movies) scores
   every tracked face in every frame: is it speaking the audio heard now? During
   each dubbed line, a face track is the speaker when most of its frames in the
   line score as speaking; it is then re-rendered for the whole line (pauses
   included), except stretches of two seconds or more where it is silent.
   Voice-over, narration, off-screen voices, listeners, masks, statues and
   posters keep their original picture.
   Without ASD scores the fallback is per speaker: the identity whose lip
   opening (106 landmarks) follows the loudness of that speaker's speech.
5. Targets: the chosen face's box and 106 landmarks per frame; every other
   frame stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np

CUT_THRESHOLD = 26.0  # mean absolute thumbnail difference that starts a new shot
TRACK_IOU = 0.3
TRACK_MAX_GAP = 6  # frames a track survives without a detection
IDENTITY_SIMILARITY = 0.42  # ArcFace cosine above which two tracks are the same person
MIN_VISIBILITY = 0.25  # a face must be on screen for this share of a speaker's frames
MIN_FRAMES = 25  # ... and for at least this many frames
MIN_CORRELATION = 0.12  # lip opening vs speech loudness (best lag within +-3 frames)
MIN_LIP_MOTION = 0.02  # std of the lip-opening ratio; a still mouth (statue, photo) stays below
PAD_FRAMES = 2  # lips start and settle slightly around the audio
MIN_FACE_WIDTH = int(
    os.getenv("MIN_FACE_WIDTH", "48")
)  # px; smaller faces are too small to re-render convincingly.
# 48 lets a webcam corner in a 720p screen recording (about 56 px wide) be lip-synced; raise it if tiny faces look worse redrawn
ASD_MIN_FRAMES = 10  # a face must be visible this long in a line to be judged (Light-ASD minimum)
ASD_SPEAKING_SHARE = 0.4  # share of its frames in the line that must score as confidently speaking
ASD_CONFIDENT = (
    0.5  # smoothed Light-ASD score that counts as confident speech (0 is its undecided boundary)
)
ASD_MIN_SPEAKING = 8  # ... and at least this many frames
MIN_FRONTALNESS = 0.25  # nose between the eyes: 1 frontal, 0 full profile; near-profile mouths are not re-rendered
# InsightFace 106-point layout: 52/61 mouth corners, 71/53 outer lip centre top/bottom.
MOUTH_LEFT, MOUTH_RIGHT, LIP_TOP, LIP_BOTTOM = 52, 61, 71, 53


@dataclass
class FaceTargets:
    frame_count: int
    bbox: np.ndarray  # (F, 4) float, NaN where no target
    lmk: np.ndarray  # (F, 106, 2) float, NaN where no target
    track: np.ndarray  # (F,) int, -1 where no target
    report: dict = field(default_factory=dict)

    @property
    def active(self) -> np.ndarray:
        return self.track >= 0

    def save(self, path):
        np.savez_compressed(
            path, frame_count=self.frame_count, bbox=self.bbox, lmk=self.lmk, track=self.track
        )


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def scene_cuts(thumbs: np.ndarray) -> np.ndarray:
    cuts = np.zeros(len(thumbs), dtype=bool)
    if len(thumbs) > 1:
        diff = np.abs(thumbs[1:].astype(np.float32) - thumbs[:-1].astype(np.float32)).mean(
            axis=(1, 2)
        )
        cuts[1:] = diff > CUT_THRESHOLD
    cuts[0] = True
    return cuts


def build_tracks(frames: np.ndarray, boxes: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    track_of = np.full(len(frames), -1, dtype=np.int32)
    active: dict[int, tuple[int, np.ndarray]] = {}
    next_id = 0
    order = np.argsort(frames, kind="stable")
    by_frame: dict[int, list[int]] = {}
    for i in order:
        by_frame.setdefault(int(frames[i]), []).append(int(i))
    for frame in sorted(by_frame):
        if cuts[frame]:
            active.clear()
        candidates = []
        for det in by_frame[frame]:
            for tid, (last, box) in active.items():
                if frame - last <= TRACK_MAX_GAP:
                    overlap = _iou(boxes[det], box)
                    if overlap >= TRACK_IOU:
                        candidates.append((overlap, det, tid))
        used_det, used_track = set(), set()
        for overlap, det, tid in sorted(candidates, reverse=True):
            if det in used_det or tid in used_track:
                continue
            track_of[det] = tid
            used_det.add(det)
            used_track.add(tid)
        for det in by_frame[frame]:
            if track_of[det] < 0:
                track_of[det] = next_id
                next_id += 1
            active[int(track_of[det])] = (frame, boxes[det])
    return track_of


def track_identities(track_of: np.ndarray, embeddings: np.ndarray) -> dict[int, int]:
    tracks = sorted(set(int(t) for t in track_of))
    means = {}
    for tid in tracks:
        vectors = embeddings[track_of == tid].astype(np.float32)
        mean = vectors.mean(axis=0)
        norm = np.linalg.norm(mean)
        means[tid] = mean / norm if norm > 0 else mean
    parent = {tid: tid for tid in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            if float(means[a] @ means[b]) >= IDENTITY_SIMILARITY:
                parent[find(b)] = find(a)
    roots = {}
    return {tid: roots.setdefault(find(tid), len(roots)) for tid in tracks}


def lip_opening(landmarks: np.ndarray) -> np.ndarray:
    """Outer lip height over mouth width for each detection (scale and distance independent)."""
    width = np.linalg.norm(landmarks[:, MOUTH_RIGHT] - landmarks[:, MOUTH_LEFT], axis=1) + 1e-6
    return np.linalg.norm(landmarks[:, LIP_BOTTOM] - landmarks[:, LIP_TOP], axis=1) / width


def loudness_envelope(audio_path, frame_count: int, fps: float = 25.0) -> np.ndarray:
    """Log RMS of a 16 kHz mono WAV per video frame."""
    import wave

    with wave.open(str(audio_path), "rb") as stream:
        rate = stream.getframerate()
        samples = (
            np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )
        if stream.getnchannels() > 1:
            samples = samples.reshape(-1, stream.getnchannels()).mean(axis=1)
    hop = int(rate / fps)
    envelope = np.full(frame_count, np.log(1e-4), dtype=np.float32)
    usable = min(frame_count, len(samples) // hop)
    if usable:
        blocks = samples[: usable * hop].reshape(usable, hop)
        envelope[:usable] = np.log(np.sqrt((blocks**2).mean(axis=1)) + 1e-4)
    return envelope


def best_lag_correlation(a: np.ndarray, b: np.ndarray, max_lag: int = 3) -> float:
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        x = a[max(0, -lag) : len(a) - max(0, lag)]
        y = b[max(0, lag) : len(b) - max(0, -lag)]
        if len(x) > 10 and x.std() > 1e-6 and y.std() > 1e-6:
            best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def face_tracks(faces: dict) -> np.ndarray:
    """Track id of every detection (shots from thumbnails, then box overlap)."""
    if not len(faces["frame"]):
        return np.zeros(0, np.int32)
    return build_tracks(faces["frame"], faces["bbox"], scene_cuts(faces["thumbs"]))


def smooth_scores(scores: np.ndarray, frames: np.ndarray, track_of: np.ndarray) -> np.ndarray:
    """Light-ASD's smoothing: mean over -2..+2 frames within each track."""
    smoothed = np.full(len(scores), np.nan, dtype=np.float32)
    for tid in np.unique(track_of):
        rows = np.where(track_of == tid)[0]
        rows = rows[np.argsort(frames[rows], kind="stable")]
        values = scores[rows]
        for k, row in enumerate(rows):
            window = values[max(0, k - 2) : k + 3]
            window = window[np.isfinite(window)]
            if len(window):
                smoothed[row] = window.mean()
    return smoothed


HAND_COVER = 0.2  # share of the mouth box covered by a hand box that counts as occluded
HAND_MARGIN = 2  # frames of margin around an occluded frame (hands move fast)


def hand_covered(bbox: np.ndarray, track: np.ndarray, hands: dict | None) -> np.ndarray:
    """Per-frame flag: the chosen face's mouth box is covered by a hand. Removes those frames (with a margin)
    from the redraw so the model never paints lips over fingers."""
    covered = np.zeros(len(track), dtype=bool)
    if not hands or not len(hands.get('frame', [])):
        return covered
    by_frame: dict[int, list] = {}
    for frame, box in zip(hands['frame'], hands['bbox']):
        by_frame.setdefault(int(frame), []).append(box)
    for frame in np.where(track >= 0)[0]:
        x1, y1, x2, y2 = bbox[frame]
        w, h = x2 - x1, y2 - y1
        mx1, my1, mx2, my2 = x1 + 0.1 * w, y1 + 0.5 * h, x2 - 0.1 * w, y2 + 0.05 * h
        mouth_area = max(1.0, (mx2 - mx1) * (my2 - my1))
        for hx1, hy1, hx2, hy2 in by_frame.get(int(frame), []):
            inter = max(0.0, min(mx2, hx2) - max(mx1, hx1)) * max(
                0.0, min(my2, hy2) - max(my1, hy1)
            )
            if inter / mouth_area >= HAND_COVER:
                covered[max(0, frame - HAND_MARGIN) : frame + HAND_MARGIN + 1] = True
                break
    return covered


def compute_targets(
    faces: dict,
    slots: list[tuple[float, float, str]],
    fps: float = 25.0,
    envelope: np.ndarray | None = None,
    asd: np.ndarray | None = None,
    track_of: np.ndarray | None = None,
    hands: dict | None = None,
) -> FaceTargets:
    """`envelope`: speech loudness per frame (fallback active-speaker cue).
    `asd`: Light-ASD score per detection (>= 0 speaking, NaN unscored); `track_of`: track per detection.
    """
    frame_count = int(faces["frame_count"])
    frames, boxes = faces["frame"], faces["bbox"]
    cuts = scene_cuts(faces["thumbs"])
    if track_of is None:
        track_of = build_tracks(frames, boxes, cuts) if len(frames) else np.zeros(0, np.int32)
    if asd is not None and len(frames):
        return _targets_from_asd(faces, slots, fps, cuts, track_of, asd, hands)
    identity_of_track = track_identities(track_of, faces["emb"]) if len(frames) else {}
    identity = np.array([identity_of_track.get(int(t), -1) for t in track_of], dtype=np.int32)
    opening = lip_opening(faces["lmk"]) if len(frames) else np.zeros(0)
    widths = boxes[:, 2] - boxes[:, 0] if len(frames) else np.zeros(0)
    if envelope is None:
        envelope = np.zeros(frame_count, np.float32)

    speaker_at = np.full(frame_count, "", dtype=object)
    for start, end, speaker in sorted(slots, key=lambda s: s[0]):
        a = max(0, int(round(start * fps)) - PAD_FRAMES)
        b = min(frame_count, int(round(end * fps)) + PAD_FRAMES)
        speaker_at[a:b] = speaker
    speakers = sorted({s for _, _, s in slots})
    identities = sorted(set(identity_of_track.values()))
    det_speaker = speaker_at[frames] if len(frames) else np.zeros(0, dtype=object)

    mapping, evidence = {}, {}
    for speaker in speakers:
        speaker_frames = int((speaker_at == speaker).sum())
        best = None
        for ident in identities:
            during = (identity == ident) & (widths >= MIN_FACE_WIDTH) & (det_speaker == speaker)
            if not during.any():
                continue
            # one detection per frame (the largest), in time order
            order = np.lexsort((-widths[during], frames[during]))
            picked_frames, first = np.unique(frames[during][order], return_index=True)
            series = opening[during][order][first]
            visible = len(picked_frames) / max(1, speaker_frames)
            motion = float(series.std())
            correlation = (
                best_lag_correlation(series, envelope[picked_frames])
                if envelope.std() > 0
                else -1.0
            )
            accepted = (
                visible >= MIN_VISIBILITY
                and len(picked_frames) >= MIN_FRAMES
                and motion >= MIN_LIP_MOTION
                and (correlation >= MIN_CORRELATION or envelope.std() == 0)
            )
            evidence.setdefault(speaker, []).append(
                {
                    "identity": ident,
                    "visible": round(visible, 2),
                    "lip_motion": round(motion, 3),
                    "correlation": round(correlation, 3),
                    "accepted": bool(accepted),
                }
            )
            score = correlation if envelope.std() > 0 else motion
            if accepted and (best is None or score > best[0]):
                best = (score, ident)
        if best:
            mapping[speaker] = best[1]

    bbox = np.full((frame_count, 4), np.nan, dtype=np.float32)
    lmk = np.full((frame_count, 106, 2), np.nan, dtype=np.float32)
    track = np.full(frame_count, -1, dtype=np.int32)
    for i in np.argsort(-widths) if len(frames) else []:
        frame = int(frames[i])
        speaker = speaker_at[frame]
        if (
            not speaker
            or mapping.get(speaker) != identity[i]
            or widths[i] < MIN_FACE_WIDTH
            or track[frame] >= 0
        ):
            continue
        bbox[frame], lmk[frame], track[frame] = boxes[i], faces["lmk"][i], track_of[i]
    _fill_short_gaps(bbox, lmk, track, speaker_at)
    covered = hand_covered(bbox, track, hands)
    track[covered] = -1
    speech_frames = int((speaker_at != "").sum())
    targeted = int((track >= 0).sum())
    report = {
        "shots": int(cuts.sum()),
        "tracks": int(len(set(track_of.tolist()))) if len(frames) else 0,
        "identities": len(identities),
        "speaker_faces": {s: int(i) for s, i in mapping.items()},
        "unmatched_speakers": [s for s in speakers if s not in mapping],
        "evidence": evidence,
        "speech_frames": speech_frames,
        "target_frames": targeted,
        "off_screen_speech_frames": speech_frames - targeted,
        "hand_covered_frames": int(covered.sum()),
    }
    return FaceTargets(frame_count, bbox, lmk, track, report)


def frontalness(kps: np.ndarray) -> np.ndarray:
    """From the 5 SCRFD keypoints (eyes, nose, mouth corners): min/max of the nose-to-eye horizontal distances."""
    left = np.abs(kps[:, 2, 0] - kps[:, 0, 0])
    right = np.abs(kps[:, 1, 0] - kps[:, 2, 0])
    return np.minimum(left, right) / (np.maximum(left, right) + 1e-6)


def _targets_from_asd(faces, slots, fps, cuts, track_of, asd, hands=None) -> FaceTargets:
    frame_count = int(faces["frame_count"])
    frames, boxes, landmarks = faces["frame"], faces["bbox"], faces["lmk"]
    widths = boxes[:, 2] - boxes[:, 0]
    frontal = frontalness(faces["kps"]) if "kps" in faces else np.ones(len(frames), np.float32)
    score = smooth_scores(asd.astype(np.float32), frames, track_of)
    identity_of_track = track_identities(track_of, faces["emb"])
    bbox = np.full((frame_count, 4), np.nan, dtype=np.float32)
    lmk = np.full((frame_count, 106, 2), np.nan, dtype=np.float32)
    track = np.full(frame_count, -1, dtype=np.int32)
    best = np.full(frame_count, -np.inf, dtype=np.float32)
    speaker_at = np.full(frame_count, "", dtype=object)
    lines, faces_of, unconfirmed = [], {}, []

    def line_rows(inside, tid):
        rows = np.where(inside & (track_of == tid))[0]
        return rows[np.argsort(frames[rows], kind="stable")]

    def redraw(rows, verdict):
        # every frame of this face in the line is redrawn: the dubbed voice decides when the mouth opens,
        # so keeping the original mouth in "silent" stretches would show the original words instead
        keep = frontal[rows] >= MIN_FRONTALNESS
        verdict["profile_frames"] = int((frontal[rows] < MIN_FRONTALNESS).sum())
        verdict["synced_frames"] = int(keep.sum())
        if not keep.any():
            verdict["note"] = "speaking in profile: left as filmed"
        mean_score = float(score[rows].mean())
        for row in rows[keep]:
            frame = int(frames[row])
            if mean_score > best[frame]:  # two speakers at once: the stronger one
                best[frame] = mean_score
                bbox[frame], lmk[frame], track[frame] = boxes[row], landmarks[row], track_of[row]
        return keep

    for start, end, speaker in sorted(slots, key=lambda s: s[0]):
        a = max(0, int(round(start * fps)) - PAD_FRAMES)
        b = min(frame_count, int(round(end * fps)) + PAD_FRAMES)
        if b <= a:
            continue
        speaker_at[a:b] = speaker
        inside = (frames >= a) & (frames < b) & (widths >= MIN_FACE_WIDTH) & np.isfinite(score)
        verdicts = []
        for tid in np.unique(track_of[inside]):
            rows = line_rows(inside, tid)
            confident = score[rows] >= ASD_CONFIDENT
            share = float(confident.mean())
            accepted = (
                len(rows) >= ASD_MIN_FRAMES
                and share >= ASD_SPEAKING_SHARE
                and int(confident.sum()) >= ASD_MIN_SPEAKING
            )
            verdicts.append(
                {
                    "track": int(tid),
                    "identity": int(identity_of_track.get(int(tid), -1)),
                    "frames": int(len(rows)),
                    "speaking_share": round(share, 2),
                    "mean_score": round(float(score[rows].mean()), 2),
                    "accepted": bool(accepted),
                }
            )
            if not accepted:
                continue
            if redraw(rows, verdicts[-1]).any():
                faces_of.setdefault(speaker, {}).setdefault(identity_of_track.get(int(tid), -1), 0)
                faces_of[speaker][identity_of_track.get(int(tid), -1)] += int(
                    verdicts[-1]["synced_frames"]
                )
        line = {
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": speaker,
            "faces": verdicts,
            "lip_synced": any(v.get("synced_frames", 0) > 0 for v in verdicts),
        }
        lines.append(line)
        if not line["lip_synced"]:
            unconfirmed.append((line, speaker, inside))
    mapping = {s: max(faces_of[s], key=faces_of[s].get) for s in faces_of}
    # a line the detector could not confirm (the clip's first second, a short or quiet line) whose on-screen face is
    # the speaker's own identity, settled by the rest of the video: it is theirs, so it is redrawn like every other line
    for line, speaker, inside in unconfirmed:
        wanted = mapping.get(speaker, -1)
        for verdict in line["faces"]:
            if (
                wanted >= 0
                and verdict["identity"] == wanted
                and verdict["frames"] >= ASD_MIN_FRAMES
            ):
                verdict.update(
                    accepted=True,
                    note="accepted by identity: this speaker's face in the rest of the video",
                )
                redraw(line_rows(inside, verdict["track"]), verdict)
        line["lip_synced"] = any(v.get("synced_frames", 0) > 0 for v in line["faces"])
    _fill_short_gaps(bbox, lmk, track, speaker_at)
    covered = hand_covered(bbox, track, hands)
    track[covered] = -1
    speakers = sorted({s for _, _, s in slots})
    speech_frames = int((speaker_at != "").sum())
    targeted = int((track >= 0).sum())
    report = {
        "method": "Light-ASD (AVA) active speaker detection",
        "shots": int(cuts.sum()),
        "tracks": int(len(set(track_of.tolist()))),
        "identities": len(set(identity_of_track.values())),
        "speaker_faces": {s: int(i) for s, i in mapping.items()},
        "unmatched_speakers": [s for s in speakers if s not in mapping],
        "lines_lip_synced": sum(1 for line in lines if line["lip_synced"]),
        "lines": lines,
        "speech_frames": speech_frames,
        "target_frames": targeted,
        "off_screen_speech_frames": speech_frames - targeted,
        "hand_covered_frames": int(covered.sum()),
    }
    return FaceTargets(frame_count, bbox, lmk, track, report)


def _fill_short_gaps(bbox, lmk, track, speaker_at, max_gap: int = 3):
    """Interpolate one to three missed detections inside a speaking span of the same track."""
    frame_count = len(track)
    i = 0
    while i < frame_count:
        if track[i] >= 0:
            j = i + 1
            while j < frame_count and track[j] < 0 and j - i <= max_gap:
                j += 1
            if (
                j < frame_count
                and track[j] == track[i]
                and j - i > 1
                and all(speaker_at[k] for k in range(i + 1, j))
            ):
                for k in range(i + 1, j):
                    w = (k - i) / (j - i)
                    bbox[k] = bbox[i] * (1 - w) + bbox[j] * w
                    lmk[k] = lmk[i] * (1 - w) + lmk[j] * w
                    track[k] = track[i]
            i = j if j > i + 1 else i + 1
        else:
            i += 1


def load_faces(path) -> dict:
    data = np.load(path)
    return {key: data[key] for key in data.files}
