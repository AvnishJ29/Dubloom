"""Lip sync: the stage entry points the job pipeline calls.

1. The source is normalised to 25 fps (`video_tools.prepare_render_source`).
2. Face analysis (InsightFace, `workers/face_worker.py`) records every face in every frame (`face_analysis.py`).
3. `speaker_faces.compute_targets` decides, per dubbed line, which face is speaking.
4. The renderer re-draws the lower face of that person only:
   - LatentSync 1.6 (`latentsync.py`, `workers/latentsync_worker.py`), on a CUDA GPU;
   - Wav2Lip (`wav2lip.py`), in chunks, when no GPU is available.
   Frames outside dubbed lines, listeners, statues and off-screen voices keep the original picture.
5. The dubbed mix is muxed and the export validated.

Steps 1-2 and active-speaker detection also run early, in the background (`precompute_faces`), so
`synchronize` usually finds them cached.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from backend.config import settings
from backend.media.media_validation import validate_video
from backend.media.video_input import run
from backend.rendering import finish
from backend.rendering.face_analysis import (
    analyse_faces,
    detect_active_speakers,
    detect_hands,
    speech_audio,
)
from backend.rendering.latentsync import growing_preview, latentsync_available, render_latentsync
from backend.rendering.video_tools import FPS, frame_count, frame_rate, prepare_render_source
from backend.rendering.wav2lip import render_wav2lip
from backend.speakers.speaker_faces import (
    compute_targets,
    face_tracks,
    load_faces,
    loudness_envelope,
    track_identities,
)

logger = logging.getLogger(__name__)


def choose_engine(requested: str | None) -> str:
    """latentsync on a GPU when available; wav2lip otherwise."""
    requested = (requested or settings.lip_sync_engine or 'auto').lower()
    if requested.startswith('wav2lip'):
        return 'wav2lip'
    if latentsync_available() and (settings.device == 'cuda' or requested == 'latentsync'):
        return 'latentsync'
    return 'wav2lip'


def precompute_faces(video: Path, directory: Path, control=None, separation=None):
    """Face analysis and active-speaker detection depend only on the source picture and sound, so they can run
    while the script is still being transcribed, translated and voiced; `synchronize` then finds their cached
    results by digest and skips both. `separation` is the vocal-separation future: the clean vocals stem gives
    a better speaking score, so detection waits for it."""
    normalized = prepare_render_source(video, directory)
    if control:
        control.check()
    faces_path = analyse_faces(normalized, directory, control)
    if faces_path is None:
        return
    if separation is not None:
        try:
            separation.result()
        except Exception:  # noqa: BLE001
            pass  # detection falls back to the full mix, exactly as synchronize does
    faces = load_faces(faces_path)
    detect_active_speakers(normalized, directory, faces_path, face_tracks(faces), control)


def speaker_face_evidence(
    video: Path, directory: Path, slots: list, control=None
) -> dict[str, dict]:
    """{speaker: {identity, speaking_frames, face_gender, face_gender_share}} from the cached face analysis and
    active-speaker scores (computed in the background after diarization). Empty when no face data exists.
    """
    normalized = prepare_render_source(video, directory)
    faces_path = analyse_faces(normalized, directory, control)
    if faces_path is None or not slots:
        return {}
    faces = load_faces(faces_path)
    track_of = face_tracks(faces)
    asd = detect_active_speakers(normalized, directory, faces_path, track_of, control)
    targets = compute_targets(faces, slots, FPS, None, asd=asd, track_of=track_of)
    identity_of_track = track_identities(track_of, faces['emb']) if len(faces['frame']) else {}
    identity = np.array([identity_of_track.get(int(t), -1) for t in track_of], dtype=np.int32)
    genders = faces.get('gender')
    evidence = {}
    for speaker, ident in (targets.report.get('speaker_faces') or {}).items():
        rows = np.where(identity == ident)[0]
        entry = {
            'identity': int(ident),
            'speaking_frames': int(
                sum(
                    v.get('synced_frames', 0)
                    for line in targets.report.get('lines', [])
                    if line['speaker'] == speaker
                    for v in line['faces']
                )
            ),
        }
        if genders is not None and len(rows):
            votes = genders[rows]
            votes = votes[votes >= 0]
            if len(votes) >= 10:
                male = float((votes == 1).mean())
                entry['face_gender'] = 'male' if male >= 0.5 else 'female'
                entry['face_gender_share'] = round(max(male, 1 - male), 3)
        evidence[speaker] = entry
    return evidence


def synchronize(
    video: Path,
    audio: Path,
    directory: Path,
    progress=None,
    control=None,
    sync_audio: Path | None = None,
    checkpoint: Path | None = None,
    enhance: bool = False,
    slots: list | None = None,
    engine: str | None = None,
    on_preview=None,
) -> Path:
    """Render the lip-synced video and mux the dubbed audio. `control` may cancel.

    `sync_audio` (voice only, no background bed) drives the mouth model; `audio`
    is the full mix delivered in the MP4. `slots` are (start, end, speaker) of the dubbed lines.
    `progress(fraction, label)` reports 0..1 for the whole stage.
    """
    report_progress = progress or (lambda fraction, label: None)
    check = control.check if control else (lambda: None)
    sync_audio = sync_audio if sync_audio and sync_audio.is_file() else audio
    timing, clock = {}, time.monotonic()

    def lap(name):
        nonlocal clock
        now = time.monotonic()
        timing[name] = round(timing.get(name, 0) + now - clock, 1)
        clock = now

    normalized = prepare_render_source(video, directory)
    lap('render_source')
    check()
    total_frames = frame_count(normalized)
    targets_path, face_report = _speaking_targets(
        normalized, directory, slots, control, report_progress, lap
    )
    check()

    engine = choose_engine(engine)
    if engine == 'latentsync' and targets_path is None:
        engine = 'wav2lip'
    lips = directory / 'lips.mp4'
    plan = None
    if engine == 'latentsync':
        with growing_preview(video, normalized, directory, audio, total_frames, on_preview):
            summary = render_latentsync(
                normalized,
                sync_audio,
                targets_path,
                lips,
                directory,
                control,
                lambda f: report_progress(0.18 + 0.6 * f, 'Lip sync'),
                spans_only=True,
            )
        lap('latentsync')
        plan = _restoration_plan(summary, directory, targets_path, control, report_progress)
        lap('restore')
    else:
        summary = render_wav2lip(
            normalized,
            sync_audio,
            targets_path,
            lips,
            directory,
            total_frames,
            checkpoint,
            enhance,
            control,
            lambda f: report_progress(0.18 + 0.8 * f, 'Lip sync'),
        )
    check()
    _add_face_report(summary, engine, total_frames, face_report)

    final = directory / 'dubbed_video.mp4'
    source_rate = frame_rate(video)
    resample = bool(source_rate) and abs(source_rate - FPS) > 0.01
    if resample:
        summary['frame_rate'] = round(source_rate, 3)
    if plan is not None:
        # one pass: the restored spans go into the picture and the dub is encoded once (backend/rendering/finish.py)
        report_progress(0.9, 'Writing the dub')
        finish.assemble(
            normalized,
            plan,
            audio,
            final,
            total_frames,
            FPS,
            source_rate,
            control,
            lambda f: report_progress(0.9 + 0.1 * f, 'Writing the dub'),
        )
    else:
        _mux(video, lips, audio, final, total_frames, resample)
    summary['export'] = validate_video(final, total_frames / FPS)
    lap('export')
    summary['timing'] = {
        **timing,
        'total': round(sum(timing.values()), 1),
    }  # seconds per sub-step of this stage
    (directory / 'lip_sync_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report_progress(1.0, 'Lip sync')
    return final


def _speaking_targets(normalized, directory, slots, control, report_progress, lap):
    """Who speaks in which frame: faces, active-speaker scores, loudness and hands combined into the per-frame
    targets the renderer redraws (`targets.npz`). Returns (targets path, report); (None, {}) without faces.
    """
    report_progress(0.0, 'Finding faces')
    faces_path = analyse_faces(
        normalized, directory, control, lambda f: report_progress(0.13 * f, 'Finding faces')
    )
    lap('faces')
    if faces_path is None:
        return None, {}
    faces = load_faces(faces_path)
    track_of = face_tracks(faces)
    report_progress(0.13, 'Finding who speaks')
    asd = detect_active_speakers(
        normalized,
        directory,
        faces_path,
        track_of,
        control,
        lambda f: report_progress(0.13 + 0.05 * f, 'Finding who speaks'),
    )
    lap('active_speaker')
    loudness_source = speech_audio(directory)
    envelope = (
        loudness_envelope(loudness_source, int(faces['frame_count']))
        if loudness_source.is_file()
        else None
    )
    hands_path = detect_hands(normalized, directory, control, frames=np.unique(faces['frame']))
    hands = dict(np.load(hands_path)) if hands_path else None
    lap('hands')
    targets = compute_targets(
        faces, slots or [], FPS, envelope, asd=asd, track_of=track_of, hands=hands
    )
    targets_path = directory / 'targets.npz'
    targets.save(targets_path)
    lap('targets')
    return targets_path, targets.report


def _restoration_plan(summary, directory, targets_path, control, report_progress) -> list[dict]:
    """How each rendered LatentSync span goes into the picture: restored when `latentsync_enhance` is on
    (`finish.restore_spans`, recorded in the summary), raw otherwise or when restoration fails."""
    manifest = json.loads((directory / 'latentsync_cache' / 'spans.json').read_text())
    if not settings.latentsync_enhance:
        return finish.plan_restoration(manifest, targets_path, False, False)
    report_progress(0.8, 'Restoring faces')
    try:
        plan, detail = finish.restore_spans(
            manifest,
            targets_path,
            control,
            lambda f: report_progress(0.8 + 0.1 * f, 'Restoring faces'),
        )
    except Exception as exc:  # noqa: BLE001  (restoration is a finish, never a reason to fail)
        logger.warning('Face restoration failed for %s: %s', directory.name, exc)
        plan, detail = finish.plan_restoration(manifest, targets_path, False, False), {
            'applied': False,
            'reason': str(exc)[:120],
        }
    summary['mouth_enhanced'] = detail
    return plan


def _add_face_report(summary: dict, engine: str, total_frames: int, face_report: dict):
    """Add to the render summary what the faces decided: speaking faces, off-screen voices, coverage, warnings."""
    summary.update(
        {
            'engine': engine,
            'frames': total_frames,
            'device': settings.device,
            'speech_frames': face_report.get('speech_frames'),
            'speaker_faces': face_report.get('speaker_faces', {}),
            'off_screen_speakers': face_report.get('unmatched_speakers', []),
            'off_screen_speech_frames': face_report.get('off_screen_speech_frames'),
            'hand_covered_frames': face_report.get('hand_covered_frames', 0),
            'face_evidence': face_report.get('evidence', {}),
            'active_speaker': face_report.get('method', 'lip opening vs loudness'),
            'lines_lip_synced': face_report.get('lines_lip_synced'),
            'lines': face_report.get('lines', []),
        }
    )
    speech = summary.get('speech_frames') or 0
    summary['coverage_percent'] = (
        round(100 * summary['synced_frames'] / speech, 1) if speech else 0.0
    )
    warnings = []
    if summary['off_screen_speakers']:
        warnings.append(
            'Some voices have no visible speaking face (narration or off-screen); their lines are dubbed '
            'without changing any face.'
        )
    summary['warnings'] = warnings


def _mux(video: Path, lips: Path, audio: Path, final: Path, total_frames: int, resample: bool):
    """The Wav2Lip picture with the dubbed mix as the delivered MP4, back at the source frame rate if `resample`."""
    if resample:
        # Back to the source frame rate: untouched frames map back to exactly the original pictures (no judder).
        picture = [
            '-vf',
            f'fps={frame_rate(video, text=True)}',
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-crf',
            '12',
            '-pix_fmt',
            'yuv420p',
        ]
    else:
        picture = ['-c:v', 'copy']
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        lips,
        '-i',
        audio,
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        *picture,
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-ar',
        '48000',
        '-af',
        'apad',
        '-t',
        total_frames / FPS,
        '-movflags',
        '+faststart',
        final,
    )


def voice_only(video: Path, audio: Path, directory: Path) -> Path:
    """The dubbed mix over the original picture, no lip sync: a stream copy of the video, so it takes seconds."""
    final = directory / 'dubbed_video.mp4'
    duration = float(
        json.loads(
            run('ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', video)
        )['format']['duration']
    )
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        video,
        '-i',
        audio,
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        '-c:v',
        'copy',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-ar',
        '48000',
        '-af',
        'apad',
        '-t',
        duration,
        '-movflags',
        '+faststart',
        final,
    )
    summary = {
        'engine': 'none',
        'model': 'Voice only',
        'frames': frame_count(video),
        'synced_frames': 0,
        'reused_frames': 0,
        'render_seconds': 0.0,
        'device': 'cpu',
        'speech_frames': None,
        'speaker_faces': {},
        'off_screen_speakers': [],
        'lines_lip_synced': None,
        'lines': [],
        'coverage_percent': 0.0,
        'hand_covered_frames': 0,
        'warnings': ['Lip sync was not requested: the mouths are the original picture.'],
        'export': validate_video(final, duration),
        'timing': {'export': 0.0, 'total': 0.0},
    }
    (directory / 'lip_sync_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return final
