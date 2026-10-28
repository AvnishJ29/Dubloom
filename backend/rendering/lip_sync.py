"""Lip sync: face analysis, active-speaker targets, rendering, and the final MP4.

1. The source is normalised to 25 fps.
2. Face analysis (InsightFace, `workers/face_worker.py`) records every face in every frame.
3. `speaker_faces.compute_targets` decides, per dubbed line, which face is speaking.
4. The renderer re-draws the lower face of that person only:
   - LatentSync 1.6 (`workers/latentsync_worker.py`), on a CUDA GPU;
   - Wav2Lip, in chunks, when no GPU is available.
   Frames outside dubbed lines, listeners, statues and off-screen voices keep the original picture.
5. The dubbed mix is muxed and the export validated.
"""

import hashlib
import json
import threading
import time
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from backend.rendering import finish
from backend.config import ROOT, settings
from backend.media.media_validation import validate_video
from backend.speakers.speaker_faces import (
    compute_targets,
    face_tracks,
    load_faces,
    loudness_envelope,
)
from backend.media.video_input import run

FPS = 25


def _digest(path: Path) -> str:
    """SHA-256 of a file, remembered next to it: hashing a multi-GB 4K render source took 80 s per stage pass."""
    stat = path.stat()
    sidecar = path.with_name(path.name + '.digest')
    key = f"{stat.st_size}:{stat.st_mtime_ns}:"
    if sidecar.is_file():
        cached = sidecar.read_text()
        if cached.startswith(key):
            return cached[len(key) :]
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        sidecar.write_text(key + digest)
    except OSError:
        pass
    return digest


def _frame_size(path: Path) -> tuple[int, int]:
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height',
            '-of',
            'json',
            path,
        )
    )
    stream = (probe.get('streams') or [{}])[0]
    return int(stream.get('width') or 0), int(stream.get('height') or 0)


def _frame_count(path: Path) -> int:
    """Frames of a video: from the container's own count (instant); decoded and counted only when it is missing.
    Counting a 4K render source by decoding it cost 84 s on every lip-sync pass."""
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=nb_frames',
            '-of',
            'json',
            path,
        )
    )
    declared = (probe.get('streams') or [{}])[0].get('nb_frames')
    if declared and str(declared).isdigit() and int(declared) > 0:
        return int(declared)
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-count_frames',
            '-show_entries',
            'stream=nb_read_frames',
            '-of',
            'json',
            path,
        )
    )
    return int(probe['streams'][0]['nb_read_frames'])


def _frame_rate(path: Path, text: bool = False):
    """The source's frame rate as a number, or as ffmpeg's exact fraction (e.g. 24000/1001)."""
    probe = json.loads(
        run(
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=r_frame_rate,avg_frame_rate',
            '-of',
            'json',
            path,
        )
    )
    stream = probe['streams'][0] if probe.get('streams') else {}
    fraction = stream.get('r_frame_rate') or '0/1'
    numerator, denominator = (float(x) for x in fraction.split('/'))
    if not denominator or numerator / denominator > 100:  # variable-rate sources: use the average
        fraction = stream.get('avg_frame_rate') or '0/1'
        numerator, denominator = (float(x) for x in fraction.split('/'))
    if text:
        return fraction
    return numerator / denominator if denominator else 0.0


def face_analysis_available() -> bool:
    return (
        bool(settings.latentsync_python)
        and (settings.insightface_root / 'models' / 'buffalo_l').is_dir()
    )


def asd_available() -> bool:
    return face_analysis_available() and (settings.asd_dir / settings.asd_weights).is_file()


def latentsync_available() -> bool:
    return (
        bool(settings.latentsync_python)
        and (settings.latentsync_checkpoints / 'latentsync_unet.pt').is_file()
        and (settings.latentsync_dir / 'configs' / 'unet' / 'stage2_512.yaml').is_file()
        and face_analysis_available()
    )


def choose_engine(requested: str | None) -> str:
    """latentsync on a GPU when available; wav2lip otherwise."""
    requested = (requested or settings.lip_sync_engine or 'auto').lower()
    if requested.startswith('wav2lip'):
        return 'wav2lip'
    if latentsync_available() and (settings.device == 'cuda' or requested == 'latentsync'):
        return 'latentsync'
    return 'wav2lip'


def _stream(command, cwd, control, on_line, env=None, label='worker'):
    from backend.jobs.gpus import pool

    with subprocess.Popen(
        command,
        cwd=cwd,
        env=pool.env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        if control:
            control.register(process)
        tail = []
        try:
            for line in process.stdout:
                tail = (tail + [line])[-25:]
                on_line(line)
            code = process.wait()
        finally:
            if control:
                control.unregister(process)
        if control:
            control.check()
        if code:
            relevant = [
                line for line in tail if 'Warning' not in line and 'warnings.warn' not in line
            ]
            raise RuntimeError(f'{label} failed: ' + ''.join(relevant)[-1600:])


def analyse_faces(normalized: Path, directory: Path, control=None, progress=None) -> Path | None:
    """Per-frame faces for the normalised video (cached by video content)."""
    if not face_analysis_available():
        return None
    output = directory / 'faces.npz'
    stamp = directory / 'faces.sha256'
    video_digest = _digest(normalized)
    digest = f"{video_digest}:{settings.face_det_size}:{settings.face_stride}:g2"  # g2: faces carry gender and age
    if output.is_file() and stamp.is_file():
        # Reuse an analysis of the same video at the same detector size and the same or a finer stride
        # (stamps written before strides existed were per-frame at 640).
        cached = (stamp.read_text().split(':') + ['640', '1', ''])[:4]
        if (
            cached[0] == video_digest
            and cached[1] == str(settings.face_det_size)
            and cached[2].isdigit()
            and int(cached[2]) <= settings.face_stride
            and cached[3] == 'g2'
        ):
            return (
                output  # same video, detector and stride, and the analysis carries gender and age
            )

    total = _frame_count(normalized)
    # On CPU the models are bound by per-call overhead, not cores: several workers on parts of the video are faster.
    workers = max(1, min(settings.face_workers, total // 250 or 1))
    bounds = [round(total * k / workers) for k in range(workers + 1)]
    done = [0] * workers
    parts = [directory / f'faces_part{k}.npz' for k in range(workers)]

    def run_part(k):
        def on_line(line):
            if progress and line.startswith('PROGRESS '):
                done[k] = int(line.split()[1])
                progress(sum(done) / max(1, total))

        command = [
            settings.latentsync_python,
            '-u',
            str(ROOT / 'backend' / 'workers' / 'face_worker.py'),
            str(normalized),
            str(parts[k]),
            '--root',
            str(settings.insightface_root),
            '--device',
            settings.device,
            '--det-size',
            str(settings.face_det_size),
            '--stride',
            str(settings.face_stride),
            '--threads',
            str(max(2, (os.cpu_count() or 4) // workers)),
            '--first',
            str(bounds[k]),
            '--last',
            str(bounds[k + 1] if k + 1 < workers else -1),
        ]
        _stream(command, ROOT, control, on_line, label='Face analysis')

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(run_part, k) for k in range(workers)]:
            future.result()
    _merge_faces(parts, output)
    for part in parts:
        part.unlink(missing_ok=True)
    stamp.write_text(digest)
    return output


def _merge_faces(parts: list[Path], output: Path):
    loaded = [dict(np.load(part)) for part in parts]
    if len(loaded) == 1:
        merged = loaded[0]
    else:
        merged = {key: loaded[0][key] for key in ('fps', 'width', 'height')}
        merged['frame_count'] = loaded[-1]['frame_count']
        merged['first'] = 0
        for key in (
            'thumbs',
            'frame',
            'bbox',
            'score',
            'kps',
            'lmk',
            'emb',
            'mouth',
            'gender',
            'age',
        ):
            if all(key in part for part in loaded):
                merged[key] = np.concatenate([part[key] for part in loaded])
    np.savez_compressed(output, **merged)


def detect_hands(normalized: Path, directory: Path, control=None, frames=None) -> Path | None:
    """Hand boxes per frame (MediaPipe), cached by the render source; None when the detector is unavailable.
    `frames`: the frames that have a face; the others are never lip-synced, so they are not searched.
    """
    output, stamp = directory / 'hands.npz', directory / 'hands.sha256'
    digest = _digest(normalized) + ':h1'
    if output.is_file() and stamp.is_file() and stamp.read_text() == digest:
        return output
    if not Path(settings.hands_python).is_file():
        logger.warning(
            'Hand detection skipped: %s is missing (run scripts/setup_models.sh)',
            settings.hands_python,
        )
        return None
    command = [
        settings.hands_python,
        '-u',
        str(ROOT / 'backend' / 'workers' / 'hands_worker.py'),
        str(normalized),
        str(output),
    ]
    if frames is not None:
        wanted = directory / 'hands_frames.npy'
        np.save(wanted, np.asarray(sorted(set(int(f) for f in frames)), dtype=np.int64))
        command += ['--frames', str(wanted)]
    try:
        _stream(command, ROOT, control, lambda line: None, label='Hand detection')
    except Exception as exc:  # noqa: BLE001  (no hands data: nothing is excluded, as before)
        logger.warning('Hand detection unavailable: %s', exc)
        return None
    stamp.write_text(digest)
    return output


def detect_active_speakers(
    normalized: Path,
    directory: Path,
    faces_path: Path,
    track_of: np.ndarray,
    control=None,
    progress=None,
) -> np.ndarray | None:
    """Light-ASD score per face detection (>= 0: speaking), cached; None if unavailable."""
    if not asd_available():
        return None
    speech = (
        directory / 'vocals.wav'
        if (directory / 'vocals.wav').is_file()
        else directory / 'source_audio.wav'
    )
    if not speech.is_file():
        return None
    audio16k = directory / 'asd_audio16k.wav'
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        speech,
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'pcm_s16le',
        '-map_metadata',
        '-1',
        '-fflags',
        '+bitexact',
        audio16k,
    )
    tracks_path = directory / 'face_tracks.npz'
    np.savez_compressed(tracks_path, track=track_of)
    output, stamp = directory / 'asd.npz', directory / 'asd.sha256'
    digest = hashlib.sha256(
        f"{(directory / 'faces.sha256').read_text()}:{_digest(audio16k)}:{settings.asd_weights}:".encode()
        + np.ascontiguousarray(track_of).tobytes()
    ).hexdigest()
    if not (output.is_file() and stamp.is_file() and stamp.read_text() == digest):

        def on_line(line):
            if progress and line.startswith('PROGRESS '):
                _, done, total = line.split()
                progress(0.8 * int(done) / max(1, int(total)))
            elif progress and line.startswith('Scored '):
                parts = line.split()
                progress(0.8 + 0.2 * int(parts[1]) / max(1, int(parts[3])))

        command = [
            settings.latentsync_python,
            '-u',
            str(ROOT / 'backend' / 'workers' / 'asd_worker.py'),
            str(normalized),
            str(audio16k),
            str(faces_path),
            str(tracks_path),
            str(output),
            '--repo',
            str(settings.asd_dir),
            '--weights',
            settings.asd_weights,
            '--device',
            settings.device,
        ]
        _stream(command, ROOT, control, on_line, label='Active speaker detection')
        stamp.write_text(digest)
    with np.load(output) as data:
        scores = data['score']
    return scores if len(scores) == len(track_of) else None


def prepare_render_source(video: Path, directory: Path) -> Path:
    """The 25 fps picture every lip-sync step works on; rebuilt only when the source changes."""
    normalized = directory / 'render_source.mp4'
    source_digest = directory / 'render_source.sha256'
    video_digest = _digest(video) + ':q8'
    if not (
        normalized.is_file()
        and source_digest.is_file()
        and source_digest.read_text() == video_digest
    ):
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-i',
            video,
            '-an',
            '-vf',
            f'fps={FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-crf',
            '8',
            '-pix_fmt',
            'yuv420p',
            normalized,
        )  # near-lossless: these pixels end up in the dub
        source_digest.write_text(video_digest)
    return normalized


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
        except (
            Exception
        ):  # noqa: BLE001  (detection falls back to the full mix, exactly as synchronize does)
            pass
    faces = load_faces(faces_path)
    detect_active_speakers(normalized, directory, faces_path, face_tracks(faces), control)


def speaker_face_evidence(
    video: Path, directory: Path, slots: list, control=None
) -> dict[str, dict]:
    """{speaker: {identity, speaking_frames, face_gender, face_gender_share}} from the cached face analysis and
    active-speaker scores (computed in the background after diarization). Empty when no face data exists.
    """
    from backend.speakers.speaker_faces import track_identities

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


PREVIEW_EVERY = 45  # seconds between rebuilds of the growing preview while spans render
PREVIEW_HEIGHT = 720
PREVIEW_MAX_SOURCE = 1080  # the growing preview is built only for sources up to 1080p (decoding 4K every minute is too slow)


def _grow_preview(
    normalized: Path,
    directory: Path,
    audio: Path,
    total_frames: int,
    source_rate,
    stop: threading.Event,
    on_preview,
):
    """Every PREVIEW_EVERY seconds: the spans rendered so far (raw, unrestored) pasted into a 720p review cut."""
    manifest_path = directory / 'latentsync_cache' / 'spans.json'
    built = -1
    while not stop.wait(PREVIEW_EVERY):
        try:
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            ready = [s for s in manifest['spans'] if Path(s['file']).is_file()]
            if not ready or len(ready) == built or manifest.get('complete'):
                continue
            plan = [
                {
                    'first': s['first'],
                    'end': s['end'],
                    'raw': Path(s['file']),
                    'restored': Path(s['file']),
                }
                for s in ready
            ]
            partial = directory / 'review_video.part.mp4'
            finish.assemble(
                normalized,
                plan,
                audio,
                partial,
                total_frames,
                FPS,
                source_rate,
                None,
                None,
                preview_height=PREVIEW_HEIGHT,
            )
            partial.replace(directory / 'review_video.mp4')
            built = len(ready)
            if on_preview:
                on_preview(len(ready), len(manifest['spans']))
        except Exception as exc:  # noqa: BLE001  (the preview is a courtesy; the render goes on)
            logger.warning('Growing preview for %s skipped: %s', directory.name, exc)


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
    total_frames = _frame_count(normalized)

    # ---- who speaks in which frame ----
    report_progress(0.0, 'Finding faces')
    faces_path = analyse_faces(
        normalized, directory, control, lambda f: report_progress(0.13 * f, 'Finding faces')
    )
    lap('faces')
    targets_path = None
    face_report = {}
    if faces_path is not None:
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
        loudness_source = (
            directory / 'vocals.wav'
            if (directory / 'vocals.wav').is_file()
            else directory / 'source_audio.wav'
        )
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
        face_report = targets.report
        lap('targets')
    check()

    engine = choose_engine(engine)
    if engine == 'latentsync' and targets_path is None:
        engine = 'wav2lip'
    lips = directory / 'lips.mp4'
    plan = None
    if engine == 'latentsync':
        stop_preview = threading.Event()
        grower = None
        if on_preview and min(_frame_size(normalized)) <= PREVIEW_MAX_SOURCE:
            grower = threading.Thread(
                target=_grow_preview,
                args=(
                    normalized,
                    directory,
                    audio,
                    total_frames,
                    _frame_rate(video),
                    stop_preview,
                    on_preview,
                ),
                name='preview',
                daemon=True,
            )
            grower.start()
        try:
            summary = _render_latentsync(
                normalized,
                sync_audio,
                targets_path,
                lips,
                directory,
                control,
                lambda f: report_progress(0.18 + 0.6 * f, 'Lip sync'),
                spans_only=True,
            )
        finally:
            stop_preview.set()
            if grower:
                grower.join(timeout=120)
        lap('latentsync')
        manifest = json.loads((directory / 'latentsync_cache' / 'spans.json').read_text())
        if settings.latentsync_enhance:
            report_progress(0.8, 'Restoring faces')
            try:
                plan, detail = finish.restore_spans(
                    manifest,
                    targets_path,
                    control,
                    lambda f: report_progress(0.8 + 0.1 * f, 'Restoring faces'),
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001  (restoration is a finish, never a reason to fail)
                logger.warning('Face restoration failed for %s: %s', directory.name, exc)
                plan, detail = finish.plan_restoration(manifest, targets_path, False, False), {
                    'applied': False,
                    'reason': str(exc)[:120],
                }
            summary['mouth_enhanced'] = detail
        else:
            plan = finish.plan_restoration(manifest, targets_path, False, False)
        lap('restore')
    else:
        summary = _render_wav2lip(
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
    final = directory / 'dubbed_video.mp4'
    source_rate = _frame_rate(video)
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
        if source_rate and abs(source_rate - FPS) > 0.01:
            summary['frame_rate'] = round(source_rate, 3)
    elif source_rate and abs(source_rate - FPS) > 0.01:
        # Back to the source frame rate: untouched frames map back to exactly the original pictures (no judder).
        picture = [
            '-vf',
            f'fps={_frame_rate(video, text=True)}',
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-crf',
            '12',
            '-pix_fmt',
            'yuv420p',
        ]
        summary['frame_rate'] = round(source_rate, 3)
    else:
        picture = ['-c:v', 'copy']
    if plan is None:
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
    summary['export'] = validate_video(final, total_frames / FPS)
    lap('export')
    summary['timing'] = {
        **timing,
        'total': round(sum(timing.values()), 1),
    }  # seconds per sub-step of this stage
    (directory / 'lip_sync_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report_progress(1.0, 'Lip sync')
    return final


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
        'frames': _frame_count(video),
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


def _target_widths(targets_path: Path) -> np.ndarray:
    """Width in px of every redrawn face target."""
    with np.load(targets_path) as data:
        track, bbox = data['track'], data['bbox']
    return (bbox[track >= 0, 2] - bbox[track >= 0, 0]) if (track >= 0).any() else np.zeros(0)


def restore_faces(
    lips: Path, targets: Path, output: Path, control=None, progress=None, max_face: int = 0
) -> dict:
    """GFPGAN over the redrawn lower face of every frame narrower than `max_face` px (0 = all)
    (backend/workers/restore_worker.py, own venv)."""
    result = {
        'applied': False,
        'model': 'GFPGAN v1.4',
        'max_face': max_face,
        'temporal': settings.restore_temporal,
    }
    with np.load(targets) as data:
        total = int((data['track'] >= 0).sum())

    def on_line(line):
        if line.startswith('Restored ') and progress and total:
            progress(min(0.99, int(line.split()[1]) / total))
        elif line.startswith('DONE '):
            fields = dict(item.split('=') for item in line.split()[1:])
            result.update(
                applied=int(fields['restored']) > 0,
                frames=int(fields['restored']),
                seconds=float(fields['seconds']),
            )

    command = [
        settings.restore_python,
        '-u',
        str(ROOT / 'backend' / 'workers' / 'restore_worker.py'),
        str(lips),
        str(targets),
        str(output),
        '--weights',
        str(settings.gfpgan_weights),
        '--max-face',
        str(max_face),
        '--temporal',
        str(settings.restore_temporal),
    ]
    _stream(command, ROOT, control, on_line, label='Face restoration')
    return result


def _render_latentsync(
    normalized,
    sync_audio,
    targets_path,
    output,
    directory,
    control,
    progress,
    spans_only: bool = False,
) -> dict:
    voice16 = directory / 'lips_voice16k.wav'
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        sync_audio,
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'pcm_s16le',
        voice16,
    )
    command = [
        settings.latentsync_python,
        '-u',
        str(ROOT / 'backend' / 'workers' / 'latentsync_worker.py'),
        str(normalized),
        str(voice16),
        str(targets_path),
        str(output),
        '--repo',
        str(settings.latentsync_dir),
        '--checkpoints',
        str(settings.latentsync_checkpoints),
        '--vae',
        str(settings.latentsync_vae),
        '--steps',
        str(settings.latentsync_steps),
        '--guidance',
        str(settings.latentsync_guidance),
        '--device',
        settings.device,
        '--cache',
        str(directory / 'latentsync_cache'),
    ]
    if (directory / 'render_source.sha256').is_file():
        command += ['--video-key', (directory / 'render_source.sha256').read_text()]
    if settings.latentsync_deepcache:
        command.append('--deepcache')
    if spans_only:
        command.append('--spans-only')
    command += ['--windows', str(settings.latentsync_windows)]
    result = {
        'synced_frames': 0,
        'render_seconds': 0.0,
        'model': 'LatentSync 1.6',
        'settings': {'steps': settings.latentsync_steps, 'guidance': settings.latentsync_guidance},
    }

    def on_line(line):
        if line.startswith('Rendered '):
            parts = line.split()
            done, total = int(parts[1]), int(parts[3])
            progress(done / max(1, total))
        elif line.startswith('DONE '):
            fields = dict(item.split('=') for item in line.split()[1:])
            result['synced_frames'] = int(fields['synced'])
            result['render_seconds'] = float(fields['seconds'])
            result['reused_frames'] = int(fields.get('reused', 0))

    _stream(command, ROOT, control, on_line, label='LatentSync')
    return result


def _target_arrays(targets_path):
    if not targets_path:
        return None
    with np.load(targets_path) as data:
        return {key: data[key] for key in ('bbox', 'lmk', 'track')}


def _slice_digest(arrays, start: int, stop: int) -> str:
    if arrays is None:
        return 'detector'
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(np.ascontiguousarray(arrays[key][start:stop]).tobytes())
    return digest.hexdigest()


def _render_wav2lip(
    normalized,
    sync_audio,
    targets_path,
    output,
    directory,
    total_frames,
    checkpoint,
    enhance,
    control,
    progress,
) -> dict:
    checkpoint = checkpoint or settings.wav2lip_accurate_checkpoint
    if not checkpoint.is_file():
        checkpoint = settings.wav2lip_checkpoint
    enhance = bool(enhance and settings.esrgan_weights.is_file())
    chunk_frames = max(1, round(settings.wav2lip_chunk_seconds * FPS))
    count = math.ceil(total_frames / chunk_frames)
    pictures, reports = [], []
    targets = _target_arrays(targets_path)
    source_digest = (
        (directory / 'render_source.sha256').read_text()
        if (directory / 'render_source.sha256').is_file()
        else ''
    )
    for index, start_frame in enumerate(range(0, total_frames, chunk_frames)):
        if control:
            control.check()
        frames = min(chunk_frames, total_frames - start_frame)
        face = directory / f'lip_face_{index:04d}.mp4'
        sound = directory / f'lip_audio_{index:04d}.wav'
        synced = directory / f'lip_done_{index:04d}.mp4'
        report_path = directory / f'lip_report_{index:04d}.json'
        picture = directory / f'lip_picture_{index:04d}.mp4'
        progress(index / count)
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-ss',
            start_frame / FPS,
            '-i',
            sync_audio,
            '-t',
            frames / FPS + 0.2,
            '-ar',
            '16000',
            '-ac',
            '1',
            '-c:a',
            'pcm_s16le',
            '-map_metadata',
            '-1',
            '-fflags',
            '+bitexact',
            sound,
        )
        # A chunk is re-rendered only when its own picture, voice or speaking-face targets changed.
        stamp = {
            'video': source_digest,
            'targets': _slice_digest(targets, start_frame, start_frame + frames),
            'audio': _digest(sound),
            'checkpoint': checkpoint.name,
            'enhance': enhance,
            'start': start_frame,
            'frames': frames,
        }
        if report_path.is_file() and synced.is_file():
            cached = json.loads(report_path.read_text(encoding='utf-8'))
            if cached.get('stamp') == stamp and _frame_count(synced) == frames:
                reports.append(cached)
                run(
                    'ffmpeg',
                    '-y',
                    '-v',
                    'error',
                    '-i',
                    synced,
                    '-map',
                    '0:v:0',
                    '-an',
                    '-c:v',
                    'copy',
                    picture,
                )
                pictures.append(picture)
                continue
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-ss',
            start_frame / FPS,
            '-i',
            normalized,
            '-frames:v',
            frames,
            '-an',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-crf',
            '14',
            face,
        )
        command = [
            settings.wav2lip_python,
            '-u',
            'inference.py',
            '--checkpoint_path',
            str(checkpoint),
            '--face',
            str(face),
            '--audio',
            str(sound),
            '--outfile',
            str(synced),
            '--report',
            str(report_path),
            '--threads',
            str(settings.wav2lip_threads),
            '--face_det_interval',
            str(settings.wav2lip_face_interval),
            '--runtime',
            settings.lip_sync_runtime,
            '--openvino_device',
            settings.openvino_device,
            '--resize_factor',
            str(settings.wav2lip_resize_factor),
        ]
        if targets_path:
            command += ['--targets', str(targets_path), '--target_offset', str(start_frame)]
        if enhance:
            command += ['--enhance_weights', str(settings.esrgan_weights)]
        env = {
            **os.environ,
            'NUMBA_DISABLE_JIT': '1',
            'OMP_NUM_THREADS': str(settings.wav2lip_threads),
        }

        def on_line(line, index=index, frames=frames):
            if line.startswith('Rendered '):
                progress((index + min(int(line.split()[1]) / frames, 0.999)) / count)

        _stream(command, ROOT / 'Wav2Lip', control, on_line, env=env, label='Wav2Lip')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['frames'] != frames:
            raise RuntimeError(f'Lip-sync chunk {index + 1} lost frames')
        report['stamp'] = stamp
        report_path.write_text(json.dumps(report), encoding='utf-8')
        reports.append(report)
        # Drop per-chunk AAC padding before concatenating so later chunks stay aligned with the speech.
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-i',
            synced,
            '-map',
            '0:v:0',
            '-an',
            '-c:v',
            'copy',
            picture,
        )
        pictures.append(picture)
    concat = directory / 'lip_chunks.txt'
    concat.write_text('\n'.join(f"file '{path.name}'" for path in pictures), encoding='utf-8')
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-f',
        'concat',
        '-safe',
        '0',
        '-i',
        concat,
        '-c:v',
        'copy',
        output,
    )
    return {
        'synced_frames': sum(r['synced_frames'] for r in reports),
        'model': checkpoint.stem,
        'render_seconds': round(sum(r.get('render_seconds', 0) for r in reports), 2),
        'mouth_enhanced': enhance,
        'detector': 'studio targets' if targets_path else 'SFD',
    }
