"""Run the full dubbing pipeline and persist job progress.

Stages: source -> audio -> speakers -> transcribe -> translate -> voice -> mix -> lipsync.
Completed work is cached inside the job folder so a stopped job resumes without
repeating paid model requests, and a finished job can be re-voiced cheaply.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock

from backend.speech.asr import (
    DegenerateTranscript,
    degenerate,
    transcribe,
    script_mismatch,
    guess_source_language,
)
from backend.media.audio_merge import assemble_timeline
from backend.providers.bodhan_api import BodhanAPI, BodhanBudgetError, BodhanRateLimitError
from backend.config import ROOT, settings
from backend.speakers.diarization import refine_turns, speaker_turns
from backend.rendering.lip_models import checkpoint_for
from backend.speakers.gender import classify_genders, decide_gender, split_mixed_speakers
from backend.jobs.gpus import pool
from backend.rendering.lip_sync import (
    voice_only,
    precompute_faces,
    speaker_face_evidence,
    synchronize,
)
from backend.providers import openrouter
from backend.voices.library import speaking_rate
from backend.speech.fitting import DEFAULT_SPEAKING_RATE, SPEAKING_RATE, fit_lines, prefit_lines
from backend.media.timeline import HARD_MIN_SPEED, dubbed_slots, placement_notes
from backend.speech.llm_translation import translate_lines
from backend.speech.openrouter_tts import OPENROUTER_LANGUAGES
from backend.speech.styles import TONE_TO_STYLE, normalise_style
from backend.providers.usage import Usage
from backend.voices.engine import VoiceState, voice_references
from backend.models import ACTIVE_STATUSES, Segment
from backend.media.separation import (
    extract_full_audio,
    separate_background,
    separation_available,
    separation_stamp,
)
from backend.speakers.speaker_profiles import speaker_profiles
from backend.speech.script_supervisor import check_transcripts, post_edit, profile_content
from backend.speakers.speaker_review import review_speakers
from backend.media.video_input import (
    SourceError,
    run,
    cut_audio,
    download_video,
    extract_audio,
    make_poster,
    write_preview,
    prepare_source,
    probe,
    duration,
    run,
)
from backend.speech.translation import translate
from backend.voices.assignment import assign_voices, overlapping_turns, voice_reuse_warnings

logger = logging.getLogger(__name__)

STAGES = [
    ("source", "Fetching source"),
    ("audio", "Extracting audio"),
    ("speakers", "Finding speakers"),
    ("transcribe", "Transcribing"),
    ("translate", "Translating"),
    ("voice", "Generating voices"),
    ("mix", "Mixing audio"),
    ("lipsync", "Lip sync"),
]
TURNS_VERSION = 6  # bump when diarization post-processing changes, so resumed jobs re-run it
FACE_EVIDENCE_WAIT = 300  # seconds to wait for the background face pass before voicing
STAGE_PROGRESS = {
    "source": (2, 12),
    "audio": (12, 15),
    "speakers": (15, 22),
    "transcribe": (22, 44),
    "translate": (44, 58),
    "voice": (58, 78),
    "mix": (78, 84),
    "lipsync": (84, 100),
}
RETAINED_FILES = {
    "status.json",
    "dubbed_video.mp4",
    "preview_video.mp4",
    "dubbed_audio.wav",
    "dubbed_voice.wav",
    "lip_sync_report.json",
    "source.mp4",
    "source_audio.wav",
    "speaker_turns.json",
    "speaker_turns_raw.json",
    "poster.jpg",
    "background.wav",
    "vocals.wav",
    "separation.txt",
    "faces.npz",
    "faces.sha256",
    "render_source.mp4",
    "render_source.sha256",
    "render_source.mp4.digest",
    "source.mp4.digest",
    "targets.npz",
    "latentsync_cache",
    "asd.npz",
    "asd.sha256",
    "versions",
    "takes",
    "shorts",
    "gender.json",
    "gender.sha256",
    "hands.npz",
    "hands.sha256",
}
BACKGROUND_MODES = {"separate", "duck", "none"}
RETAINED_PREFIXES = (
    "tts_",
    "ref_",
    "lip_done_",
    "lip_report_",
)  # voices and rendered lip-sync chunks are reused on edits


def _plain(value):
    """NumPy scalars as Python values: a fresh dub's timings come from diarization as NumPy floats, so anything computed
    from them (a comparison gives numpy.bool) must still be storable. Anything else is a real bug and still raises.
    """
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persist job records atomically and serialize mutations within this process."""

    def __init__(self):
        self.lock = Lock()
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        return settings.jobs_dir / job_id / "status.json"

    def save(self, job_id: str, **changes):
        """Merge changes into a job record and replace its JSON file atomically."""
        path = self.path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            now = utcnow()
            data = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {
                    "id": job_id,
                    "status": "queued",
                    "progress": 0,
                    "stage": "Queued",
                    "stage_key": None,
                    "segments": [],
                    "created_at": now,
                }
            )
            data.update(changes)
            data["updated_at"] = now
            temp = path.with_suffix(".tmp")
            temp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=_plain), encoding="utf-8"
            )
            temp.replace(path)
        return data

    def get(self, job_id: str):
        """Read one job record, returning None for a missing or unreadable record."""
        path = self.path(job_id)
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def version(self, job_id: str) -> int:
        try:
            return self.path(job_id).stat().st_mtime_ns
        except OSError:
            return 0

    def list(self):
        with self.lock:
            jobs = []
            for directory in settings.jobs_dir.iterdir():
                path = directory / "status.json"
                if not directory.is_dir() or not path.is_file():
                    continue
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                job["segment_count"] = len(job.get("segments") or [])
                job.pop("segments", None)
                jobs.append(job)
            return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            directory = (settings.jobs_dir / job_id).resolve()
            if (
                directory.parent != settings.jobs_dir.resolve()
                or not (directory / "status.json").is_file()
            ):
                return False
            try:
                if (
                    json.loads((directory / "status.json").read_text(encoding="utf-8")).get(
                        "status"
                    )
                    in ACTIVE_STATUSES
                ):
                    raise RuntimeError("Cancel the job before deleting it")
            except (OSError, json.JSONDecodeError):
                pass
            shutil.rmtree(directory)
            return True

    def clear(self):
        with self.lock:
            root = settings.jobs_dir.resolve()
            direct_children = [path for path in root.iterdir() if path.resolve().parent == root]
            folders = [
                path
                for path in direct_children
                if path.is_dir() and (path / "status.json").is_file()
            ]
            files = [path for path in direct_children if path.is_file()]
            active = []
            for folder in folders:
                status_file = folder / "status.json"
                if status_file.is_file():
                    try:
                        if (
                            json.loads(status_file.read_text(encoding="utf-8")).get("status")
                            in ACTIVE_STATUSES
                        ):
                            active.append(folder.name)
                    except (OSError, json.JSONDecodeError):
                        continue
            if active:
                raise RuntimeError("Wait for active dubbing jobs to finish before clearing history")
            for path in folders:
                shutil.rmtree(path)
            for path in files:
                path.unlink()
            return len(folders) + len(files)


store = JobStore()


class CancelledError(RuntimeError):
    """The job was cancelled by the user."""


class JobControl:
    """Cancellation token plus the child processes that must die with it."""

    def __init__(self):
        self.event = threading.Event()
        self.processes: set = set()
        self.lock = Lock()

    def cancel(self):
        self.event.set()
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    def check(self):
        if self.event.is_set():
            raise CancelledError("Cancelled")

    def terminate_processes(self):
        """Stop child processes (for example a separation still running after a failure)."""
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass

    def register(self, process):
        with self.lock:
            self.processes.add(process)
        if self.event.is_set():
            process.terminate()

    def unregister(self, process):
        with self.lock:
            self.processes.discard(process)


_controls: dict[str, JobControl] = {}
_controls_lock = Lock()
heavy_slots = pool  # one GPU per render slot (backend/jobs/gpus.py)
separation_slots = BoundedSemaphore(
    1
)  # separation is 20 s of light GPU work: it must not queue behind long renders
# Vocal separation overlaps with the network-bound model stages of the same job.
background_executor = ThreadPoolExecutor(
    max_workers=max(1, settings.job_workers), thread_name_prefix="separation"
)


def control_for(job_id: str) -> JobControl:
    with _controls_lock:
        control = _controls.get(job_id)
        if control is None or control.cancelled:
            control = _controls[job_id] = JobControl()
        return control


def release_control(job_id: str):
    with _controls_lock:
        _controls.pop(job_id, None)


def cancel_job(job_id: str) -> bool:
    with _controls_lock:
        control = _controls.get(job_id)
    if control is None:
        return False
    control.cancel()
    return True


def is_running(job_id: str) -> bool:
    with _controls_lock:
        return job_id in _controls


def remove_intermediates(directory: Path):
    """Keep playback files plus everything needed to resume or re-voice cheaply."""
    for artifact in directory.iterdir():
        if artifact.name in RETAINED_FILES or artifact.name.startswith(RETAINED_PREFIXES):
            continue
        try:
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        except OSError:
            pass


def run_segment_stage(segments, worker, max_workers, completed, control: JobControl | None = None):
    """Run independent requests concurrently, keeping stage barriers and order."""

    def guarded(index, segment):
        if control:
            control.check()
        return worker(index, segment)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(guarded, index, segment) for index, segment in enumerate(segments)]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                completed(count)
        except Exception:
            for future in futures:
                future.cancel()
            raise


def heavy_stage(control: JobControl, job_id: str, waiting_stage: str):
    """Context manager bounding CPU-heavy work to `RENDER_WORKERS` jobs at a time."""

    class Slot:
        def __enter__(self):
            if not heavy_slots.acquire(blocking=False):
                store.save(job_id, stage=waiting_stage, waiting=True)
                while not heavy_slots.acquire(timeout=1):
                    control.check()
                store.save(job_id, waiting=False)
            return self

        def __exit__(self, *exc):
            heavy_slots.release()
            return False

    return Slot()


def run_job(job_id: str, resume: bool = False):
    """Wait out model quotas while reusing completed stages between attempts."""
    control = control_for(job_id)
    try:
        if (store.get(job_id) or {}).get('status') == 'cancelled':
            return  # cancelled while waiting its turn
        for cycle in range(8):
            process_job(job_id, resume or cycle > 0, control)
            job = store.get(job_id)
            if not job or not job.get('retryable') or job.get('status') != 'failed':
                return
            if cycle == 7:
                return
            delay = 180
            retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            store.save(
                job_id,
                status='running',
                error=None,
                stage='Model rate limit; waiting to retry',
                retry_at=retry_at,
                retry_attempt=cycle + 1,
            )
            for _ in range(delay):
                if control.cancelled:
                    store.save(
                        job_id,
                        status='cancelled',
                        stage='Cancelled',
                        retry_at=None,
                        retryable=False,
                    )
                    return
                time.sleep(1)
    finally:
        release_control(job_id)


def resume_lip_sync(job_id: str):
    """Resume rendering saved speech without repeating paid model requests."""
    control = control_for(job_id)
    directory = settings.jobs_dir / job_id
    try:
        store.save(
            job_id,
            status='running',
            error=None,
            progress=84,
            stage='Resuming lip sync',
            stage_key='lipsync',
        )
        with heavy_stage(control, job_id, 'Waiting for a render slot'):
            _render(job_id, directory, control)
        store.save(
            job_id,
            status='completed',
            progress=100,
            stage='Ready',
            stage_key=None,
            finished_at=utcnow(),
            review_url=None,
        )
        remove_intermediates(directory)
    except CancelledError:
        store.save(job_id, status='cancelled', stage='Cancelled', stage_key=None)
    except Exception as exc:
        logger.exception('Lip sync resume failed for job %s', job_id)
        store.save(job_id, status='failed', stage='Failed', stage_key=None, error=str(exc)[:500])
    finally:
        release_control(job_id)


def _render(job_id: str, directory: Path, control: JobControl):
    low, high = STAGE_PROGRESS['lipsync']
    job = store.get(job_id) or {}
    model, checkpoint = checkpoint_for(job.get('lip_sync_model'))
    enhance = (
        bool(job.get('enhance_mouth', settings.mouth_enhance)) and settings.esrgan_weights.is_file()
    )
    slots = dubbed_slots(job.get('segments') or [])

    def progress(fraction, label):
        store.save(
            job_id,
            progress=round(low + (high - low) * fraction, 1),
            stage=f'{label} {int(fraction * 100)}%',
        )

    store.save(job_id, lip_sync_model=model, enhance_mouth=enhance)

    def on_preview(ready, total):
        store.save(
            job_id,
            review_url=f'/api/jobs/{job_id}/review',
            review_version=utcnow(),
            review_note=f'{ready} of {total} face spans lip-synced so far' if total else '',
        )

    if model == 'none':
        progress(0.5, 'Writing the dub')
        voice_only(directory / 'source.mp4', directory / 'dubbed_audio.wav', directory)
    else:
        synchronize(
            directory / 'source.mp4',
            directory / 'dubbed_audio.wav',
            directory,
            progress,
            control,
            sync_audio=directory / 'dubbed_voice.wav',
            checkpoint=checkpoint,
            enhance=enhance,
            slots=slots,
            engine='latentsync' if model == 'latentsync' else 'wav2lip',
            on_preview=on_preview,
        )
    store.save(job_id, stage='Preparing the preview')
    write_preview(directory / 'dubbed_video.mp4', directory / 'preview_video.mp4')
    report_path = directory / 'lip_sync_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    synced = {
        round(float(line['start']), 1) for line in report.get('lines', []) if line.get('lip_synced')
    }
    lines = job.get('segments') or []
    for line in lines:
        if line.get('voiced_text'):
            line['state'] = 'lip-synced' if round(float(line['start']), 1) in synced else 'voiced'
    store.save(
        job_id,
        video_url=f'/api/jobs/{job_id}/video',
        audio_url=f'/api/jobs/{job_id}/audio',
        lip_sync=report,
        render_version=7,
        segments=lines,
    )
    return synced


def length_targets(lines, speaker_voices: dict, language: str) -> dict:
    """{index: characters} each line should have to fill its slot at its cast voice's pace (the editor writes to it)."""
    out = {}
    for index, segment in enumerate(lines):
        voice = (speaker_voices or {}).get(segment.speaker, "")
        rate = (
            speaking_rate(voice, SPEAKING_RATE.get(language, DEFAULT_SPEAKING_RATE))
            if voice
            else SPEAKING_RATE.get(language, DEFAULT_SPEAKING_RATE)
        )
        out[index] = max(4, round(rate * max(0.05, segment.end - segment.start)))
    return out


def write_review_cut(job_id: str, directory: Path, video: Path, dubbed_audio: Path):
    """The dubbed mix over the original picture (a stream copy, seconds): playable while the lips render."""
    try:
        duration = float(
            json.loads(
                run(
                    'ffprobe',
                    '-v',
                    'error',
                    '-show_entries',
                    'format=duration',
                    '-of',
                    'json',
                    video,
                )
            )['format']['duration']
        )
        partial = directory / 'review_video.part.mp4'
        run(
            'ffmpeg',
            '-y',
            '-v',
            'error',
            '-i',
            video,
            '-i',
            dubbed_audio,
            '-map',
            '0:v:0',
            '-map',
            '1:a:0',
            '-c:v',
            'copy',
            '-c:a',
            'aac',
            '-b:a',
            '160k',
            '-ar',
            '48000',
            '-af',
            'apad',
            '-t',
            duration,
            '-movflags',
            '+faststart',
            partial,
        )
        partial.replace(directory / 'review_video.mp4')
        store.save(
            job_id,
            review_url=f'/api/jobs/{job_id}/review',
            review_version=utcnow(),
            review_note='dubbed voice over the original picture; lips not rendered yet',
        )
    except Exception as exc:  # noqa: BLE001  (a courtesy for the wait, never a reason to fail)
        logger.warning('Review cut for %s skipped: %s', job_id, exc)


def _carry_styles(slots: list[Segment], turns: list[Segment]):
    """Give every final slot the delivery style of the reviewed turn it overlaps most, and the review
    transcripts of all the turns it covers (the second opinion for the transcript check)."""
    for slot in slots:
        best, best_overlap, texts = None, 0.0, []
        for turn in sorted(turns, key=lambda t: t.start):
            overlap = min(slot.end, turn.end) - max(slot.start, turn.start)
            if turn.speaker != slot.speaker or overlap <= 0:
                continue
            if turn.reference_text:
                texts.append(turn.reference_text)
            if overlap > best_overlap:
                best, best_overlap = turn, overlap
        if best is not None and best.style:
            slot.style = best.style
        slot.reference_text = " ".join(texts)


def _separate_in_background(job_id: str, directory: Path, video: Path, control: JobControl) -> Path:
    """Run the vocal separation inside a heavy slot while ASR/translation/TTS proceed."""
    stamp = directory / 'separation.txt'
    if (
        (directory / 'background.wav').is_file()
        and stamp.is_file()
        and stamp.read_text() == separation_stamp()
    ):
        store.save(job_id, background={'mode': 'separate', 'status': 'ready', 'progress': 1})
        return directory / 'background.wav'
    store.save(job_id, background={'mode': 'separate', 'status': 'queued', 'progress': 0})
    while not separation_slots.acquire(timeout=1):
        control.check()
    try:
        control.check()
        last = {'step': -1}

        def progress(fraction):
            step = int(fraction * 25)
            if step != last['step']:
                last['step'] = step
                store.save(
                    job_id,
                    background={
                        'mode': 'separate',
                        'status': 'running',
                        'progress': round(fraction, 2),
                    },
                )

        store.save(job_id, background={'mode': 'separate', 'status': 'running', 'progress': 0})
        result = separate_background(video, directory, control, progress)
        store.save(job_id, background={'mode': 'separate', 'status': 'ready', 'progress': 1})
        return result
    finally:
        separation_slots.release()


def pitch_reader(audio: Path):
    """A function (start, end) -> median pitch in Hz over the original mix, or None when unvoiced."""
    import subprocess
    import numpy as np
    from backend.speakers.speaker_profiles import median_pitch

    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(audio), '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768

    def pitch(start: float, end: float):
        return median_pitch(samples[int(start * 16000) : int(end * 16000)])[0]

    return pitch


def _prepare_faces_in_background(
    job_id: str, directory: Path, video: Path, control: JobControl, separation
):
    """Face analysis and active-speaker detection inside a heavy slot while the API stages run."""
    while not heavy_slots.acquire(timeout=1):
        control.check()
    try:
        control.check()
        precompute_faces(video, directory, control, separation)
    finally:
        heavy_slots.release()


def acquire_source(job: dict, directory: Path, report) -> Path:
    """Reuse a saved source, normalise an upload, or download from the URL."""
    saved = directory / 'source.mp4'
    if saved.is_file():
        return saved
    clip = (job.get('clip_start'), job.get('clip_end')) if job.get('clip_end') else None
    max_seconds = settings.max_video_minutes * 60
    if clip:
        max_seconds = max(
            max_seconds, SECTION_SOURCE_MAX_SECONDS
        )  # the limit applies to the part, the whole may be longer
    info = None
    if job.get('source_type') == 'upload':
        uploads = sorted(directory.glob('upload.*'))
        if not uploads:
            raise SourceError('The uploaded file is no longer available; create the project again')
        raw = uploads[0]
        report('Preparing uploaded video', 0.4)
    else:
        info = download_video(
            job['source_url'],
            directory,
            max_seconds,
            settings.max_upload_mb * 1024 * 1024,
            lambda fraction: report(f'Downloading video {int(fraction * 100)}%', fraction * 0.7),
            section=(float(clip[0]), float(clip[1])) if clip else None,
        )
        raw = info.path
        if info.title and not job.get('title'):
            title = info.title[:120]
            if clip:
                title += f" ({_clock(clip[0])}–{_clock(clip[1])})"
            store.save(job['id'], title=title)
        if info.meta:
            store.save(job['id'], source_meta=info.meta)
        report('Preparing video', 0.75)
    prepare_source(raw, saved, settings.max_video_height)
    if clip and not (
        info is not None and info.sectioned
    ):  # a range download already holds only the part
        report('Cutting the part to dub', 0.9)
        cut_section(saved, clip[0], clip[1])
    return saved


SECTION_SOURCE_MAX_SECONDS = 2 * 3600
# what the script view shows must survive a resume: a re-render rebuilds the lines from the speaker turns
KEPT_ON_RESUME = (
    'machine_translation',
    'asr_text',
    'reference_text',
    'text_source',
    'style_override',
    'take',
    'source_edited',
    'condensed',
    'ai_edited',
    'history',
)


def restore_line(segment: Segment, saved: dict) -> Segment:
    """A line rebuilt from the speaker turns gets back what an earlier run saved for it: its words, voice, edits and history."""
    segment.text = saved.get('text') or ''
    segment.translation = saved.get('translation') or ''
    segment.voice = saved.get('voice') or ''
    segment.voiced_text = saved.get('voiced_text') or ''
    segment.voiced_style = saved.get('voiced_style') or ''
    segment.tts_engine = saved.get('tts_engine') or ''
    segment.edited = bool(saved.get('edited'))
    for key in KEPT_ON_RESUME:
        if saved.get(key):
            setattr(segment, key, saved[key])
    return segment


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def cut_section(video: Path, start: float, end: float):
    """Keep only [start, end] of the prepared source, re-encoded so the cut lands exactly on the frame."""
    total = duration(video)
    end = min(end, total)
    if end - start < 1:
        raise SourceError(
            f"The part to dub ({_clock(start)}–{_clock(end)}) lies outside the video ({_clock(total)} long)"
        )
    cut = video.with_name('source_cut.mp4')
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-ss',
        f'{start:.3f}',
        '-i',
        video,
        '-t',
        f'{end - start:.3f}',
        '-c:v',
        'libx264',
        '-preset',
        'faster',
        '-crf',
        '16',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-movflags',
        '+faststart',
        cut,
    )
    cut.replace(video)


def process_job(job_id: str, resume: bool = False, control: JobControl | None = None):
    """Run one pipeline attempt, reuse saved stages on resume, and persist progress."""
    directory = settings.jobs_dir / job_id
    control = control or control_for(job_id)
    client = None
    stage_started = {'key': None, 'at': time.monotonic()}
    stage_seconds: dict[str, float] = {}

    def mark(key: str | None, label: str, progress: float | None = None, **extra):
        control.check()
        now = time.monotonic()
        previous = stage_started['key']
        if previous and previous != key:
            stage_seconds[previous] = round(
                stage_seconds.get(previous, 0) + now - stage_started['at'], 1
            )
            extra['stage_seconds'] = stage_seconds
        if previous != key:
            stage_started.update(key=key, at=now)
        payload = {'stage': label, 'stage_key': key, **extra}
        if progress is not None:
            payload['progress'] = round(progress, 1)
        store.save(job_id, **payload)

    def span(key: str, fraction: float) -> float:
        low, high = STAGE_PROGRESS[key]
        return low + (high - low) * max(0.0, min(1.0, fraction))

    def add_notes(*notes):
        notes = [note for note in notes if note]
        if notes:
            store.save(
                job_id, review_warnings=(store.get(job_id) or {}).get('review_warnings', []) + notes
            )

    try:
        job = store.get(job_id) or {}
        previous = job if resume else None
        usage = Usage(job.get('usage') if resume else None)
        language, source_language = job['language'], job.get('source_language')
        voice, num_speakers = job.get('voice'), job.get('num_speakers')
        keep_background = job.get('keep_background', settings.keep_background_default)
        background_mode = (
            job.get('background_mode') or (settings.background_mode if keep_background else 'none')
        ).lower()
        if background_mode not in BACKGROUND_MODES:
            background_mode = settings.background_mode
        background_notes: list[str] = []
        separation = None
        overrides = job.get('voice_overrides') or {}
        script = job.get('translation_script') or settings.translation_script
        # Delivery style: project default plus per-speaker overrides (Bodhan Indic-Speak styles).
        delivery_style = normalise_style(
            job.get('delivery_style') or TONE_TO_STYLE.get(job.get('script_tone') or '', 'neutral')
        )
        speaker_styles = {
            speaker: normalise_style(style)
            for speaker, style in (job.get('speaker_styles') or {}).items()
        }
        # stage_seconds is this run only: carrying earlier runs over made resumed jobs look slower than they were
        store.save(
            job_id,
            status="running",
            retryable=False,
            retry_at=None,
            error=None,
            started_at=utcnow(),
            delivery_style=delivery_style,
            translation_script=script,
        )  # this run: the UI's 'took' is the latest dub
        mark('source', 'Fetching source', span('source', 0))
        video = acquire_source(
            job, directory, lambda label, fraction: mark('source', label, span('source', fraction))
        )
        info = probe(video)
        video_duration = info['duration']
        if video_duration > settings.max_video_minutes * 60:
            raise RuntimeError(f"Video exceeds the {settings.max_video_minutes}-minute limit")
        poster = directory / 'poster.jpg'
        if not poster.is_file():
            try:
                make_poster(video, poster, min(1.0, video_duration / 3))
            except RuntimeError:
                logger.warning('Poster frame failed for %s', job_id)
        mark(
            'audio',
            'Extracting source audio',
            span('audio', 0),
            duration=round(video_duration, 2),
            width=info['width'],
            height=info['height'],
            source_url_ready=True,
            poster_url=f'/api/jobs/{job_id}/poster' if poster.is_file() else None,
            source_video_url=f'/api/jobs/{job_id}/source',
        )
        source_audio = directory / "source_audio.wav"
        if not (resume and source_audio.is_file()):
            extract_audio(video, source_audio)
        mark('speakers', 'Finding speakers', span('speakers', 0))
        result_path = directory / 'speaker_turns.json'
        result = None
        if resume and result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if (
                    result.get('num_speakers') != num_speakers
                    or result.get('version') != TURNS_VERSION
                ):
                    result = None
            except (OSError, json.JSONDecodeError):
                result = None
        dialogue = source_audio
        if (
            result is None
            and settings.diarize_dialogue_stem
            and separation_available()
            and video_duration <= settings.separation_max_minutes * 60
        ):
            # Speakers are found on the dialogue stem: music and effects confuse diarization in films and trailers.
            mark('speakers', 'Separating dialogue from music', span('speakers', 0))
            try:
                _separate_in_background(job_id, directory, video, control)
                if (directory / 'vocals.wav').is_file():
                    dialogue = directory / 'dialogue16k.wav'
                    extract_audio(directory / 'vocals.wav', dialogue)
            except CancelledError:
                raise
            except Exception as exc:
                control.check()
                logger.warning(
                    'Dialogue separation failed for %s (%s); diarizing the full mix', job_id, exc
                )
        if result is None:
            with heavy_stage(control, job_id, 'Waiting for a free analysis slot'):
                control.check()
                raw_path = directory / 'speaker_turns_raw.json'
                if settings.diarization_backend == 'diarizen' and settings.diarizen_python:
                    command = [
                        settings.diarizen_python,
                        '-u',
                        str(ROOT / 'backend' / 'workers' / 'diarizen_worker.py'),
                        str(dialogue),
                        str(raw_path),
                        '--model',
                        str(settings.diarizen_model),
                        '--embedding',
                        str(settings.diarizen_embedding),
                        '--speakers',
                        str(num_speakers or 0),
                        '--device',
                        settings.device,
                    ]
                elif settings.diarization_backend in {'sherpa', 'diarizen'}:
                    command = [
                        sys.executable,
                        '-u',
                        '-m',
                        'backend.workers.diarization_worker',
                        str(source_audio),
                        str(raw_path),
                        str(num_speakers or 0),
                    ]
                else:
                    command = None
                if command:
                    with subprocess.Popen(
                        command,
                        cwd=settings.jobs_dir.parent,
                        env=pool.env(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    ) as process:
                        control.register(process)
                        tail = []
                        try:
                            for line in process.stdout:
                                tail = (tail + [line])[-20:]
                                if line.startswith('PROGRESS '):
                                    _, done, total = line.split()
                                    if int(total):
                                        mark(
                                            'speakers',
                                            f'Finding speakers {done}/{total}',
                                            span('speakers', int(done) / int(total)),
                                        )
                            code = process.wait()
                        finally:
                            control.unregister(process)
                        control.check()
                        if code:
                            raise RuntimeError(
                                'Diarization worker failed: ' + ''.join(tail)[-1000:]
                            )
                    result = json.loads(raw_path.read_text(encoding='utf-8'))
                    raw_turns = [
                        Segment(t['start'], t['end'], t['speaker']) for t in result['segments']
                    ]
                    if dialogue != source_audio:
                        result['mode'] += ' on the dialogue stem'
                    if 'DiariZen' in result.get('mode', ''):
                        reviewed = None
                        if settings.speaker_review and raw_turns:
                            mark('speakers', 'Checking who says each line', span('speakers', 0.9))
                            reviewed = review_speakers(
                                raw_turns, dialogue, directory / 'review', source_language, usage
                            )
                        if reviewed:
                            raw_turns, result['review'] = reviewed
                            result['mode'] += ' + speaker review'
                        slots = refine_turns(raw_turns, dialogue)
                        _carry_styles(slots, raw_turns)
                        result['segments'] = [t.public() for t in slots]
                else:
                    segments, diarization_mode = speaker_turns(
                        source_audio, num_speakers=num_speakers
                    )
                    result = {
                        'mode': diarization_mode,
                        'segments': [segment.public() for segment in segments],
                    }
                turns = [
                    Segment(**{k: v for k, v in item.items() if k in Segment.__dataclass_fields__})
                    for item in result['segments']
                ]
                try:
                    classified = classify_genders(source_audio, turns, directory, control)
                    heard_genders = {
                        s: (v or {}).get('gender')
                        for s, v in (((result.get('review') or {}).get('voices')) or {}).items()
                    }
                    split = split_mixed_speakers(
                        turns, classified, heard_genders, pitch_of=pitch_reader(source_audio)
                    )
                    if split:
                        logger.info('Split mixed speakers for %s: %s', job_id, split)
                        result['segments'] = [t.public() for t in turns]
                        result['split_speakers'] = split
                except Exception:
                    logger.exception('Gender classification failed for %s', job_id)
                    classified = {}
                try:
                    result['profiles'] = speaker_profiles(dialogue, turns)
                except Exception:
                    logger.exception('Speaker profiling failed for %s', job_id)
                    result['profiles'] = {}
                reviewed_voices = (result.get('review') or {}).get('voices') or {}
                for new_id, old_id in (result.get('split_speakers') or {}).items():
                    gender = (classified.get(new_id) or {}).get('gender')
                    result['profiles'].setdefault(new_id, {})['name'] = (
                        f"Second voice ({gender})" if gender else "Second voice"
                    )
                    result['profiles'][new_id]['split_from'] = old_id
                for speaker, profile in result['profiles'].items():
                    heard = (reviewed_voices.get(speaker) or {}).get('gender')
                    profile.update(decide_gender(profile, classified.get(speaker), heard))
                for speaker, voice_info in reviewed_voices.items():
                    if voice_info.get('name'):
                        result['profiles'].setdefault(speaker, {})['name'] = voice_info['name']
                result['num_speakers'] = num_speakers
                result['version'] = TURNS_VERSION
                result_path.write_text(json.dumps(result), encoding='utf-8')
        segments = [
            Segment(
                **{key: value for key, value in item.items() if key in Segment.__dataclass_fields__}
            )
            for item in result['segments']
        ]
        diarization_mode = result['mode']
        profiles = result.get('profiles') or {}
        if not segments:
            raise RuntimeError("No speech was detected in the source video")
        cached_segments = {
            (item['start'], item['end'], item['speaker']): item
            for item in (previous or {}).get('segments', [])
        }
        for segment in segments:
            restore_line(
                segment, cached_segments.get((segment.start, segment.end, segment.speaker), {})
            )
        speakers = list(dict.fromkeys(segment.speaker for segment in segments))
        if not source_language:
            # the reviewer wrote the lines independently: Latin letters mean English speech, and Bodhan ASR needs that hint
            # (without it, English comes back as Telugu or Tamil letters spelling the sounds)
            source_language = guess_source_language([s.reference_text for s in segments])
            if source_language:
                store.save(job_id, source_language=source_language)
                job['source_language'] = source_language
        content = job.get('script_profile') or {}
        if not content and settings.script_supervision:
            heard = [
                Segment(s.start, s.end, s.speaker, text=s.reference_text)
                for s in segments
                if s.reference_text
            ]
            content = profile_content(
                heard, job.get('source_meta') or {}, job.get('title') or '', source_language, usage
            )
            if content:
                store.save(job_id, script_profile=content)
        voice_reasons = {}
        speaker_voices = assign_voices(
            speakers,
            language,
            voice,
            profiles,
            overrides,
            genre=content.get('genre'),
            reasons=voice_reasons,
            preferred_voices=job.get('voices') or None,
        )
        store.save(job_id, voice_reasons=voice_reasons)
        if background_mode == 'separate':
            if not separation_available():
                background_mode = 'duck'
                background_notes.append(
                    'Vocal separation is not installed; the original mix was ducked under the dub instead.'
                )
            elif video_duration > settings.separation_max_minutes * 60:
                background_mode = 'duck'
                background_notes.append(
                    f'Vocal separation is skipped above {settings.separation_max_minutes} minutes; '
                    'the original mix was ducked under the dub instead.'
                )
        if background_mode == 'separate':
            separation = background_executor.submit(
                _separate_in_background, job_id, directory, video, control
            )
        else:
            store.save(
                job_id,
                background={
                    'mode': background_mode,
                    'status': 'ready' if background_mode == 'duck' else 'off',
                    'progress': 1,
                },
            )
        # Faces and active speakers depend only on the source, so they are found now, while the script is
        # transcribed, translated and voiced; the lip-sync stage then starts straight at the render.
        voice_only_dub = (job.get('lip_sync_model') or '').lower() == 'none'
        faces_ahead = (
            None
            if voice_only_dub
            else background_executor.submit(
                _prepare_faces_in_background, job_id, directory, video, control, separation
            )
        )
        mark(
            'transcribe',
            'Transcribing speech',
            span('transcribe', 0),
            diarization_mode=diarization_mode,
            speaker_count=len(speakers),
            speaker_profiles=profiles,
            speaker_voices=speaker_voices,
            review_warnings=overlapping_turns(segments) + voice_reuse_warnings(speaker_voices),
        )
        client = BodhanAPI(
            on_rate_limit=lambda delay: store.save(
                job_id, stage=f'Bodhan model busy (rate limit); retrying in {delay:g} s'
            )
        )
        workers = settings.model_request_workers
        streaming = (
            job.get('pipeline_mode') or 'staged'
        ) == 'streaming'  # each line is translated the moment it is heard
        content_early = job.get('script_profile') or {}

        # ---- Bodhan Indic-Transcribe, one turn per request ----
        asr_state: dict = {}

        # Transcribe from the dialogue stem when there is one: the words without the music.
        asr_audio = (
            directory / 'dialogue16k.wav'
            if (directory / 'dialogue16k.wav').is_file()
            else source_audio
        )

        def transcribe_segment(index, segment):
            snippet = directory / f"source_segment_{index:04d}.wav"
            cut_audio(asr_audio, segment.start, segment.end, snippet)
            try:
                segment.text = transcribe(client, snippet, source_language, usage, asr_state)
                segment.text_source = ''
                if segment.reference_text and script_mismatch(segment.text, segment.reference_text):
                    # English sounds spelled in an Indic script: the reviewer's words are the speech, Bodhan's are not
                    logger.warning(
                        'Script mismatch at %.1fs in %s: %r vs %r; using the reviewer transcript',
                        segment.start,
                        job_id,
                        segment.text[:30],
                        segment.reference_text[:30],
                    )
                    segment.asr_text, segment.text, segment.text_source = (
                        segment.text,
                        segment.reference_text,
                        'review',
                    )
            except DegenerateTranscript as exc:
                # Bodhan looped on one token. The reviewer heard the same slot independently: use its words, and say so.
                if not segment.reference_text:
                    raise RuntimeError(
                        f"Bodhan returned unusable text for the line at {segment.start:.1f}s and no second "
                        "transcript exists; try again later"
                    ) from exc
                logger.warning(
                    'Using the reviewer transcript for %s at %.1fs: Bodhan returned %r',
                    job_id,
                    segment.start,
                    str(exc)[:30],
                )
                segment.text, segment.text_source = segment.reference_text, 'review'
            finally:
                snippet.unlink(missing_ok=True)
            segment.state = 'heard'
            if streaming and segment.text and not segment.translation:
                segment.translation = translate(
                    client, segment.text, language, script, source_language, usage
                )
                segment.state = 'translated'
                if settings.script_supervision:  # enhanced with the lines heard so far as context
                    so_far = [
                        piece for piece in segments[: index + 1] if piece.text and piece.translation
                    ]
                    post_edit(
                        so_far,
                        language,
                        source_language,
                        script,
                        usage,
                        only=[segment],
                        content=content_early,
                        targets={
                            len(so_far) - 1: length_targets([segment], speaker_voices, language)[0]
                        },
                    )

        remaining = [
            segment for segment in segments if not segment.text or degenerate(segment.text)
        ]  # junk cached text is redone
        cached_count = len(segments) - len(remaining)
        total = len(segments)

        def transcribed(count):
            done = cached_count + count
            mark(
                'transcribe',
                f"{'Hearing and translating' if streaming else 'Transcribing'} {done} of {total}",
                span('transcribe', done / total),
                segments=[piece.public() for piece in segments if piece.text],
            )  # every line, so the script grows as it is heard

        if cached_count:
            transcribed(0)
        run_segment_stage(remaining, transcribe_segment, workers, transcribed, control)
        if remaining and settings.script_supervision and settings.transcript_check:
            mark('transcribe', 'Checking the transcript', span('transcribe', 1))
            check_transcripts(segments, source_language, usage, only=remaining)
        store.save(
            job_id,
            asr_engine_used=asr_state.get('engine') or job.get('asr_engine_used') or 'bodhan',
            usage=usage.as_dict(),
        )

        spoken = [segment for segment in segments if segment.text]
        if not spoken:
            raise RuntimeError("Bodhan ASR returned no speech for this video")
        mark(
            'translate',
            'Translating speech',
            span('translate', 0),
            segments=[piece.public() for piece in spoken],
        )

        # ---- Bodhan Indic-Translate, one line per request ----
        def translate_segment(index, segment):
            segment.translation = translate(
                client, segment.text, language, script, source_language, usage
            )
            segment.state = 'translated'

        untranslated = [segment for segment in spoken if not segment.translation]
        translated_count = len(spoken) - len(untranslated)

        def translated_progress(count):
            done = translated_count + count
            mark(
                'translate',
                f'Translating {done} of {len(spoken)}',
                span('translate', done / len(spoken)),
                segments=[piece.public() for piece in spoken],
            )

        if translated_count:
            translated_progress(0)
        translation_engine = 'bodhan'
        try:
            run_segment_stage(
                untranslated, translate_segment, workers, translated_progress, control
            )
        except BodhanRateLimitError as exc:
            pending = [segment for segment in untranslated if not segment.translation]
            if not openrouter.covers_bodhan() or not pending:
                raise
            logger.warning(
                'Bodhan translation unavailable for %s (%s); OpenRouter translates the rest',
                job_id,
                exc,
            )
            lines = [
                {
                    'id': index,
                    'speaker': segment.speaker,
                    'text': segment.text,
                    'seconds': segment.end - segment.start,
                }
                for index, segment in enumerate(pending)
            ]
            results, _ = translate_lines(
                lines,
                language,
                profiles,
                lambda done, total: mark(
                    'translate',
                    f'Translating scene {done} of {total}',
                    span(
                        'translate', (translated_count + len(pending) * done / total) / len(spoken)
                    ),
                ),
                script=script,
                usage=usage,
            )
            for index, segment in enumerate(pending):
                segment.translation = results[index]
            translation_engine = 'openrouter'
            translated_progress(len(pending))
        content = job.get('script_profile') or {}
        if untranslated and settings.script_supervision:
            if not content:
                mark('translate', 'Reading the content', span('translate', 0.95))
                content = profile_content(
                    spoken,
                    job.get('source_meta') or {},
                    job.get('title') or '',
                    source_language,
                    usage,
                )
                if content:
                    store.save(job_id, script_profile=content)
            mark('translate', 'Editing the translation', span('translate', 1))
            post_edit(
                spoken,
                language,
                source_language,
                script,
                usage,
                only=untranslated,
                content=content,
                targets=length_targets(spoken, speaker_voices, language),
            )
        store.save(job_id, translation_engine_used=translation_engine, usage=usage.as_dict())

        translated = [segment for segment in spoken if segment.translation]
        if not translated:
            raise RuntimeError("Bodhan translation returned no text")
        # ---- faces against voices: a person on screen settles gender and undoes a false split, automatically ----
        if faces_ahead is not None and not faces_ahead.done():
            mark('voice', 'Checking faces against voices', span('voice', 0))
        try:
            if faces_ahead is not None:
                faces_ahead.result(timeout=FACE_EVIDENCE_WAIT)
            evidence = speaker_face_evidence(
                video, directory, [(s.start, s.end, s.speaker) for s in translated], control
            )
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  (faces are an improvement, never a requirement)
            logger.warning(
                'Face evidence unavailable for %s: %s',
                job_id,
                str(exc)
                or f'{type(exc).__name__}: the face analysis took longer than {FACE_EVIDENCE_WAIT} s; voices kept as cast',
            )
            evidence = {}
        changed = False
        for split_id, origin in list((result.get('split_speakers') or {}).items()):
            same_face = evidence.get(split_id, {}).get('identity') is not None and evidence.get(
                split_id, {}
            ).get('identity') == evidence.get(origin, {}).get('identity')
            if same_face and split_id in profiles:
                logger.info(
                    'Merging %s back into %s for %s: one face speaks both', split_id, origin, job_id
                )
                for segment in segments:
                    if segment.speaker == split_id:
                        segment.speaker = origin
                merged = profiles.pop(split_id, {})
                profiles[origin]['speech_seconds'] = round(
                    (profiles[origin].get('speech_seconds') or 0)
                    + (merged.get('speech_seconds') or 0),
                    2,
                )
                profiles[origin]['turns'] = (profiles[origin].get('turns') or 0) + (
                    merged.get('turns') or 0
                )
                result['split_speakers'].pop(split_id, None)
                changed = True
        for speaker, profile in profiles.items():
            seen = evidence.get(speaker)
            if seen and seen.get('face_gender'):
                before = profile.get('gender')
                heard = (
                    ((result.get('review') or {}).get('voices') or {})
                    .get(speaker, {})
                    .get('gender')
                )
                classified_turns = (
                    profile.get('gender_votes')
                    and {
                        'gender': profile['gender'],
                        'confidence': profile.get('gender_confidence', 0),
                    }
                ) or None
                profile.update(decide_gender(profile, classified_turns, heard, seen))
                profile['face_gender'] = seen['face_gender']
                changed = changed or profile.get('gender') != before
        if changed:
            speakers = list(dict.fromkeys(segment.speaker for segment in segments))
            voice_reasons = {}
            speaker_voices = assign_voices(
                speakers,
                language,
                voice,
                profiles,
                overrides,
                genre=content.get('genre'),
                reasons=voice_reasons,
                preferred_voices=job.get('voices') or None,
            )
            result['segments'] = [s.public() for s in segments]
            result['profiles'] = profiles
            result_path.write_text(json.dumps(result), encoding='utf-8')
            store.save(
                job_id,
                speaker_profiles=profiles,
                speaker_voices=speaker_voices,
                voice_reasons=voice_reasons,
                speaker_count=len(speakers),
                segments=[piece.public() for piece in segments],
            )
        elif evidence:
            store.save(job_id, speaker_profiles=profiles)
        mark(
            'voice',
            'Generating voices',
            span('voice', 0),
            segments=[piece.public() for piece in translated],
        )
        # Assign voices before concurrent requests so completion order never
        # changes a speaker's voice or the final timeline order.
        for segment in translated:
            segment.voice = speaker_voices[segment.speaker]

        def style_of(segment):
            # the user's direction for this line wins, then the speaker's or the project's style, then the line's own emotion
            if segment.style_override:
                return normalise_style(segment.style_override)
            if segment.speaker in speaker_styles:
                return speaker_styles[segment.speaker]
            return delivery_style if delivery_style != 'neutral' else normalise_style(segment.style)

        voice_mode = (
            'clone'
            if job.get('tts_engine') == 'clone'
            and openrouter.configured()
            and language in OPENROUTER_LANGUAGES
            else 'auto'
        )
        references = {}
        if voice_mode == 'clone':
            clean = None
            if separation is not None:
                mark('voice', 'Waiting for clean voice samples', span('voice', 0))
                try:
                    separation.result()
                    clean = (
                        directory / 'vocals.wav' if (directory / 'vocals.wav').is_file() else None
                    )
                except CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Separation failed before cloning for %s: %s', job_id, exc)
            references = voice_references(
                directory, segments, speakers, clean or extract_full_audio(video, directory)
            )
        voices = VoiceState(client, language, speaker_voices, voice_mode, references, usage)

        def voice_segment(_, task):
            index, segment = task
            speech = directory / f"tts_{index:04d}.wav"
            segment.tts_engine = voices.render(
                segment.speaker, segment.translation, speech, style_of(segment)
            )
            segment.audio_path = str(speech)
            segment.voiced_text = segment.translation
            segment.voiced_style = style_of(segment)
            segment.state = 'voiced'

        unvoiced = []
        previous_engine = (previous or {}).get('tts_engine_used') or ''
        for index, segment in enumerate(translated):
            speech = directory / f"tts_{index:04d}.wav"
            saved = cached_segments.get((segment.start, segment.end, segment.speaker), {})
            voiced_text = saved.get('voiced_text') or saved.get('translation')
            cached_engine = saved.get('tts_engine') or previous_engine
            mode_ok = (cached_engine == 'clone') == (voice_mode == 'clone')
            if (
                resume
                and voiced_text == segment.translation
                and saved.get('voice') == segment.voice
                and (saved.get('voiced_style') or 'neutral') == style_of(segment)
                and mode_ok
                and speech.is_file()
                and speech.stat().st_size > 44
            ):
                segment.audio_path = str(speech)
                segment.voiced_text = segment.translation
                segment.voiced_style = style_of(segment)
                segment.tts_engine = cached_engine or 'bodhan'
                segment.state = 'voiced'
                if voices.engine is None and cached_engine:
                    voices.engine = cached_engine
            else:
                unvoiced.append((index, segment))
        # Size new lines to their slots before voicing them (text rewrites are cheap, re-voicing is paid).
        if unvoiced and settings.timing_fit:
            mark('voice', 'Fitting lines to their timing', span('voice', 0))
            prefit_lines(
                [segment for _, segment in unvoiced],
                language,
                usage,
                source_language,
                context=translated,
            )
        voiced_count = len(translated) - len(unvoiced)

        def voiced_progress(count):
            done = voiced_count + count
            mark(
                'voice',
                f'Voicing {done} of {len(translated)}',
                span('voice', 0.8 * done / len(translated)),
                segments=[piece.public() for piece in translated],
            )

        if voiced_count:
            voiced_progress(0)
        run_segment_stage(unvoiced, voice_segment, workers, voiced_progress, control)
        tts_used = voices.engine or previous_engine or 'bodhan'
        # One engine per project: if Bodhan dropped out mid-way, re-voice the other lines with the engine in use.
        mixed = [
            (index, segment)
            for index, segment in enumerate(translated)
            if segment.tts_engine and segment.tts_engine != tts_used
        ]
        if mixed:
            mark(
                'voice',
                f'Re-voicing {len(mixed)} lines for a consistent voice',
                span('voice', 0.82),
            )
            run_segment_stage(mixed, voice_segment, workers, lambda count: None, control)

        # ---- fit every line to its original slot (0.9x - 1.1x), rewriting and re-voicing as needed ----
        def render_line(segment):
            segment.tts_engine = voices.render(
                segment.speaker, segment.translation, Path(segment.audio_path), style_of(segment)
            )
            segment.voiced_text = segment.translation

        def fit_progress(round_index, count):
            mark(
                'voice',
                f'Fitting {count} lines to their timing (round {round_index + 1})',
                span('voice', 0.86 + 0.04 * round_index),
            )

        fit = fit_lines(
            translated,
            directory,
            language,
            render_line,
            workers,
            run_segment_stage,
            control,
            fit_progress,
            usage,
            source_language,
        )

        def fit_note(piece, ratio):
            who = f"{piece.speaker.replace('_', ' ').title()} at {piece.start:.1f}s"
            if piece.edited:
                return f"Your edit of {who} plays at {ratio:.2f}x; {'shorten' if ratio > 1 else 'lengthen'} it for natural pacing"
            if ratio < settings.fit_min_speed:
                pause = max(0.0, (piece.end - piece.start) * (1 - ratio / HARD_MIN_SPEED))
                return f"{who} is a short line: spoken at a natural pace" + (
                    f", ending {pause:.1f}s before the original" if pause >= 0.05 else ""
                )
            return f"{who} needed {ratio:.2f}x to fit its slot"

        if settings.timing_mode == 'exact':

            def fit_note(
                piece, ratio
            ):  # noqa: F811  (lines always fill their slot; only the pace is worth a note)
                who = f"{piece.speaker.replace('_', ' ').title()} at {piece.start:.1f}s"
                if ratio > settings.pace_max:
                    return f"{who} is long for its slot: played at {settings.pace_max:.2f}x and allowed to run into the pause after it"
                return f"{who} is short for its slot: played at {settings.pace_min:.2f}x and ends a little early"

        fit_notes = [fit_note(piece, ratio) for piece, ratio in fit.outside]
        store.save(
            job_id,
            tts_engine_used=tts_used,
            speaker_engine_voices=voices.labels(),
            fit_rounds=fit.rounds,
            fit_summary={
                'in_band': fit.in_band,
                'total': fit.total,
                'rewritten': fit.rewritten,
                'band': [settings.fit_min_speed, settings.fit_max_speed],
            },
            usage=usage.as_dict(),
        )
        bed = None
        if separation is not None:
            if not separation.done():
                mark(
                    'mix',
                    'Waiting for vocal separation',
                    span('mix', 0),
                    segments=[piece.public() for piece in translated],
                )
            try:
                bed = separation.result()
            except CancelledError:
                raise
            except Exception as exc:
                control.check()
                logger.warning('Vocal separation failed for %s: %s', job_id, exc)
                background_mode = 'duck'
                background_notes.append(
                    f'Vocal separation failed ({str(exc)[:80]}); the original mix was ducked instead.'
                )
                store.save(job_id, background={'mode': 'duck', 'status': 'ready', 'progress': 1})
        if background_mode == 'duck':
            bed = extract_full_audio(video, directory)
        mark(
            'mix',
            'Placing lines on the timeline',
            span('mix', 0),
            segments=[piece.public() for piece in translated],
            background_mode=background_mode,
        )
        add_notes(*background_notes)
        dubbed_audio = assemble_timeline(segments, video_duration, directory, bed, background_mode)
        timing_warnings = placement_notes(segments)
        mark(
            'lipsync',
            'Synchronizing lips',
            span('lipsync', 0),
            timing_warnings=timing_warnings,
            segments=[piece.public() for piece in translated],
            audio_url=f'/api/jobs/{job_id}/audio',
        )
        write_review_cut(
            job_id, directory, video, dubbed_audio
        )  # the dub is audible now, over the original picture
        if faces_ahead is not None and not faces_ahead.done():
            mark('lipsync', 'Finishing face analysis', span('lipsync', 0))
        try:
            if faces_ahead is not None:
                faces_ahead.result()
        except CancelledError:
            raise
        except (
            Exception
        ) as exc:  # noqa: BLE001  (the render redoes whatever the early pass did not finish)
            logger.warning('Early face analysis for %s did not finish: %s', job_id, exc)
        with heavy_stage(control, job_id, 'Waiting for a render slot'):
            control.check()
            synced = _render(job_id, directory, control) or set()
        for piece in translated:
            piece.state = 'lip-synced' if round(piece.start, 1) in synced else 'voiced'
        mark(
            None,
            'Ready',
            100,
            status='completed',
            finished_at=utcnow(),
            usage=usage.as_dict(),
            review_url=None,
            timing_warnings=timing_warnings,
            segments=[piece.public() for piece in translated],
        )
        remove_intermediates(directory)
    except CancelledError:
        logger.info('Job %s cancelled', job_id)
        store.save(
            job_id,
            status='cancelled',
            stage='Cancelled',
            stage_key=None,
            retryable=False,
            retry_at=None,
        )
    except Exception as exc:
        logger.exception('Dubbing job %s failed', job_id)
        control.terminate_processes()
        store.save(
            job_id,
            status="failed",
            stage="Failed",
            stage_key=None,
            error=str(exc)[:500],
            retryable=isinstance(exc, BodhanRateLimitError)
            and not isinstance(exc, BodhanBudgetError),
        )
    finally:
        if client:
            client.close()
