"""Runtime settings loaded from `.env` (see README for every option)."""

from dataclasses import dataclass
from pathlib import Path
import os
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _int(name: str, default: int, low: int | None = None, high: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_CPUS = os.cpu_count() or 4


def _gpu_available() -> bool:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_DEVICE = os.getenv("DEVICE", "auto").strip().lower() or "auto"
DEVICE = ("cuda" if _gpu_available() else "cpu") if _DEVICE == "auto" else _DEVICE


def _venv_python(name: str) -> str:
    path = ROOT / name / "bin" / "python"
    return str(path) if path.exists() else ""


@dataclass(frozen=True)
class Settings:
    # Server
    """Immutable environment-derived settings for the studio and model workers."""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int("PORT", 8001, 1, 65535)
    studio_password: str = os.getenv("STUDIO_PASSWORD", "").strip()
    # Bodhan speech models (console.bodhan.ai): Indic-Transcribe, Indic-Translate, Indic-Speak
    bodhan_asr_key: str = os.getenv("BODHAN_ASR_API_KEY", "")
    bodhan_translate_key: str = os.getenv("BODHAN_TRANSLATE_API_KEY", "")
    bodhan_tts_key: str = os.getenv("BODHAN_TTS_API_KEY", "")
    bodhan_base_url: str = os.getenv("BODHAN_BASE_URL", "https://api.bodhan.ai").rstrip("/")
    # Keep speech recognition, translation and voices on Bodhan: wait out its rate limits instead of
    # letting OpenRouter take those stages over (OpenRouter still reviews speakers and edits the script).
    bodhan_only: bool = _bool("BODHAN_ONLY", False)
    model_request_workers: int = _int("MODEL_REQUEST_WORKERS", 2, 1, 8)
    # native: target language in its own script; codemix: keeps common English words in English
    translation_script: str = os.getenv("TRANSLATION_SCRIPT", "native").strip().lower() or "native"
    # OpenRouter models, used for any stage Bodhan cannot serve (and for cloned voices)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model: str = (
        os.getenv("OPENROUTER_MODEL", "google/gemini-3.8-flash").strip()
        or "google/gemini-3.8-flash"
    )
    openrouter_fallback_models: tuple = tuple(
        m.strip()
        for m in os.getenv(
            "OPENROUTER_FALLBACK_MODELS", "google/gemini-3.7-flash,openai/gpt-4.1-mini"
        ).split(",")
        if m.strip()
    )
    openrouter_asr_model: str = os.getenv(
        "OPENROUTER_ASR_MODEL", "google/gemini-3.5-transcribe"
    ).strip()
    openrouter_asr_chat_model: str = os.getenv(
        "OPENROUTER_ASR_CHAT_MODEL", "google/gemini-3.8-flash"
    ).strip()
    openrouter_tts_model: str = os.getenv(
        "OPENROUTER_TTS_MODEL", "google/gemini-3.8-flash-tts"
    ).strip()
    openrouter_clone_model: str = os.getenv("OPENROUTER_CLONE_MODEL", "fish-audio/s2.1-pro").strip()
    # Shorten lines that would need a noticeable speed-up (needs OpenRouter)
    timing_fit: bool = _bool("TIMING_FIT", True)
    # Lip sync
    wav2lip_checkpoint: Path = ROOT / os.getenv(
        "WAV2LIP_CHECKPOINT", "Wav2Lip/checkpoints/wav2lip_gan.pth"
    )
    # The non-GAN Wav2Lip weights score better on lip-sync accuracy (LSE-D/LSE-C); the GAN is sharper.
    wav2lip_accurate_checkpoint: Path = ROOT / os.getenv(
        "WAV2LIP_ACCURATE_CHECKPOINT", "Wav2Lip/checkpoints/wav2lip.pth"
    )
    lip_sync_model: str = os.getenv("LIP_SYNC_MODEL", "wav2lip").strip().lower() or "wav2lip"
    wav2lip_python: str = os.getenv("WAV2LIP_PYTHON", "").strip() or sys.executable
    wav2lip_chunk_seconds: int = _int("WAV2LIP_CHUNK_SECONDS", 15, 3, 120)
    wav2lip_resize_factor: int = _int("WAV2LIP_RESIZE_FACTOR", 1, 1, 4)
    wav2lip_threads: int = _int("WAV2LIP_THREADS", min(8, _CPUS), 1, 64)
    wav2lip_face_interval: int = _int("WAV2LIP_FACE_INTERVAL", 2, 1, 5)
    lip_sync_runtime: str = os.getenv("LIP_SYNC_RUNTIME", "openvino").strip() or "openvino"
    openvino_device: str = os.getenv("OPENVINO_DEVICE", "CPU").strip() or "CPU"
    # Compute device for the model workers: cuda (H100 etc.) or cpu
    device: str = DEVICE
    # Diarization: diarizen (DiariZen WavLM-Large + VBx, best open accuracy) or sherpa (light ONNX fallback)
    diarization_backend: str = os.getenv("DIARIZATION_BACKEND", "").strip().lower() or (
        "diarizen" if _venv_python(".venv-diarizen") else "sherpa"
    )
    diarization_threads: int = _int("DIARIZATION_THREADS", min(8, _CPUS), 1, 64)
    diarizen_python: str = os.getenv("DIARIZEN_PYTHON", "").strip() or _venv_python(
        ".venv-diarizen"
    )
    diarizen_model: Path = ROOT / os.getenv("DIARIZEN_MODEL", "models/diarizen-v2")
    diarizen_embedding: Path = ROOT / os.getenv(
        "DIARIZEN_EMBEDDING", "models/pyannote-wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
    )
    # Diarize the Demucs dialogue stem (music removed) and let an audio LLM review who says each line
    diarize_dialogue_stem: bool = _bool("DIARIZE_DIALOGUE_STEM", True)
    speaker_review: bool = _bool("SPEAKER_REVIEW", True)
    # Check Bodhan transcripts and post-edit Bodhan translations in context (OpenRouter) before voicing
    transcript_check: bool = _bool(
        "TRANSCRIPT_CHECK", False
    )  # off: the source transcript is never rewritten by a model
    script_supervision: bool = _bool("SCRIPT_SUPERVISION", True)
    # Lip sync engine: latentsync (LatentSync 1.6, needs a CUDA GPU with >= 18 GB), wav2lip, or auto
    lip_sync_engine: str = os.getenv("LIP_SYNC_ENGINE", "auto").strip().lower() or "auto"
    hands_python: str = os.getenv("HANDS_PYTHON", "").strip() or _venv_python(
        ".venv-hands"
    )  # MediaPipe lives apart from LatentSync
    latentsync_python: str = os.getenv("LATENTSYNC_PYTHON", "").strip() or _venv_python(
        ".venv-latentsync"
    )
    latentsync_dir: Path = ROOT / os.getenv("LATENTSYNC_DIR", "third_party/LatentSync")
    latentsync_checkpoints: Path = ROOT / os.getenv(
        "LATENTSYNC_CHECKPOINTS", "models/latentsync-1.6"
    )
    latentsync_vae: Path = ROOT / os.getenv("LATENTSYNC_VAE", "models/sd-vae-ft-mse")
    latentsync_steps: int = _int("LATENTSYNC_STEPS", 20, 1, 100)
    latentsync_guidance: float = float(os.getenv("LATENTSYNC_GUIDANCE", "1.5"))
    latentsync_windows: int = _int(
        "LATENTSYNC_WINDOWS", 1, 1, 8
    )  # 16-frame windows denoised per UNet batch (same output; fills the GPU)
    # every synthesised clip is levelled with the user's TTS normaliser: speech RMS to this dBFS, then limited to this peak
    tts_target_rms: float = float(os.getenv("TTS_TARGET_RMS", "-20.0"))
    tts_peak_dbfs: float = float(os.getenv("TTS_PEAK_DBFS", "-1.0"))
    latentsync_enhance: bool = _bool(
        "LATENTSYNC_ENHANCE", True
    )  # restore the redrawn lower face after LatentSync
    lip_restore: str = (
        os.getenv("LIP_RESTORE", "gfpgan").strip().lower() or "gfpgan"
    )  # gfpgan (natural detail) or esrgan (sharpen only)
    restore_temporal: float = max(
        0.0, min(0.9, float(os.getenv("RESTORE_TEMPORAL", "0")))
    )  # 0 = per-frame restoration; 0.5 = smooth the restored detail over time
    restore_workers: int = _int(
        "RESTORE_WORKERS", 2, 1, 4
    )  # GFPGAN processes / ESRGAN threads restoring spans side by side
    restore_split_px: int = _int(
        "RESTORE_SPLIT_PX", 200, 0, 4096
    )  # GFPGAN redraws faces narrower than this; Real-ESRGAN sharpens the rest
    gfpgan_weights: Path = ROOT / os.getenv("GFPGAN_WEIGHTS", "models/gfpgan/GFPGANv1.4.pth")
    restore_python: str = os.getenv("RESTORE_PYTHON", "").strip() or _venv_python(".venv-restore")
    latentsync_deepcache: bool = _bool("LATENTSYNC_DEEPCACHE", True)
    insightface_root: Path = ROOT / os.getenv("INSIGHTFACE_ROOT", "models/insightface")
    # Light-ASD active speaker detection (AVA-trained); runs in the LatentSync environment
    asd_dir: Path = ROOT / os.getenv("ASD_DIR", "third_party/Light-ASD")
    asd_weights: str = os.getenv("ASD_WEIGHTS", "weight/pretrain_AVA_CVPR.model")
    face_det_size: int = _int("FACE_DET_SIZE", 640, 256, 1280)
    # Analyse every frame on a GPU; every second frame on CPU (repeated in between)
    face_stride: int = _int("FACE_STRIDE", 1 if DEVICE == "cuda" else 2, 1, 5)
    # Parallel face-analysis workers on parts of the video (CPU: about one per 6 cores; GPU: one)
    face_workers: int = _int(
        "FACE_WORKERS", 1 if DEVICE == "cuda" else max(1, min(3, _CPUS // 6)), 1, 8
    )
    # Dubbed lines are fitted exactly into the original speech slot; allowed tempo change
    fit_min_speed: float = float(os.getenv("FIT_MIN_SPEED", "0.9"))
    fit_max_speed: float = float(os.getenv("FIT_MAX_SPEED", "1.1"))
    fit_rounds: int = _int("FIT_ROUNDS", 4, 0, 6)
    # Scheduling: model stages of several jobs may overlap; CPU-heavy stages are bounded.
    job_workers: int = _int("JOB_WORKERS", 2, 1, 16)
    render_workers: int = _int("RENDER_WORKERS", 1, 1, 8)
    # Media limits
    jobs_dir: Path = ROOT / os.getenv("JOBS_DIR", "jobs")
    max_video_minutes: int = _int("MAX_VIDEO_MINUTES", 30, 1, 600)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 2048, 1, 65536)
    max_video_height: int = _int("MAX_VIDEO_HEIGHT", 720, 240, 2160)
    keep_background_default: bool = _bool("KEEP_BACKGROUND", True)
    # separate: Demucs removes the original voices from the bed (no echo); duck: lower the original mix under speech
    background_mode: str = os.getenv("BACKGROUND_MODE", "separate").strip().lower() or "separate"
    separation_model: str = os.getenv("SEPARATION_MODEL", "htdemucs").strip() or "htdemucs"
    # roformer: BS/Mel-RoFormer vocal models via audio-separator in their own venv (cleaner bed, fewer voice ghosts)
    separation_backend: str = (
        os.getenv("SEPARATION_BACKEND", "demucs").strip().lower() or "demucs"
    )  # demucs | roformer
    roformer_model: str = os.getenv("ROFORMER_MODEL", "vocals_mel_band_roformer.ckpt").strip()
    separate_python: Path = ROOT / os.getenv("SEPARATE_PYTHON", ".venv-separate/bin/python")
    separator_models: Path = ROOT / os.getenv("SEPARATOR_MODELS", "models/separator")
    # exact: every line is stretched to fill its original slot precisely (lips move for the whole span);
    # natural: pace stays within 0.8x-1.35x and a line may end early or lose its tail instead.
    # Hugging Face repo (or local dir) of the audio gender classifier; blank disables it (pitch + reviewer only)
    gender_model: str = os.getenv(
        "GENDER_MODEL", "audeering/wav2vec2-large-robust-6-ft-age-gender"
    ).strip()
    # exact mode: pace band that still sounds like the same speaker; beyond it a line borrows the following silence or ends early
    pace_min: float = float(os.getenv("PACE_MIN", "0.85"))
    pace_max: float = float(os.getenv("PACE_MAX", "1.15"))
    timing_mode: str = (
        "natural" if os.getenv("TIMING_MODE", "exact").strip().lower() == "natural" else "exact"
    )
    separation_jobs: int = _int("SEPARATION_JOBS", 2, 1, 8)
    separation_threads: int = _int("SEPARATION_THREADS", min(8, _CPUS), 1, 64)
    separation_max_minutes: int = _int("SEPARATION_MAX_MINUTES", 20, 1, 600)
    background_level: float = float(
        os.getenv("BACKGROUND_LEVEL", "0.9")
    )  # separated bed, outside speech
    background_speech_level: float = float(
        os.getenv("BACKGROUND_SPEECH_LEVEL", "0.6")
    )  # separated bed, under speech
    duck_level: float = float(os.getenv("DUCK_LEVEL", "0.6"))  # original mix, outside speech
    duck_speech_level: float = float(
        os.getenv("DUCK_SPEECH_LEVEL", "0.04")
    )  # original mix, under speech
    # Mouth-region super-resolution after Wav2Lip (Real-ESRGAN compact)
    mouth_enhance: bool = _bool("MOUTH_ENHANCE", True)
    esrgan_weights: Path = ROOT / os.getenv(
        "ESRGAN_WEIGHTS", "Wav2Lip/checkpoints/realesr-general-x4v3.pth"
    )


settings = Settings()
