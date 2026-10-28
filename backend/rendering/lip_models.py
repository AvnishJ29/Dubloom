"""Selectable lip-sync models."""

from pathlib import Path

from backend.config import settings

LIP_SYNC_MODELS = {
    "latentsync": {
        "label": "LatentSync 1.6 (best quality, GPU)",
        "note": "Diffusion lip sync at 512 px: the most accurate and natural mouths. Needs a CUDA GPU.",
    },
    "wav2lip": {
        "label": "Wav2Lip (fast, CPU)",
        "note": "Original Wav2Lip weights: runs anywhere, softer mouth detail.",
    },
    "wav2lip_gan": {
        "label": "Wav2Lip GAN (fast, CPU)",
        "note": "GAN-refined Wav2Lip: crisper teeth and lips than plain Wav2Lip.",
    },
    "none": {
        "label": "Voice only (fastest, no lip sync)",
        "note": "The dubbed voice over the original picture; mouths are left as filmed. No GPU, minutes instead of tens of minutes.",
    },
}


def default_model() -> str:
    """The model for a job that names none: LatentSync on a CUDA machine that has it (unless LIP_SYNC_MODEL
    asks for Wav2Lip GAN or voice only), otherwise the configured model, or Wav2Lip."""
    from backend.rendering.latentsync import latentsync_available

    configured = (settings.lip_sync_model or "").lower()
    if (
        latentsync_available()
        and settings.device == "cuda"
        and configured in {"", "auto", "latentsync", "wav2lip"}
    ):
        return "latentsync"
    return configured if configured in LIP_SYNC_MODELS and configured != "latentsync" else "wav2lip"


def checkpoint_for(model: str | None) -> tuple[str, Path]:
    """Return (model id, Wav2Lip checkpoint used if the Wav2Lip engine renders)."""
    wanted = (model or default_model()).lower()
    if wanted in {"latentsync", "none"}:
        return wanted, settings.wav2lip_accurate_checkpoint
    candidates = {
        "wav2lip": settings.wav2lip_accurate_checkpoint,
        "wav2lip_gan": settings.wav2lip_checkpoint,
    }
    if wanted not in candidates:
        wanted = "wav2lip"
    if candidates[wanted].is_file():
        return wanted, candidates[wanted]
    for name, path in candidates.items():
        if path.is_file():
            return name, path
    return wanted, candidates[wanted]


def available_models() -> list[dict]:
    """Every model in LIP_SYNC_MODELS with whether this machine can run it, for the studio's picker."""
    from backend.rendering.latentsync import latentsync_available

    files = {
        "wav2lip": settings.wav2lip_accurate_checkpoint,
        "wav2lip_gan": settings.wav2lip_checkpoint,
    }
    out = []
    for name, meta in LIP_SYNC_MODELS.items():
        available = (
            True
            if name == "none"
            else (
                (latentsync_available() and settings.device == "cuda")
                if name == "latentsync"
                else files[name].is_file()
            )
        )
        out.append({"id": name, "available": available, **meta})
    return out
