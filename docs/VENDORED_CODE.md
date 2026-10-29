# Required bundled implementations

These folders are dependencies of the studio, rather than obsolete experiments. They are marked as vendored in GitHub's language/diff classification.

| Path | Why it remains | Integration |
| --- | --- | --- |
| `Wav2Lip/` | Modified local lip-sync fallback, face detection and rendering utilities | `backend/rendering/wav2lip.py`; checkpoint paths in `.env.example` |
| `backend/vendor/normalize_volume.py` | Preserved audio-normalization implementation used by the studio | `backend/media/audio_norm.py` |
| `fonts/` | Noto fonts for every script the studio dubs into (Devanagari, Bengali, Tamil, Arabic and the rest), for burned-in subtitles; SIL Open Font License in `fonts/OFL.txt` | `backend/media/shorts.py` |

Wav2Lip's README describes its upstream project; its external service examples are not Dubloom's setup instructions. Use the root [README](../README.md) for this studio.

The obsolete upstream dependency pins are omitted. Install studio dependencies from the single root `requirements.txt`; isolated worker dependencies are handled by `scripts/setup_models.sh`.

Other downloaded model code lives in the ignored `third_party/` directory. Isolated environments and model weights are created by `scripts/setup_models.sh`. GFPGAN runtime weights can also appear under the ignored `gfpgan/weights/` directory.

Preserve upstream notices and review the applicable code/model terms when redistributing. Do not remove runtime dependency files just because the main studio imports them indirectly through workers or subprocesses.

## Source versus local output

- Track application source, tests, lockfiles, configuration templates, required fonts and architecture documentation.
- Build `frontend/dist/` locally; CI publishes the build as an artifact.
- Keep credentials, rendered jobs, caches, logs, editor backups and local review notes outside Git.
