# Dubloom

A video-dubbing studio for 23 supported languages. Import a link or video, translate the dialogue, match voices to speakers, adjust the script, and export the dub with optional lip sync, audio and subtitles.

Built with **FastAPI + Vue 3**. Supports source video up to 4K when configured; lip sync applies only to visible speaking faces. Audio/text are processed by the configured AI providers.

## Key features

- Video links or file uploads, with an optional source clip range.
- Speaker detection, matched studio voices, and supported voice cloning.
- Editable transcripts and translations, AI rewrites, delivery controls and line takes.
- Duration fitting, preserved background audio, and optional lip sync for visible speakers.
- Video/audio/subtitle exports, short clips, resumable jobs and render-cache reuse.
- Owner-change history, plus configurable studio sign-in and processing limits.

The pipeline is **source → audio/speakers → transcribe → translate → voice → fit/mix → lip sync → export**.

Speech services use Bodhan **Indic-Transcribe, Indic-Translate and Indic-Speak**. OpenRouter supports script/review work, cloning and configured fallbacks. Local workers run speaker analysis, audio separation, face analysis and LatentSync/Wav2Lip. See the [pipeline guide](docs/PIPELINE.md) for the model and code behind each phase.

## Requirements

- Linux/Bash, Git, curl and Python 3.12 with pip/venv.
- FFmpeg 7 with Rubber Band and subtitle/libass support, plus ffprobe on `PATH`; Node.js 22+ and npm.
- NVIDIA CUDA GPU for LatentSync; voice-only and Wav2Lip provide CPU options.
- Bodhan ASR, translation and TTS keys; an OpenRouter key for script editing, review and configured model fallbacks.
- Internet access and disk space for model weights. Setup creates isolated Python 3.10/3.12 model environments using `uv`.

## Install and run

Authenticate to GitHub first: this repository is private.

```bash
git clone https://github.com/AvnishJ29/Dubloom.git
cd Dubloom
python3.12 -m venv .venv             # explicitly select the studio's Python version
cp -n .env.example .env              # fill in the keys listed below
scripts/setup_models.sh cuda         # use "cpu" for a CPU setup
./run.sh
```

Open **http://localhost:8001**. `run.sh` installs the studio dependencies when needed, builds a missing frontend, prepares fallback diarization models and starts the API/UI. For a later UI rebuild, run `REBUILD_UI=1 ./run.sh`.

## Configuration

All settings are described in [`.env.example`](.env.example). Keep actual credentials in the untracked `.env`.

| Setting | Purpose |
| --- | --- |
| `BODHAN_ASR_API_KEY`, `BODHAN_TRANSLATE_API_KEY`, `BODHAN_TTS_API_KEY` | Speech recognition, translation and voices |
| `OPENROUTER_API_KEY` | Script work, speaker review and available fallbacks |
| `BODHAN_ONLY` | Set `true` to keep ASR, translation and studio voices on Bodhan; reviewer/script work can still use OpenRouter |
| `STUDIO_PASSWORD` | Optional studio sign-in |
| `HOST`, `PORT` | Bind address and server port; defaults `0.0.0.0:8001` |
| `MAX_VIDEO_HEIGHT` | Source resolution limit; example config is 720, set 2160 for up to 4K |
| `LIP_SYNC_MODEL` | Blank/default selection, `latentsync`, `wav2lip`, `wav2lip_gan` or `none` |
| `YTDLP_NODE` | Node 22+ executable when the Node on `PATH` is older |
| `JOBS_DIR` | Persistent project/media storage; defaults to `jobs/` |

Timing, background audio, concurrency, model paths and upload limits are also configurable in that file.

## Use the studio

1. Paste a video link or upload a file. Choose the language, lip-sync/voice-only mode and optional clip range.
2. Review the result. Edit script lines, voices, delivery or takes, then re-render.
3. Download video, audio or subtitles, or create a short. History records owner changes; studio maintenance renders update the associated entry.

The pipeline reuses unchanged speech, model analysis and rendered spans. Processing time depends on video length, faces, hardware and provider limits.

## Code structure

| Path | Responsibility |
| --- | --- |
| `main.py`, `run.sh` | Application entry point and launcher |
| `backend/api.py`, `backend/config.py`, `backend/models.py` | HTTP routes, environment settings and shared records |
| `backend/jobs/` | Pipeline orchestration, persistence, render history, takes and GPU slots |
| `backend/media/` | Media input, audio processing, timing, validation, subtitles and shorts |
| `backend/speech/` | ASR, translation, script supervision, TTS and duration fitting |
| `backend/speakers/` | Diarization, speaker profiles, face evidence and gender analysis |
| `backend/voices/` | Voice catalogue, library data, casting and synthesis routing |
| `backend/providers/` | Bodhan/OpenRouter clients, pacing and usage accounting |
| `backend/rendering/` | Lip-sync engines, face restoration and final assembly |
| `backend/workers/` | Model processes running in isolated environments |
| `backend/tests/` | Backend regression tests |
| `frontend/src/components/` | Vue screens and reusable controls |
| `frontend/src/lib/` | API client, formatting, timing and history helpers |
| `frontend/src/styles/`, `frontend/src/style.css` | Styles by concern and their ordered entry point |
| `frontend/tests/`, `frontend/scripts/` | Client tests and formatting tools |
| `scripts/`, `scripts/systemd/` | Model setup, deployment and optional systemd service |
| `docs/` | Setup/video guides and editable Mermaid pipeline diagrams |
| `fonts/` | Fonts required by burned-in subtitles |
| `Wav2Lip/`, `backend/vendor/` | Vendored implementations; see [vendored-code guide](docs/VENDORED_CODE.md) |
| `models/`, `third_party/`, `.venv*/`, `gfpgan/weights/` | Local weights, checkouts and environments; not committed |
| `jobs/`, `logs/`, `scratch/` | Runtime data; not committed |

Generated `frontend/dist/` assets are built locally and are not committed. `run.sh` builds them when missing. Keep source assets in `frontend/public/` and application code in `frontend/src/`.

[`requirements.txt`](requirements.txt) is the single committed Python dependency manifest for the studio, its alternative SpeechBrain path and formatting tool. `run.sh` installs it after selecting CPU/CUDA Torch wheels. The model setup script keeps incompatible worker dependencies in their own environments; install this manifest into the studio environment only.

## Development and checks

Keep the API running on port 8001, then use `npm run dev` inside `frontend/`. Vite serves the development UI on port 5173 and proxies `/api` to the backend.

```bash
# From the repository root, with the studio environment installed:
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m black --check backend main.py
.venv/bin/python -m unittest discover -s backend/tests
.venv/bin/python -m compileall -q backend main.py

# Frontend dependencies, checks and production build:
cd frontend
npm ci
npm test
npm run format:check
npm run build
```

Backend tests use FFmpeg/ffprobe; model/provider interactions are generally mocked. The full setup is needed for real end-to-end dubbing. API documentation is available at **http://localhost:8001/docs**. CI checks formatting, client tests, Python syntax and the frontend build.

See the [video/code walkthrough guide](docs/CODE_WALKTHROUGH.md), [development conventions](docs/DEVELOPMENT.md), [full pipeline](docs/PIPELINE.md) and [cluster deployment script](scripts/deploy_to_cluster.sh).

## Evaluation submission guide

| Shared evaluation requirement | Where to review |
| --- | --- |
| Clean, readable and organized code | Feature packages listed above; formatting rules in [development conventions](docs/DEVELOPMENT.md) |
| Appropriate comments and docstrings | Module and function interfaces, worker inputs, timing rules and persistence code |
| Project overview, features, setup and run instructions | This README, with configuration details in [`.env.example`](.env.example) |
| Folder structure, code organization and key components | Code structure above and the [phase/model walkthrough](docs/PIPELINE.md) |

Repository: [AvnishJ29/Dubloom](https://github.com/AvnishJ29/Dubloom). It is private as requested; evaluators must be granted repository access.

## License

Project code: [MIT](LICENSE). Downloaded models retain their own licenses, including DiariZen CC BY-NC 4.0, InsightFace research/non-commercial terms, LatentSync OpenRAIL++, Light-ASD MIT and GFPGAN Apache 2.0.
