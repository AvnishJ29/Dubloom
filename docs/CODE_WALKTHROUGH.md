# Dubloom: setup and code-walkthrough video

A recording guide for a separate **10–12 minute** technical video. Show the repository and terminal, then follow one job from the Vue form through the API, pipeline and export. Use an existing completed result for playback so the recording does not wait for GPU rendering.

## Recording plan

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–0:40 | Studio home page and one completed dub | Problem, core features and result |
| 0:40–2:10 | README, setup commands, `.env.example` | Dependencies, configuration and launch |
| 2:10–3:00 | Repository tree | Responsibilities and local/generated folders |
| 3:00–4:15 | `main.py`, Vue entry point, form and API | One creation request and job scheduling |
| 4:15–7:45 | Pipeline and stage modules | Audio, speakers, ASR, translation, voices, timing, mix, lip sync and export |
| 7:45–9:20 | Job record, progress client, cache and editing code | State, concurrency, resume and incremental re-renders |
| 9:20–10:30 | Tests, workflow and one real result | Validation, limitations and closing demo |

These timings are a guide. Explain a few representative functions rather than reading every file.

## 1. Opening narration

> “Dubloom is a video-dubbing studio for 23 supported languages. A user can upload a video or paste a link, choose a target language, and create a voice-only or lip-synced dub. The user can then edit individual script lines, change voices and delivery, and download video, audio or subtitles.
>
> “The frontend is Vue 3, and the backend is FastAPI. The implementation combines remote speech services with local models for speaker analysis, audio separation, face analysis and lip sync. I’ll show how to run it and then trace a single job through the code.”

Show a brief original/dubbed comparison. Choose a clip with a clearly visible speaking face when demonstrating lip sync. Narration-only video demonstrates dubbing, but may correctly produce zero lip-synced frames.

## 2. Setup and launch

Show [README.md](../README.md), [run.sh](../run.sh), [setup_models.sh](../scripts/setup_models.sh), and [`.env.example`](../.env.example). Keep actual keys, `.env` and credential-manager windows off the recording.

Prerequisites: Linux/Bash, Python 3.12 with pip/venv, FFmpeg 7 and ffprobe, Node.js 22+, Git and curl. LatentSync needs a CUDA GPU. CPU options include voice-only and Wav2Lip.

```bash
git clone https://github.com/AvnishJ29/Dubloom.git
cd Dubloom
python3.12 -m venv .venv
node --version
ffmpeg -version
cp -n .env.example .env
# Fill in provider keys privately, outside the recording.
scripts/setup_models.sh cuda
./run.sh
```

Use `scripts/setup_models.sh cpu` for CPU setup. The private repository requires GitHub authentication.

> “The setup script creates separate model environments and downloads the required code and weights. This isolation matters because the model packages use different dependency versions. The launcher installs the main studio dependencies when needed, builds a missing frontend, prepares fallback diarization models and starts the app.”

Open `http://localhost:8001` and `http://localhost:8001/docs`. Show `GET /api/health`. The hosted studio uses its existing tunnel URL; localhost is the default for a new local installation.

Installation and rendering may take longer than the video. Cut the waiting time and label it as skipped setup/rendering time.

For development:

```bash
# Terminal 1, repository root
./run.sh

# Terminal 2
cd frontend
npm ci
npm run dev
```

The development UI runs on port 5173 and proxies API requests to port 8001. Rebuild the served UI with `npm run build` in `frontend/`, or `REBUILD_UI=1 ./run.sh`.

## 3. Folder structure

Open the project tree and explain these boundaries:

```text
Dubloom/
├── main.py                    # Uvicorn entry point
├── run.sh                     # Studio launcher
├── backend/
│   ├── api.py                 # Routes and request validation
│   ├── config.py              # Environment settings
│   ├── models.py              # Shared Segment records
│   ├── jobs/                  # Pipeline, persistence, versions and takes
│   ├── media/                 # Input, audio, timing and deliverables
│   ├── speech/                # ASR, translation, script editing and TTS
│   ├── speakers/              # Diarization and speaker/face evidence
│   ├── voices/                # Catalogue, casting and synthesis routing
│   ├── providers/             # API clients, pacing and usage
│   ├── rendering/             # Lip sync, restoration and assembly
│   ├── workers/               # Isolated model processes
│   ├── tests/                 # Backend regression tests
│   └── vendor/                # Required normalization implementation
├── frontend/
│   ├── src/App.vue            # Studio shell and selected job
│   ├── src/components/        # Creation, playback and editing UI
│   ├── src/lib/               # API, timing and history helpers
│   ├── src/styles/            # Ordered style sections via src/style.css
│   ├── tests/                 # Client regression tests
│   └── dist/                  # Generated locally; excluded from Git
├── scripts/                   # Setup, deployment and systemd template
├── docs/                      # Guides and Mermaid pipeline diagrams
├── Wav2Lip/                   # Bundled lip-sync fallback
└── .env.example               # Configuration template
```

> “The backend separates HTTP handling from orchestration and the individual processing stages. The frontend separates screen components from shared client logic. Model environments, downloaded weights, jobs, logs and scratch files are local state and are excluded from Git.”

Also identify `backend/vendor/` as copied third-party code. The application stores projects in `jobs/<id>/`; it does not use a relational database.

## 4. Follow one request through the application

Open these files in order:

1. [frontend/src/main.js](../frontend/src/main.js): `createApp(App).mount('#app')`.
2. [frontend/src/App.vue](../frontend/src/App.vue): configuration, project list, selected job and subscriptions.
3. [CreatePanel.vue](../frontend/src/components/CreatePanel.vue): `submit()` builds the creation options.
4. [lib/api.js](../frontend/src/lib/api.js): `post()` sends JSON; `upload()` sends multipart data with progress.
5. [backend/api.py](../backend/api.py): `create_job()` or `create_upload_job()` validates inputs and calls `start_job()`.
6. [backend/jobs/pipeline.py](../backend/jobs/pipeline.py): `run_job()` manages attempts; `process_job()` runs the stages.

> “A URL becomes a POST to /api/jobs; an upload goes to /api/jobs/upload. The API validates the options, saves a queued job and submits it to an executor. It returns a job ID instead of holding the HTTP request open for the entire render. The browser uses that ID to display progress and results.”

A conceptual job line looks like this:

```json
{
  "start": 2.0,
  "end": 5.0,
  "speaker": "SPEAKER_00",
  "text": "Original speech",
  "translation": "Translated speech",
  "voice": "Kavya",
  "dub_start": 2.0,
  "dub_end": 5.0
}
```

Explain `Segment` in [models.py](../backend/models.py): original timing and words, translated script, voice settings, rendered timing and provenance travel together.

## 5. Explain the pipeline

The order below is conceptual. Separation and face analysis can overlap other stages, and streaming mode can translate a line immediately after transcription. With `BODHAN_ONLY=true`, automatic speech/translation fallback is disabled; OpenRouter can still perform script and reviewer work.

| Stage | Code to open | Input → result |
| --- | --- | --- |
| Source | `video_input.py`; `pipeline.acquire_source()` | URL/upload/saved source → prepared video, metadata and poster |
| Audio | `video_input.extract_audio()`, `separation.py` | Soundtrack → speech WAV and optional vocals/background stems |
| Speakers | `diarization.py`, `speaker_review.py`, `gender.py` | Speech → timed speaker turns, profiles and casting evidence |
| ASR | `asr.transcribe()`, `bodhan_api.py` | Each speech slot → source transcript |
| Translation/script | `translation.py`, `script_supervisor.py` | Transcript and context → editable target-language script |
| Voice | `voices.VoiceState`, `tts.synthesize()` | Script, speaker and delivery → voiced line WAVs |
| Duration fitting | `fitting.py`, `timeline.plan_timeline()` | Natural speech durations → fitted text and placement decisions |
| Mix | `audio_merge.assemble_timeline()` | Placed speech and background → clean voice track and delivered mix |
| Face/lip sync | `speaker_faces.py`, `lip_sync.py`, `workers/` | Visible speaking faces and dubbed speech → rendered mouth spans |
| Finish/export | `finish.py`, `media_validation.py`, `subtitles.py` | Rendered spans, picture and mix → validated video and exportable subtitles |

Suggested narration:

> “First, the input layer downloads or accepts the source, applies any requested clip range and prepares the media. Speech-stage audio is mono, 16 kHz, 16-bit PCM WAV.
>
> “When configured, separation isolates dialogue from music. Diarization answers who spoke when. Speaker review, pitch, gender and face evidence help refine speaker identities and voice selection.
>
> “For ASR, the pipeline cuts audio at the speaker slots and calls Bodhan Indic-Transcribe. It sends a source-language hint when available. The provider client handles rate limits, and the ASR layer retries repetitive-token output. The pipeline can use the independent reviewer transcript for unusable output or script mismatches and records where the text came from.
>
> “Translation produces the target script. The script supervisor uses scene context and the expected speaking time to improve wording. Original, machine-translated and edited information can be retained separately.
>
> “The voice layer generates speech per line. Voice selection is settled before concurrent requests, so request completion order does not change the casting.
>
> “Timing is a separate problem from translation. The fitter compares measured voiced duration with the source slot and can rewrite and revoice automatic lines. User-edited wording is protected. The timeline then chooses a bounded playback speed, may use the following pause, and reports early endings or cut audio.
>
> “The mixer places the lines at their source starts, adds the chosen background bed and applies audio leveling. It writes a clean voice track for lip sync and a mixed track for the delivered video.
>
> “Face detection and active-speaker analysis determine which visible mouth should be animated. Hands and unsuitable face views can exclude frames. LatentSync renders eligible spans; restoration and the finishing stage assemble the result. Voice-only mode skips mouth changes.
>
> “Finally, export validation checks codecs, timeline alignment and decoded audio audibility. The studio exposes the full-quality download, browser preview where needed, audio and subtitles.”

### Explain timing with one example

A line occupies 4 seconds in the source but its natural dub lasts 5.2 seconds. The needed speed is `5.2 / 4 = 1.3x`. Instead of always rushing it, the code first tries a meaning-preserving rewrite and can use a following pause within its limits.

Show these distinctions:

- `fitting.py`: change automatic text to improve its natural speaking duration.
- `timeline.py`: decide where the resulting audio plays and at what tempo.
- `audio_merge.py`: apply that placement to actual audio.
- `dubbed_slots()`: cover both original speech and placed dubbed speech, so an early-ending dub closes the mouth and a borrowed pause stays animated.

Do not describe exact mode as guaranteeing an exact end time for every line: pace limits can make a line end early or require trimming.

## 6. Explain state, concurrency and edits

Open `JobStore`, `JobControl`, `run_segment_stage()` and `heavy_stage()` in [pipeline.py](../backend/jobs/pipeline.py).

> “Job state is stored in status.json. Updates use a temporary file and atomic replacement, with a process-local lock for mutations. Cancellation is cooperative and also stops registered worker processes.
>
> “Independent model requests can run concurrently, while heavy stages are bounded to protect compute resources. Isolated workers keep incompatible model dependencies out of the studio environment.
>
> “Stages reuse saved results when their inputs still match. Unchanged line audio and lip-sync spans can be reused, so an edit need not repeat the entire pipeline.”

For progress, open `job_events()` in [api.py](../backend/api.py) and `subscribe()` in [frontend/src/lib/api.js](../frontend/src/lib/api.js).

> “The UI receives server-sent events and also polls as a fallback for buffering proxies. It closes subscriptions when a job finishes or the user switches projects.”

For edits, show:

- `POST /api/jobs/{id}/script`: apply script changes.
- `POST /api/jobs/{id}/revoice`: apply voice, delivery or render settings.
- `POST /api/jobs/{id}/rewrite`: return AI proposals for user review.
- [versions.py](../backend/jobs/versions.py): archive output and remember line wording.
- [frontend/src/lib/history.js](../frontend/src/lib/history.js): show owner edits; maintenance refreshes update their associated entry.

Existing maintenance-baseline archives can be hidden from the UI while remaining on disk. This is a special case; do not say that old files were deleted.

## 7. Validation and closing demo

Show the commands and test summaries, not every test implementation:

```bash
.venv/bin/python -m unittest discover -s backend/tests
.venv/bin/python -m compileall -q backend main.py

# In the studio environment with root requirements.txt installed:
python -m black --check backend main.py

cd frontend
npm ci
npm run format:check
npm test
npm run build
```

At preparation time, **157 backend tests and 13 frontend tests passed**. These counts can change with later commits. GitHub CI checks formatting, frontend tests/build and Python syntax; the full backend suite was run in the configured cluster environment.

> “The tests cover orchestration, timing, caching, history, API handling and media processing. They do not prove that every learned-model result will be correct. For example, shorter ASR chunks can produce unusable output, and a video with no eligible visible speaking face will have no lip-synced frames. The application reports those cases rather than treating every finished render as perfect lip sync.”

Close by showing one completed job, its script and speakers, a timing indicator, playback and an export action. Optionally show a small edit and its queued progress; use a previously completed edit to demonstrate the finished result if rendering takes too long.

## Recording checklist

- Keep the README, stage modules and a completed example open in advance.
- Display `.env.example`, never real provider keys or `.env`.
- Explain your orchestration, fitting, caching, UI and integration work; identify external models and bundled upstream code.
- Use the actual CPU/GPU mode and configured resolution. “Up to 4K” is configuration-dependent.
- Label skipped installation/rendering time and prerecorded results honestly.
- Demonstrate a visible speaking face for the lip-sync example.
- Finish with the repository link, setup instructions and known limits.
