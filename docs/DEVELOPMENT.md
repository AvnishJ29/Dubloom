# Development conventions

## Where changes belong

Keep routes in `backend/api/` (one router module per concern, request bodies in `schemas.py`, shared validation in `helpers.py`), orchestration in `backend/jobs/pipeline.py`, and stage implementations in their feature packages. Shared records belong in `backend/models.py`; environment configuration belongs in `backend/config.py` and `.env.example`.

Vue components own UI state and interactions. Reusable client behavior belongs in `frontend/src/lib/`. Styles live in `frontend/src/styles/`; `frontend/src/style.css` imports them in cascade order. Preserve that order when editing styles. Generated files in `frontend/dist/` come from `npm run build` and are excluded from Git.

Do not move model imports between processes: workers use separate environments to avoid incompatible dependencies. Keep upstream model code, vendor implementations and downloaded weights separate from studio changes.

Use absolute `backend.<package>.<module>` imports across feature boundaries. Package `__init__.py` files describe responsibilities and do not eagerly load models or re-export unrelated modules. Resolve repository assets and worker scripts from `backend.config.ROOT`, rather than counting parent folders from a moved module.

Voice metadata belongs in `backend/voices/data/`. If upgrading a checkout that explicitly sets `BODHAN_VOICE_CATALOG_PATH=backend/voice_catalog.json`, update that setting to `backend/voices/data/voice_catalog.json`. Custom catalogue paths remain supported. The fallback diarization setup command is now `python -m backend.speakers.setup_local_models`.

## Formatting

Python uses Black 25.1.0, a 100-column target and existing string quotes. Install the formatter in a development environment:

```bash
python -m pip install -r requirements.txt
python -m black backend main.py
python -m black --check backend main.py
```

The configuration in `pyproject.toml` excludes vendor code. Black is pinned in the root `requirements.txt`, together with studio dependencies. CI installs only that formatter entry for its syntax/formatting job.

Frontend formatting uses the pinned Prettier development dependency:

```bash
cd frontend
npm ci
npm run format
npm run format:check
```

The formatter expands JavaScript, Vue script blocks and CSS. Vue template markup and component styles are preserved byte for byte, avoiding incidental layout changes.

## Validation

Run the commands in the [README](../README.md#development-and-checks). Formatting-only changes should preserve parsed executable code, template content, CSS rule order, imports, configuration defaults, prompt text and public API shapes.

Build the frontend after editing source, but do not commit `frontend/dist/`. Keep credentials, jobs, logs, caches, editor backups and weights out of Git. Existing CI validates frontend tests/build and backend Python syntax.

## Comments and docstrings

Document public interfaces and important side effects, especially job persistence, model inputs, timing units and provider fallback. Comments should explain constraints and decisions; avoid repeating what a line of code already says. Keep model prompts separate from explanatory prose.

## Before publishing

1. Run backend tests on the configured environment with FFmpeg available.
2. Check Python/frontend formatting and Python syntax, then run frontend tests and build.
3. Review the diff for unintended behavior changes; update README/configuration examples when behavior changes.
4. Confirm the tracked files exclude credentials, runtime jobs, logs and model weights.
5. Push and review GitHub checks. The frontend CI job uploads its production build as an artifact. Verify the existing app URL when deployment is part of the change.

Model output quality also needs representative media checks; passing unit tests alone does not establish end-to-end model accuracy.

## Repository boundaries

- Studio Python modules stay under `backend/`; retain stage modules instead of adding unrelated helpers to the API or orchestrator.
- Frontend code, source assets and tests stay in `frontend/src/`, `frontend/public/` and `frontend/tests/`.
- Keep architecture guides in `docs/`; edit the Mermaid diagrams in `PIPELINE.md` instead of committing generated image exports.
- Deployment helpers belong in `scripts/`; the systemd template is `scripts/systemd/dubloom.service`.
- Keep local review notes, environment snapshots and editor backups out of source control.
- Tests, fonts, lockfiles and vendored runtime dependencies are intentional. See [vendored-code guide](VENDORED_CODE.md).
