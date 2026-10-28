#!/usr/bin/env bash
# Start the Dubloom studio (API + built UI) from the project root.
set -euo pipefail
cd "$(dirname "$0")"
# (scripts/setup_models.sh may already have created .venv with only its own tools)
if ! { [ -x .venv/bin/python ] && .venv/bin/python -c "import fastapi, torch, demucs, sherpa_onnx" 2>/dev/null; }; then
  echo "Installing the studio environment…"
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip wheel
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TORCH_INDEX=https://download.pytorch.org/whl/cu121
  else
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
  fi
  mapfile -t TORCH_PACKAGES < <(grep -E '^(torch|torchaudio)==' requirements.txt)
  .venv/bin/pip install --quiet "${TORCH_PACKAGES[@]}" --index-url "$TORCH_INDEX"
  .venv/bin/pip install --quiet -r requirements.txt
fi
if [ ! -f frontend/dist/index.html ] || [ "${REBUILD_UI:-0}" = "1" ]; then
  echo "Building the frontend…"
  (cd frontend && npm install --no-audit --no-fund --loglevel=error && npm run build)
fi
if [ ! -f jobs/models/sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  echo "Fetching diarization models…"
  .venv/bin/python -m backend.speakers.setup_local_models
fi
exec .venv/bin/python main.py
