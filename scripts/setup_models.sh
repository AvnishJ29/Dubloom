#!/usr/bin/env bash
# Build the isolated model environments and download every checkpoint the studio needs.
#
#   scripts/setup_models.sh            # auto: CUDA when nvidia-smi sees a GPU, otherwise CPU
#   scripts/setup_models.sh cuda       # force the CUDA 12.1 builds (H100, A100, L40S ...)
#   scripts/setup_models.sh cpu        # force CPU builds (development / no GPU)
#
# Environments (each model keeps the exact versions its authors pinned):
#   .venv            the studio (FastAPI, pipeline, Wav2Lip fallback)
#   .venv-diarizen   DiariZen speaker diarization (torch 2.1.1, Python 3.10)
#   .venv-latentsync LatentSync 1.6 lip sync + InsightFace face analysis + Light-ASD (torch 2.5.1, Python 3.10)
#   .venv-hands      MediaPipe Hands for mouth occlusion (kept apart: its protobuf pin breaks onnx)
#   .venv-restore    GFPGAN face restoration (basicsr/facexlib pins kept away from LatentSync)
#   .venv-separate   audio-separator for the Mel-RoFormer vocal separation (weights download on first use)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
MODE=${1:-auto}
if [ "$MODE" = auto ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then MODE=cuda; else MODE=cpu; fi
fi
echo "== setup mode: $MODE"
[ -x .venv/bin/python ] || { python3 -m venv .venv; .venv/bin/pip install -q --upgrade pip wheel; }
.venv/bin/pip install -q uv huggingface_hub
UV=.venv/bin/uv
$UV python install 3.10 >/dev/null

if [ "$MODE" = cuda ]; then
  TORCH_DIAR="--index-url https://download.pytorch.org/whl/cu121"; TORCH_LS="--index-url https://download.pytorch.org/whl/cu121"; ORT=onnxruntime-gpu==1.21.0
else
  TORCH_DIAR="--index-url https://download.pytorch.org/whl/cpu"; TORCH_LS="--index-url https://download.pytorch.org/whl/cpu"; ORT=onnxruntime==1.21.0
fi

# ---- source code, pinned to the commits this studio was built against ----
mkdir -p third_party
[ -d third_party/DiariZen ] || git clone -q https://github.com/BUTSpeechFIT/DiariZen.git third_party/DiariZen
git -C third_party/DiariZen checkout -q 844f555 2>/dev/null || true
[ -d third_party/LatentSync ] || git clone -q https://github.com/bytedance/LatentSync.git third_party/LatentSync
git -C third_party/LatentSync checkout -q a229c39 2>/dev/null || true
# Light-ASD (MIT): active speaker detection trained on AVA movies; its 4 MB weights ship in the repository
[ -d third_party/Light-ASD ] || git clone -q https://github.com/Junhua-Liao/Light-ASD.git third_party/Light-ASD
git -C third_party/Light-ASD checkout -q ed38c23 2>/dev/null || true

# ---- DiariZen ----
echo "== DiariZen environment"
[ -x .venv-diarizen/bin/python ] || $UV venv -q -p 3.10 .venv-diarizen
DPY=.venv-diarizen/bin/python
$UV pip install -q -p $DPY torch==2.1.1 torchaudio==2.1.1 $TORCH_DIAR
$UV pip install -q -p $DPY "numpy==1.26.4" einops h5py joblib librosa scipy soundfile toml tqdm pyyaml pandas \
    "accelerate==1.6.0" tabulate matplotlib "$ORT" huggingface_hub setuptools wheel
$UV pip install -q -p $DPY -e third_party/DiariZen --no-deps
(cd third_party/DiariZen/pyannote-audio && $ROOT/$UV pip install -q -p $ROOT/$DPY -e . -c ../constraints.txt --index-strategy unsafe-best-match $TORCH_DIAR --extra-index-url https://pypi.org/simple)

# ---- LatentSync ----
echo "== LatentSync environment"
[ -x .venv-latentsync/bin/python ] || $UV venv -q -p 3.10 .venv-latentsync
LPY=.venv-latentsync/bin/python

# MediaPipe Hands (a covered mouth is never redrawn) lives in its own venv: its protobuf pin breaks onnx/InsightFace.
[ -x .venv-hands/bin/python ] || $UV venv -q -p 3.10 .venv-hands
$UV pip install -q -p .venv-hands/bin/python "mediapipe==0.10.14" "opencv-python-headless<4.11" "numpy<2"

# GFPGAN face restoration after LatentSync, in its own venv (basicsr/facexlib pins must not touch LatentSync)
[ -x .venv-restore/bin/python ] || $UV venv -q -p 3.10 .venv-restore
$UV pip install -q -p .venv-restore/bin/python torch==2.5.1 torchvision==0.20.1 $TORCH_LS
$UV pip install -q -p .venv-restore/bin/python gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0 "opencv-python-headless<4.11" "numpy<2"
DEG=$(find .venv-restore -path "*basicsr/data/degradations.py" | head -1)
[ -n "$DEG" ] && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEG"
mkdir -p models/gfpgan
[ -f models/gfpgan/GFPGANv1.4.pth ] || curl -sL -o models/gfpgan/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Vocal separation with BS/Mel-RoFormer models through audio-separator, in its own venv (SEPARATION_BACKEND=roformer).
# The model file named by ROFORMER_MODEL (default vocals_mel_band_roformer.ckpt) is downloaded into models/separator on first use.
[ -x .venv-separate/bin/python ] || $UV venv -q -p 3.12 .venv-separate
$UV pip install -q -p .venv-separate/bin/python torch==2.5.1 torchaudio==2.5.1 $TORCH_LS
$UV pip install -q -p .venv-separate/bin/python "audio-separator[gpu]" audioread
mkdir -p models/separator
$UV pip install -q -p $LPY torch==2.5.1 torchvision==0.20.1 $TORCH_LS
grep -v -E "^(torch|torchvision|--extra-index-url|onnxruntime|gradio|mediapipe)" third_party/LatentSync/requirements.txt > /tmp/latentsync-req.txt
$UV pip install -q -p $LPY -r /tmp/latentsync-req.txt "$ORT" python_speech_features==0.6

# ---- checkpoints ----
echo "== checkpoints"
mkdir -p models
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("BUT-FIT/diarizen-wavlm-large-s80-md-v2", local_dir="models/diarizen-v2")
hf_hub_download("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin", local_dir="models/pyannote-wespeaker-voxceleb-resnet34-LM")
for name in ("latentsync_unet.pt", "whisper/tiny.pt", "stable_syncnet.pt"):
    hf_hub_download("ByteDance/LatentSync-1.6", name, local_dir="models/latentsync-1.6")
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir="models/sd-vae-ft-mse",
                  allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
print("checkpoints ready")
PY
# Wav2Lip CPU fallback weights (not kept in git): public copies, checked against the files the studio was built with
.venv/bin/python - <<'PY'
import hashlib, os, shutil, urllib.request
from huggingface_hub import hf_hub_download

WEIGHTS = [  # (target, sha256, source)
    ("Wav2Lip/checkpoints/wav2lip.pth", "b78b681b68ad9fe6c6fb1debc6ff43ad05834a8af8a62ffc4167b7b34ef63c37", ("hf", "Nekochu/Wav2Lip", "wav2lip.pth")),
    ("Wav2Lip/checkpoints/wav2lip_gan.pth", "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8", ("hf", "Nekochu/Wav2Lip", "wav2lip_gan.pth")),
    ("Wav2Lip/face_detection/detection/sfd/s3fd.pth", "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543",
     ("hf", "camenduru/Wav2Lip", "face_detection/detection/sfd/s3fd.pth")),
    ("Wav2Lip/checkpoints/realesr-general-x4v3.pth", "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
     ("url", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth")),
]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

for target, expected, source in WEIGHTS:
    if os.path.isfile(target) and sha256(target) == expected:
        continue
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if source[0] == "hf":
        shutil.copyfile(hf_hub_download(source[1], source[2]), target)
    else:
        urllib.request.urlretrieve(source[1], target)
    if sha256(target) != expected:
        os.remove(target)
        raise SystemExit(f"checksum mismatch for {target}")
print("wav2lip weights ready")
PY
# InsightFace detector + 106-point landmarks (buffalo_l), fetched by insightface itself
$LPY -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='$ROOT/models/insightface', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640)); print('insightface ready')"
echo "== done ($MODE). Set DEVICE=$MODE in .env if it differs."
