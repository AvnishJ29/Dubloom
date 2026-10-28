#!/usr/bin/env bash
# Copy the studio to another machine (for example the H100 cluster) with rsync.
#
#   scripts/deploy_to_cluster.sh USER@HOST:/path/to/dubloom [-J JUMP_HOST] [--with-models] [--with-env]
#
#   -J JUMP_HOST    reach HOST through a login node (ssh -J)
#   --with-models   also copy models/ (7.5 GB); otherwise run scripts/setup_models.sh on the target
#   --with-env      also copy .env (contains API keys); otherwise copy .env.example and fill it in there
#
# Never copied: virtual environments (rebuilt per machine), jobs/, logs/, node_modules/, old experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:?usage: scripts/deploy_to_cluster.sh USER@HOST:/path [-J JUMP_HOST] [--with-models] [--with-env]}
shift
JUMP="" MODELS=0 ENV=0
while [ $# -gt 0 ]; do
  case "$1" in
    -J) JUMP="$2"; shift 2;;
    --with-models) MODELS=1; shift;;
    --with-env) ENV=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
SSH="ssh -o ServerAliveInterval=30"
[ -n "$JUMP" ] && SSH="$SSH -J $JUMP"
EXCLUDES=(--exclude '.venv*/' --exclude 'jobs/' --exclude 'logs/' --exclude 'node_modules/' --exclude '__pycache__/'
          --exclude '*.pyc' --exclude 'example_archive/' --exclude 'studio_settings.json' --exclude '.env.backup-*'
          --exclude 'Wav2Lip/temp/' --exclude 'Wav2Lip/results/' --exclude 'third_party/*/.git/' --exclude '.runtime/')
[ "$MODELS" = 1 ] || EXCLUDES+=(--exclude 'models/')
[ "$ENV" = 1 ] || EXCLUDES+=(--exclude '.env')
rsync -az --partial --info=stats1,progress2 -e "$SSH" "${EXCLUDES[@]}" ./ "$DEST/"
echo "Copied to $DEST"
echo "On the target: scripts/setup_models.sh cuda && cp -n .env.example .env (then add the keys) && ./run.sh"
