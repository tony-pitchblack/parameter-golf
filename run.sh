#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a
source .env
set +a
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS:-1}}"

if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got: $NPROC_PER_NODE" >&2
  exit 1
fi

if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  exec python train_gpt_bare.py "$@"
fi

exec torchrun --nproc_per_node="$NPROC_PER_NODE" train_gpt_bare.py "$@"
