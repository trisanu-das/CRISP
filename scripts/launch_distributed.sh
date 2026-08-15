#!/usr/bin/env bash
set -euo pipefail

# Example:
#   NUM_GPUS=2 METHOD=crisp CONFIG=config/paper_7b.yaml bash scripts/launch_distributed.sh
NUM_GPUS="${NUM_GPUS:-2}"
METHOD="${METHOD:-crisp}"
CONFIG="${CONFIG:-config/paper_7b.yaml}"

exec torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  train_launcher.py "$METHOD" --config "$CONFIG" "${@:1}"
