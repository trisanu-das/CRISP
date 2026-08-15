#!/usr/bin/env bash
# Ablation sweep matching README Section 5.5: a lambda sweep for CRISP, plus
# the three baselines at matched rollout budget. Runs sequentially -- adapt
# to your job scheduler (Slurm array, background `&` + `wait`, ...) to run
# these in parallel across GPUs instead.
#
# Usage: ./sweep.sh [config] [output_root]
set -euo pipefail

CONFIG="${1:-config/crisp_pilot.yaml}"
OUT_ROOT="${2:-runs/sweep}"
mkdir -p "$OUT_ROOT"

echo "== CRISP lambda sweep =="
for LAMBDA in 0 0.1 0.5 1.0 2.0; do
  RUN_NAME="lambda_${LAMBDA}"
  python train_launcher.py crisp --config "$CONFIG" \
    --override training.lambda_max="${LAMBDA}" \
    --override logging.run_name="${RUN_NAME}" \
    --override training.output_dir="${OUT_ROOT}/${RUN_NAME}" \
    2>&1 | tee "${OUT_ROOT}/${RUN_NAME}.log"
done

echo "== GRPO k sweep (rollout-budget comparison) =="
for K in 1 8; do
  RUN_NAME="grpo_k${K}"
  python train_launcher.py grpo --config "$CONFIG" --k "$K" \
    --override logging.run_name="${RUN_NAME}" \
    --override training.output_dir="${OUT_ROOT}/${RUN_NAME}" \
    2>&1 | tee "${OUT_ROOT}/${RUN_NAME}.log"
done

echo "== REINFORCE++ =="
python train_launcher.py reinforce_pp --config "$CONFIG" \
  --override logging.run_name="reinforce_pp" \
  --override training.output_dir="${OUT_ROOT}/reinforce_pp" \
  2>&1 | tee "${OUT_ROOT}/reinforce_pp.log"

echo "== OPSD =="
python train_launcher.py opsd --config "$CONFIG" \
  --override logging.run_name="opsd" \
  --override training.output_dir="${OUT_ROOT}/opsd" \
  2>&1 | tee "${OUT_ROOT}/opsd.log"

echo "Sweep complete. Logs and checkpoints under ${OUT_ROOT}/"
