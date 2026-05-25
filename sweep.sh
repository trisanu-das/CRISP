#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/crisp_7b.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LAUNCHER="$ROOT_DIR/scripts/train_launcher.py"

usage() {
  cat <<EOF
Usage: bash scripts/sweep.sh <target>

Targets:
  crisp             Run CRISP
  grpo_k1           Run GRPO with k=1
  grpo_k8           Run GRPO with k=8
  reinforce_pp      Run REINFORCE++
  opsd              Run OPSD
  lambda_sweep      Run a small CRISP lambda grid
  pcgrad_ablation   Run CRISP with and without PC-Grad
EOF
}

run() {
  "$PYTHON_BIN" "$LAUNCHER" "$@"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  target="$1"
  shift || true

  case "$target" in
    crisp)
      run crisp --config "$CONFIG_PATH" "$@"
      ;;
    grpo_k1)
      run grpo --config "$CONFIG_PATH" --k 1 "$@"
      ;;
    grpo_k8)
      run grpo --config "$CONFIG_PATH" --k 8 "$@"
      ;;
    reinforce_pp)
      run reinforce_pp --config "$CONFIG_PATH" "$@"
      ;;
    opsd)
      run opsd --config "$CONFIG_PATH" "$@"
      ;;
    lambda_sweep)
      for lam in 0 0.1 0.5 1.0 2.0; do
        run crisp --config "$CONFIG_PATH" --name "crisp_lambda_${lam}" --lambda-max "$lam" "$@"
      done
      ;;
    pcgrad_ablation)
      run crisp --config "$CONFIG_PATH" --name "crisp_pcgrad_on" --pc-grad on "$@"
      run crisp --config "$CONFIG_PATH" --name "crisp_pcgrad_off" --pc-grad off "$@"
      ;;
    *)
      echo "Unknown target: $target" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
