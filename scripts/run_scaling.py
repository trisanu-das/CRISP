#!/usr/bin/env python
"""Run a matched model-scaling matrix serially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "config/scaling/qwen25_1p5b.yaml",
    "config/scaling/qwen25_3b.yaml",
    "config/scaling/qwen25_7b.yaml",
    "config/scaling/qwen25_14b.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="reinforce_pp,crisp,cibo-v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for config in args.configs:
        config_path = Path(config)
        scale_name = config_path.stem
        for method in methods:
            for seed in seeds:
                output = f"runs/scaling/{scale_name}/{method}/seed_{seed}"
                cmd = [
                    sys.executable, "train_launcher.py", method,
                    "--config", str(config_path),
                    "--override", f"seed={seed}",
                    "--override", f"training.output_dir={output}",
                    "--override", f"logging.run_name={scale_name}-{method}-seed-{seed}",
                ]
                if method == "grpo":
                    cmd += ["--k", "8"]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
