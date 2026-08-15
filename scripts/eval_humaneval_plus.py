#!/usr/bin/env python
"""Run the official EvalPlus HumanEval+ evaluator on a merged model."""
from __future__ import annotations

import argparse
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged local model directory or HF model ID")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    executable = shutil.which("evalplus.evaluate") or "evalplus.evaluate"
    cmd = [
        executable,
        "--model", args.model,
        "--dataset", "humaneval",
        "--backend", args.backend,
        "--greedy",
    ]
    print(" ".join(cmd))
    if not args.dry_run:
        if shutil.which("evalplus.evaluate") is None:
            raise RuntimeError(
                "EvalPlus is not installed. Install its official package first: pip install -U evalplus"
            )
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
