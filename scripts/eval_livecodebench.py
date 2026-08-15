#!/usr/bin/env python
"""Launch the official LiveCodeBench code-generation harness.

Run this from an installed/cloned LiveCodeBench environment. The wrapper keeps
release, scenario, prompt style, and local checkpoint path explicit in the run
metadata instead of silently depending on harness defaults.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--model-style", default="qwen")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--scenario", default="codegeneration")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m", "lcb_runner.runner.main",
        "--model", args.model_style,
        "--local_model_path", args.local_model_path,
        "--scenario", args.scenario,
        "--release_version", args.release_version,
        "--evaluate",
        "--use_cache",
    ]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
