"""
Lightweight run logger.

Always writes JSONL to disk (a run is never silently unlogged), and
*optionally* mirrors to Weights & Biases if `logging.backend: wandb` in the
config and the `wandb` package is importable and `WANDB_DISABLED` isn't set
(matching the reference README's "export WANDB_DISABLED=true" escape
hatch). A wandb hiccup (no API key, offline machine, network egress
disabled, ...) prints a warning and falls back to local-only rather than
crashing a training run over a logging backend.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str, run_name: str, config: dict, backend: str = "none"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / f"{run_name}.jsonl"
        self._fh = open(self.jsonl_path, "a", buffering=1)
        self._wandb = None

        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
            backend = "none"

        if backend == "wandb":
            try:
                import wandb

                wandb_project = config.get("logging", {}).get("wandb_project", "crisp")
                wandb.init(project=wandb_project, name=run_name, config=config)
                self._wandb = wandb
            except Exception as e:  # pragma: no cover - environment dependent
                print(
                    f"[logging] wandb backend requested but unavailable ({e}); "
                    f"continuing with local JSONL only at {self.jsonl_path}"
                )
                self._wandb = None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, "time": time.time(), **metrics}
        self._fh.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"[logging] wandb.log failed ({e}); continuing with local JSONL only")

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
