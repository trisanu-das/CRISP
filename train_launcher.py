#!/usr/bin/env python
"""Command-line entry point for CRISP controls and successor methods.

Examples:

    python train_launcher.py crisp --config config/crisp_pilot.yaml
    python train_launcher.py grpo --config config/crisp_pilot.yaml --k 8
    python train_launcher.py vacs --config config/vacs_7b.yaml
    python train_launcher.py cibo-v2 --config config/cibo_v2_7b.yaml

Any config value can be overridden without editing YAML, for example:

    python train_launcher.py vacs --config config/vacs_7b.yaml \\
        --override vacs.token_credit_mix=0.5 \\
        --override logging.run_name=vacs_credit_05

The launcher validates the resolved configuration before it imports a method
runner. That prevents an invalid VACS/CIBO configuration from downloading a
model or dataset before reporting the actionable validation error.
"""
from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from typing import Optional

import yaml

from crisp.utils.config import load_config
from crisp.utils.validation import validate_config


SUPPORTED_METHODS = (
    "crisp",
    "grpo",
    "reinforce_pp",
    "opsd",
    "vacs",
    "cibo-v2",
    "sft",
)

# Module import is intentionally deferred until after config validation. VACS
# and CIBO runners are added in later implementation tasks; keeping their
# paths here makes the public CLI stable without importing model code at
# parser-import time.
_RUNNER_SPECS: dict[str, tuple[str, str]] = {
    "crisp": ("crisp.training.train", "main"),
    "grpo": ("crisp.baselines.grpo.training_loop", "main"),
    "reinforce_pp": ("crisp.baselines.reinforce_pp.training_loop", "main"),
    "opsd": ("crisp.baselines.opsd.training_loop", "main"),
    "vacs": ("crisp.training.vacs_train", "main"),
    "cibo-v2": ("crisp.training.cibo_v2_train", "main"),
    "sft": ("crisp.training.sft_train", "main"),
}


def parse_overrides(pairs: list[str]) -> dict:
    """Parse repeatable ``key.path=value`` CLI overrides into a nested dict."""
    overrides: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--override expects key.path=value, got: {pair!r}")
        key_path, value_str = pair.split("=", 1)
        if not key_path or any(not key for key in key_path.split(".")):
            raise ValueError(f"--override key path must contain non-empty dot-separated keys, got: {pair!r}")
        value = yaml.safe_load(value_str)
        cursor = overrides
        keys = key_path.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return overrides


def validate_method(method: str) -> str:
    """Return a supported method name or fail before any runner is imported."""
    if method not in SUPPORTED_METHODS:
        choices = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"Unknown method {method!r}. Supported methods: {choices}.")
    return method


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without importing any training/model modules."""
    parser = argparse.ArgumentParser(description="CRISP, VACS-CRISP, and CIBO-CRISP v2 training launcher")
    parser.add_argument("method", choices=SUPPORTED_METHODS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=None, help="rollouts per prompt (GRPO only; defaults to rollout.k from config)")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.path=value",
        help="override a config value; repeatable; values use YAML scalar/list syntax",
    )
    return parser


def _load_runner(method: str) -> Callable:
    """Import one runner lazily after method/config validation succeeds."""
    module_name, attribute = _RUNNER_SPECS[method]
    module = importlib.import_module(module_name)
    runner = getattr(module, attribute)
    if not callable(runner):  # defensive: turns a broken runner module into a clear launcher failure
        raise TypeError(f"Runner '{module_name}.{attribute}' for method {method!r} is not callable.")
    return runner


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    method = validate_method(args.method)
    overrides = parse_overrides(args.override)

    # Config validation is deliberately before _load_runner(): importing a
    # runner eventually imports torch/transformers and may load a model.
    resolved_cfg = load_config(args.config, overrides=overrides)
    validate_config(resolved_cfg)
    run = _load_runner(method)

    if method == "grpo":
        k = int(args.k if args.k is not None else resolved_cfg.rollout.k)
        run(args.config, k=k, overrides=overrides)
    else:
        run(args.config, overrides=overrides)


if __name__ == "__main__":
    main()
