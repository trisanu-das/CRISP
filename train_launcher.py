#!/usr/bin/env python3
"""Unified CLI launcher for CRISP and baseline training runs.

Examples:
    python scripts/train_launcher.py crisp --config config/crisp_7b.yaml
    python scripts/train_launcher.py grpo --config config/crisp_7b.yaml --k 8
    python scripts/train_launcher.py reinforce_pp --config config/crisp_7b.yaml
    python scripts/train_launcher.py opsd --config config/crisp_7b.yaml
    python scripts/train_launcher.py crisp --config config/crisp_7b.yaml --lambda-max 0.5 --pc-grad on
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format: {p.suffix}")


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in {"1", "true", "yes", "on"}:
        return True
    if t in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRISP launcher with direct CLI overrides")
    parser.add_argument("target", choices=["crisp", "grpo", "reinforce_pp", "opsd"], help="Run target")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--name", default=None, help="Override experiment.name")
    parser.add_argument("--output-dir", default=None, help="Override experiment.output_dir")
    parser.add_argument("--lambda-max", type=float, default=None, help="Override crisp.lambda_max")
    parser.add_argument("--lambda-schedule", choices=["cosine", "linear", "constant"], default=None, help="Override crisp.lambda_schedule")
    parser.add_argument("--pc-grad", type=parse_bool, nargs="?", const=True, default=None, help="Override crisp.pc_grad")
    parser.add_argument("--entropy-bonus", type=float, default=None, help="Override crisp.entropy_bonus")
    parser.add_argument("--correctness-gate", type=parse_bool, nargs="?", const=True, default=None, help="Override crisp.correctness_gate")
    parser.add_argument("--k", type=int, default=None, help="GRPO rollout count")
    return parser


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    if args.name is not None:
        set_nested(cfg, "experiment.name", args.name)
    if args.output_dir is not None:
        set_nested(cfg, "experiment.output_dir", args.output_dir)
    if args.lambda_max is not None:
        set_nested(cfg, "crisp.lambda_max", float(args.lambda_max))
    if args.lambda_schedule is not None:
        set_nested(cfg, "crisp.lambda_schedule", args.lambda_schedule)
    if args.pc_grad is not None:
        set_nested(cfg, "crisp.pc_grad", bool(args.pc_grad))
    if args.entropy_bonus is not None:
        set_nested(cfg, "crisp.entropy_bonus", float(args.entropy_bonus))
    if args.correctness_gate is not None:
        set_nested(cfg, "crisp.correctness_gate", bool(args.correctness_gate))
    if args.k is not None:
        set_nested(cfg, "baselines.grpo_k1.k", int(args.k))
        set_nested(cfg, "baselines.grpo_k8.k", int(args.k))

    return cfg


def run_target(target: str, cfg: Dict[str, Any]) -> None:
    if target == "crisp":
        from training.train import train
        train(cfg)
    elif target == "grpo":
        from baselines.train_grpo import train
        train(cfg)
    elif target == "reinforce_pp":
        from baselines.train_reinforce_pp import train
        train(cfg)
    elif target == "opsd":
        from baselines.train_opsd import train
        train(cfg)
    else:
        raise ValueError(f"Unknown target: {target}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    if args.target == "grpo" and args.k is not None:
        cfg.setdefault("baselines", {}).setdefault("grpo_k1", {})["k"] = args.k
        cfg.setdefault("baselines", {}).setdefault("grpo_k8", {})["k"] = args.k

    run_target(args.target, cfg)


if __name__ == "__main__":
    main()
