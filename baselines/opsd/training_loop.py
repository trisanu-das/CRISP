"""Training loop for the OPSD baseline.

This mirrors the CRISP / GRPO / REINFORCE++ training harness, but uses the
on-policy self-distillation objective defined in baselines/opsd.py.

The goal is experimental parity:
- same config loading
- same dataset loading
- same model / tokenizer loading
- same optimizer / checkpointing / logging / eval cadence
- different loss only
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from data.build_dataset import normalize_problem_answer_row
from model.load import load_model, load_tokenizer, get_model_device
from eval.run_eval import build_benchmarks, run_eval
from .opsd import opsd_step, OPSDConfig


# -----------------------------
# Config helpers
# -----------------------------


def load_config(path_or_obj: str | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(path_or_obj, Mapping):
        return dict(path_or_obj)

    path = Path(path_or_obj)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not available, cannot load YAML config")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {path.suffix}")


def deep_get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


# -----------------------------
# Reproducibility
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Dataset loading
# -----------------------------


def build_train_dataset(config: Mapping[str, Any]):
    data_cfg = config.get("data", {})
    sources = data_cfg.get("train_datasets", [])
    if not sources:
        raise ValueError("No train_datasets configured")

    datasets = []
    for source in sources:
        name = source["name"]
        split = source.get("split", "train")
        weight = float(source.get("weight", 1.0))
        ds = load_dataset(name, split=split)
        ds = ds.map(
            lambda row: normalize_problem_answer_row(row, source=name, split=split),
            remove_columns=ds.column_names,
        )
        if weight != 1.0:
            n = max(1, int(len(ds) * weight))
            ds = ds.select(range(n))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    combined = combined.shuffle(seed=int(config.get("experiment", {}).get("seed", 42)))
    return combined


class EndlessDataLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter: Optional[Iterator[Any]] = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# -----------------------------
# Logging / checkpointing
# -----------------------------


def init_wandb(config: Mapping[str, Any]):
    if wandb is None:
        return None
    logging_cfg = config.get("logging", {})
    if logging_cfg.get("backend", "wandb") != "wandb":
        return None
    return wandb.init(
        project=logging_cfg.get("wandb_project", "crisp"),
        entity=logging_cfg.get("wandb_entity"),
        config=config,
        name=deep_get(config, "experiment.name", "opsd_baseline"),
        dir=deep_get(config, "experiment.output_dir", "runs/opsd_baseline"),
        tags=list(deep_get(config, "experiment.tags", [])),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    step: int,
    output_dir: str | Path,
    tokenizer=None,
) -> Path:
    out = Path(output_dir)
    ckpt_dir = out / f"checkpoint-{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(ckpt_dir))
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(ckpt_dir))

    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"step": step, "config": config}, f, indent=2)

    return ckpt_dir


# -----------------------------
# Main training loop
# -----------------------------


def train(config_or_path: str | Mapping[str, Any]) -> Dict[str, Any]:
    config = load_config(config_or_path)
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})

    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(exp_cfg.get("output_dir", "runs/opsd_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config)
    model = load_model(config)
    device = get_model_device(model)

    optimizer_cfg = config.get("optimizer", {})
    lr = float(optimizer_cfg.get("lr", 1e-5))
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.95]))
    eps = float(optimizer_cfg.get("eps", 1e-8))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.1))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    train_ds = build_train_dataset(config)
    micro_batch_size = int(train_cfg.get("micro_batch_size", 8))
    loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, drop_last=True)
    endless_loader = EndlessDataLoader(loader)

    run = init_wandb(config)

    total_steps = int(train_cfg.get("total_steps", 1500))
    log_every = int(train_cfg.get("log_every", 1))
    save_every = int(train_cfg.get("save_every", 100))
    eval_every = int(train_cfg.get("eval_every", 100))

    cumulative_tokens = 0.0
    history: List[Dict[str, float]] = []
    best_metric = -math.inf

    opsd_cfg = OPSDConfig(
        total_steps=total_steps,
        max_new_tokens=int(config.get("rollout", {}).get("max_new_tokens", 1024)),
        temperature=float(config.get("rollout", {}).get("temperature", 0.7)),
        top_p=float(config.get("rollout", {}).get("top_p", 1.0)),
        top_k=int(config.get("rollout", {}).get("top_k", 0)),
        do_sample=bool(config.get("rollout", {}).get("do_sample", True)),
        stop_on_eos=bool(config.get("rollout", {}).get("stop_on_eos", True)),
        entropy_bonus=float(config.get("crisp", {}).get("entropy_bonus", 0.0)),
        correctness_gate=bool(config.get("crisp", {}).get("correctness_gate", True)),
        correctness_gate_mode=str(config.get("crisp", {}).get("correctness_gate_mode", "wrong_only")),
        sd_loss_weight=float(config.get("crisp", {}).get("sd_loss_weight", 1.0)),
        distill_on_prefix_only=bool(config.get("crisp", {}).get("distill_on_prefix_only", True)),
        use_on_policy_rollout=bool(config.get("crisp", {}).get("use_on_policy_rollout", True)),
    )

    for step in range(total_steps):
        batch_raw = next(endless_loader)
        if isinstance(batch_raw, Mapping):
            batch = [
                {"prompt": p, "answer": a}
                for p, a in zip(batch_raw["prompt"], batch_raw["answer"])
            ]
        else:
            batch = [dict(x) for x in batch_raw]

        logs = opsd_step(model, tokenizer, batch, opsd_cfg, step=step, optimizer=optimizer, device=device)
        cumulative_tokens += logs.get("efficiency/tokens_generated", 0.0)
        logs["efficiency/tokens_cumulative"] = cumulative_tokens
        logs["train/step"] = float(step)
        history.append(logs)

        if run is not None and (step % log_every == 0):
            run.log(logs, step=step)

        if (step + 1) % save_every == 0:
            save_checkpoint(model, optimizer, config, step + 1, output_dir=output_dir, tokenizer=tokenizer)

        if (step + 1) % eval_every == 0:
            try:
                benchmarks = build_benchmarks(config)
                eval_results = run_eval(
                    model,
                    tokenizer,
                    benchmarks,
                    seeds=list(config.get("evaluation", {}).get("seeds", [42, 123, 999])),
                    max_new_tokens=int(config.get("evaluation", {}).get("generation_max_new_tokens", 2048)),
                    deterministic=bool(config.get("evaluation", {}).get("deterministic", True)),
                    device=device,
                )
            except Exception:
                eval_results = {}

            flat_eval: Dict[str, float] = {}
            for bm, metrics in eval_results.items():
                for k_metric, v in metrics.items():
                    flat_eval[f"eval/{bm}/{k_metric}"] = float(v)
            if run is not None and flat_eval:
                run.log(flat_eval, step=step + 1)

            if eval_results:
                primary_bm = next(iter(eval_results.keys()))
                score = float(eval_results[primary_bm].get("pass@1_mean", -math.inf))
                if score > best_metric:
                    best_metric = score
                    save_checkpoint(model, optimizer, config, step + 1, output_dir=output_dir / "best", tokenizer=tokenizer)

    save_checkpoint(model, optimizer, config, total_steps, output_dir=output_dir, tokenizer=tokenizer)
    if run is not None:
        run.finish()

    return {
        "final_step": total_steps,
        "best_metric": best_metric,
        "output_dir": str(output_dir),
        "history": history,
    }


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OPSD baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to crisp_7b.yaml or JSON config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(args.config)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
