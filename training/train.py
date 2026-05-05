"""CRISP training entrypoint.

This script wires together:
- YAML / dict config loading
- dataset preparation
- model + tokenizer loading
- LoRA attachment
- optimizer setup
- repeated calls into training.crisp_step.crisp_step
- periodic checkpointing and evaluation

It is intentionally self-contained and conservative so it can run before being
integrated into a larger RL framework such as veRL or OpenRLHF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception as e:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None

from .crisp_step import crisp_step


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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Dataset preparation
# -----------------------------


def _normalize_example(example: Mapping[str, Any]) -> Dict[str, str]:
    prompt = example.get("prompt") or example.get("problem") or example.get("question") or example.get("input")
    answer = example.get("answer") or example.get("solution") or example.get("output") or example.get("target")
    if prompt is None or answer is None:
        raise KeyError(f"Example missing prompt/answer fields: {list(example.keys())}")
    return {"prompt": str(prompt), "answer": str(answer)}


def build_train_dataset(config: Mapping[str, Any]) -> Dataset:
    data_cfg = config.get("data", {})
    sources = data_cfg.get("train_datasets", [])
    if not sources:
        raise ValueError("No train_datasets configured")

    datasets: List[Dataset] = []
    for source in sources:
        name = source["name"]
        split = source.get("split", "train")
        weight = float(source.get("weight", 1.0))
        ds = load_dataset(name, split=split)
        ds = ds.map(_normalize_example, remove_columns=ds.column_names)
        if weight != 1.0:
            n = max(1, int(len(ds) * weight))
            ds = ds.select(range(n))
        datasets.append(ds)

    train_ds = concatenate_datasets(datasets).shuffle(seed=int(config.get("experiment", {}).get("seed", 42)))
    return train_ds


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
# Model / tokenizer setup
# -----------------------------


def load_tokenizer(config: Mapping[str, Any]):
    model_name = deep_get(config, "model.name")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(deep_get(config, "model.trust_remote_code", True)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(config: Mapping[str, Any]):
    model_name = deep_get(config, "model.name")
    dtype_str = str(deep_get(config, "model.dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_str in {"bf16", "bfloat16"} else torch.float16
    trust_remote_code = bool(deep_get(config, "model.trust_remote_code", True))
    use_flash = bool(deep_get(config, "model.use_flash_attention_2", True))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        use_flash_attention_2=use_flash,
    )

    if bool(deep_get(config, "model.gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if bool(deep_get(config, "lora.enabled", True)):
        if get_peft_model is None:
            raise RuntimeError("peft is not installed, cannot enable LoRA")
        lora_cfg = LoraConfig(
            r=int(deep_get(config, "lora.r", 64)),
            lora_alpha=int(deep_get(config, "lora.lora_alpha", 128)),
            lora_dropout=float(deep_get(config, "lora.lora_dropout", 0.05)),
            bias=str(deep_get(config, "lora.bias", "none")),
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(deep_get(config, "lora.target_modules", [])),
            modules_to_save=list(deep_get(config, "lora.modules_to_save", [])),
        )
        model = get_peft_model(model, lora_cfg)

    return model


# -----------------------------
# Optimizer / checkpointing
# -----------------------------


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1e-5))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
    eps = float(opt_cfg.get("eps", 1e-8))
    weight_decay = float(opt_cfg.get("weight_decay", 0.1))

    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


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

    # Save the adapter / model state.
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(ckpt_dir))
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    # Save tokenizer for reproducibility.
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(ckpt_dir))

    # Save optimizer and metadata.
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    meta = {
        "step": step,
        "config": config,
    }
    with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ckpt_dir


# -----------------------------
# Evaluation
# -----------------------------


def _build_eval_dataset_from_spec(spec: Mapping[str, Any]) -> Dataset:
    source = spec.get("source", "custom")
    split = spec.get("split", "test")

    if source == "custom":
        # Expect a prebuilt dataset via local files or later injection.
        raise ValueError("custom eval dataset requested, but no loader is configured")

    ds = load_dataset(source, split=split)
    ds = ds.map(_normalize_example, remove_columns=ds.column_names)
    return ds


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    config: Mapping[str, Any],
    step: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    eval_cfg = config.get("data", {}).get("eval_datasets", {})
    if not eval_cfg:
        return {}

    results: Dict[str, Dict[str, float]] = {}
    seeds = list(deep_get(config, "evaluation.seeds", [42, 123, 999]))
    max_new_tokens = int(deep_get(config, "evaluation.generation_max_new_tokens", 2048))
    deterministic = bool(deep_get(config, "evaluation.deterministic", True))

    from .crisp_step import STUDENT_TEMPLATE, reward_fn

    model.eval()
    for name, spec in eval_cfg.items():
        try:
            ds = _build_eval_dataset_from_spec(spec)
        except Exception:
            # Skip benchmarks that need a custom local loader; training can still proceed.
            continue

        seed_accs: List[float] = []
        seed_tokens: List[float] = []
        for seed in seeds:
            set_seed(int(seed))
            correct = 0.0
            total_tokens = 0
            for ex in ds:
                prompt = STUDENT_TEMPLATE.format(problem=ex["prompt"])
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=not deterministic,
                    temperature=0.0 if deterministic else 0.7,
                    top_p=1.0,
                )
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                correct += reward_fn(text, ex["answer"])
                total_tokens += int(out.shape[-1])
            seed_accs.append(correct / max(1, len(ds)))
            seed_tokens.append(float(total_tokens))

        results[name] = {
            "pass@1_mean": float(np.mean(seed_accs)),
            "pass@1_std": float(np.std(seed_accs)),
            "tokens_mean": float(np.mean(seed_tokens)),
            "tokens_per_correct": float(np.mean(seed_tokens) / max(np.mean(seed_accs) * max(1, len(ds)), 1e-8)),
        }

    model.train()
    return results


# -----------------------------
# Logging helpers
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
        tags=list(deep_get(config, "experiment.tags", [])),
        name=deep_get(config, "experiment.name", "crisp_7b"),
        dir=deep_get(config, "experiment.output_dir", "runs/crisp_7b"),
    )


# -----------------------------
# Main training loop
# -----------------------------


def train(config_or_path: str | Mapping[str, Any]) -> Dict[str, Any]:
    config = load_config(config_or_path)
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})

    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(exp_cfg.get("output_dir", "runs/crisp_7b"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config)
    model = load_model(config)
    device = next(model.parameters()).device
    optimizer = build_optimizer(model, config)

    # Dataset / dataloader.
    train_ds = build_train_dataset(config)
    micro_batch_size = int(train_cfg.get("micro_batch_size", 8))
    loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, drop_last=True)
    endless_loader = EndlessDataLoader(loader)

    # Optional wandb.
    run = init_wandb(config)

    total_steps = int(train_cfg.get("total_steps", 1500))
    log_every = int(train_cfg.get("log_every", 1))
    save_every = int(train_cfg.get("save_every", 100))
    eval_every = int(train_cfg.get("eval_every", 100))

    best_metric = -math.inf
    history: List[Dict[str, float]] = []
    cumulative_tokens = 0.0

    for step in range(total_steps):
        batch_raw = next(endless_loader)
        if isinstance(batch_raw, Mapping):
            # HF DataLoader may collate dicts into dict-of-lists.
            prompts = batch_raw["prompt"]
            answers = batch_raw["answer"]
            batch = [{"prompt": p, "answer": a} for p, a in zip(prompts, answers)]
        else:
            batch = [dict(x) for x in batch_raw]

        logs = crisp_step(model, tokenizer, batch, config, step=step, optimizer=optimizer, device=device)
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
                eval_results = evaluate_model(model, tokenizer, config, step=step + 1, device=device)
            except Exception as e:
                eval_results = {}

            # Flatten eval metrics for logging.
            flat_eval: Dict[str, float] = {}
            for bm, metrics in eval_results.items():
                for k, v in metrics.items():
                    flat_eval[f"eval/{bm}/{k}"] = float(v)
            if run is not None and flat_eval:
                run.log(flat_eval, step=step + 1)

            # Track a simple best model criterion if evaluation exists.
            if eval_results:
                primary_bm = next(iter(eval_results.keys()))
                score = float(eval_results[primary_bm].get("pass@1_mean", -math.inf))
                if score > best_metric:
                    best_metric = score
                    save_checkpoint(model, optimizer, config, step + 1, output_dir=output_dir / "best", tokenizer=tokenizer)

    # Final save.
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
    parser = argparse.ArgumentParser(description="Train CRISP")
    parser.add_argument("--config", type=str, required=True, help="Path to crisp_7b.yaml or JSON config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(args.config)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
