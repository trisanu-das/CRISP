"""Evaluation runner for CRISP.

Supports:
- AIME-style local benchmark files
- MATH-500 style evaluation via a dataset source
- HumanEval generation + execution-style accuracy via the `evaluate` package if available,
  with a light fallback if it is not installed

Primary outputs:
- pass@1_mean
- pass@1_std
- tokens_mean
- tokens_per_correct
- bootstrap_ci_95

The module is intentionally self-contained so it can be run as a script or imported
from training code.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import evaluate as hf_evaluate
except Exception:  # pragma: no cover
    hf_evaluate = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

from data.build_dataset import format_student_prompt, load_aime_dataset
from data.reward import math_verify
from model.load import load_base_model, load_tokenizer, get_model_device
from .metrics import SeedResult, summarize_seed_results


# -----------------------------
# Config helpers
# -----------------------------


def _config_get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config(path_or_obj: str | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(path_or_obj, Mapping):
        return dict(path_or_obj)
    path = Path(path_or_obj)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# General utilities
# -----------------------------


def _safe_float(x: Any) -> float:
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _extract_code_block(text: str) -> str:
    """Best-effort code extraction for HumanEval-style generations."""
    if not text:
        return ""
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


# -----------------------------
# Model loading
# -----------------------------


def load_eval_model(config: Mapping[str, Any]):
    """Load the base model or adapter-wrapped model for evaluation."""
    model = load_base_model(config)

    # If a PEFT adapter path is present, wrap the base model.
    adapter_path = _config_get(config, "evaluation.adapter_path", None)
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, cannot load adapter")
        model = PeftModel.from_pretrained(model, adapter_path)

    if torch.cuda.is_available():
        model = model.to(get_model_device(model))
    return model


# -----------------------------
# Benchmark loading
# -----------------------------


def load_math500_dataset(config: Mapping[str, Any]) -> Dataset:
    """Load a MATH-500-like evaluation subset.

    By default this uses hendrycks/competition_math test split. If you have a
    specific MATH-500 file locally, set evaluation.math500_source to a local path.
    """
    source = _config_get(config, "evaluation.math500_source", None)
    if source:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"MATH-500 source not found: {path}")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, dict) and "data" in rows:
                rows = rows["data"]
        else:
            raise ValueError(f"Unsupported MATH-500 file format: {path.suffix}")
        return load_aime_dataset(rows, source_name="math500")  # same normalizer schema

    # Fallback: competition_math test split.
    ds = load_dataset("hendrycks/competition_math", split="test")
    rows = []
    for row in ds:
        rows.append({
            "problem": row.get("problem", ""),
            "answer": row.get("solution", ""),
            "source": "math500",
        })
    return load_aime_dataset(rows, source_name="math500")


def load_humaneval_dataset(config: Mapping[str, Any]) -> Dataset:
    """Load HumanEval.

    Prefer the canonical dataset from HuggingFace Datasets. The returned schema
    is normalized to {prompt, answer, source, ...}, where `answer` is the
    reference completion used only for execution evaluation.
    """
    source = _config_get(config, "evaluation.humaneval_source", "openai_humaneval")
    split = _config_get(config, "evaluation.humaneval_split", "test")
    ds = load_dataset(source, split=split)

    rows = []
    for row in ds:
        prompt = row.get("prompt") or row.get("question") or row.get("problem") or ""
        # HumanEval usually provides canonical_solution / test; keep solution as reference.
        answer = row.get("canonical_solution") or row.get("solution") or row.get("answer") or ""
        rows.append({
            "problem": prompt,
            "answer": answer,
            "source": "humaneval",
            "task_id": row.get("task_id", None),
            "entry_point": row.get("entry_point", None),
            "test": row.get("test", None),
        })
    return load_aime_dataset(rows, source_name="humaneval")


# -----------------------------
# Generation
# -----------------------------


def generate_completion(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 2048,
    deterministic: bool = True,
    temperature: float = 0.0,
) -> Tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not deterministic,
            temperature=temperature if not deterministic else 0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, int(out.shape[-1])


def generate_completions_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: torch.device,
    max_new_tokens: int = 2048,
    deterministic: bool = True,
    temperature: float = 0.0,
) -> List[Tuple[str, int]]:
    """Batched version of generate_completion.

    Without vLLM's request batching, evaluating one example at a time is the
    single slowest part of a Kaggle-scale run (500 MATH-500 problems x 3 seeds
    x one generate() call each adds up fast). This pads a whole chunk of
    prompts together and generates them in one call instead.
    """
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not deterministic,
            temperature=temperature if not deterministic else 0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    results: List[Tuple[str, int]] = []
    for i in range(out.shape[0]):
        text = tokenizer.decode(out[i], skip_special_tokens=True)
        results.append((text, int(out.shape[-1])))
    return results


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# -----------------------------
# Evaluation kernels
# -----------------------------


def eval_math_style(
    model: torch.nn.Module,
    tokenizer,
    dataset: Dataset,
    *,
    device: torch.device,
    max_new_tokens: int,
    deterministic: bool,
    seed: int,
    eval_batch_size: int = 8,
) -> Dict[str, float]:
    set_seed(seed)
    correct = 0.0
    total_tokens = 0
    seed_correct: List[float] = []

    rows = list(dataset)
    for row_batch in _chunks(rows, max(1, eval_batch_size)):
        prompts = [format_student_prompt(ex["prompt"]) for ex in row_batch]
        completions = generate_completions_batch(
            model,
            tokenizer,
            prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            deterministic=deterministic,
            temperature=0.0 if deterministic else 0.7,
        )
        for ex, (pred, tok) in zip(row_batch, completions):
            score = math_verify(pred, ex["answer"])
            correct += score
            total_tokens += tok
            seed_correct.append(float(score))

    n = max(len(dataset), 1)
    acc = correct / n
    tokens_per_correct = total_tokens / max(correct, 1e-8)
    return {
        "pass@1": float(acc),
        "tokens": float(total_tokens),
        "tokens_per_correct": float(tokens_per_correct),
        "correct_count": float(correct),
        "n": float(n),
        "example_mean": float(np.mean(seed_correct) if seed_correct else 0.0),
    }


def _run_python_tests(code: str, test: str, entry_point: str, timeout_s: int = 2) -> bool:
    """Best-effort HumanEval execution checker."""
    ns: Dict[str, Any] = {}
    full = code + "\n\n" + test

    def _timeout_handler(signum, frame):
        raise TimeoutError("HumanEval execution timed out")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        exec(full, ns, ns)
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def eval_humaneval(
    model: torch.nn.Module,
    tokenizer,
    dataset: Dataset,
    *,
    device: torch.device,
    max_new_tokens: int,
    deterministic: bool,
    seed: int,
    eval_batch_size: int = 8,
) -> Dict[str, float]:
    set_seed(seed)
    correct = 0.0
    total_tokens = 0
    seed_correct: List[float] = []

    # Prefer the official evaluate package if installed.
    has_eval = hf_evaluate is not None
    pass_at_k = None
    if has_eval:
        try:
            pass_at_k = hf_evaluate.load("code_eval")
        except Exception:
            pass_at_k = None

    rows = list(dataset)
    for row_batch in _chunks(rows, max(1, eval_batch_size)):
        prompts = [format_student_prompt(ex["prompt"]) for ex in row_batch]
        completions_batch = generate_completions_batch(
            model,
            tokenizer,
            prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            deterministic=deterministic,
            temperature=0.0 if deterministic else 0.2,
        )
        for ex, (pred_text, tok) in zip(row_batch, completions_batch):
            completion = _extract_code_block(pred_text)
            total_tokens += tok

            if pass_at_k is not None and ex.get("test"):
                try:
                    result = pass_at_k.compute(
                        predictions=[completion],
                        references=[ex["test"]],
                        k=[1],
                    )
                    score = float(result.get("pass@1", 0.0))
                except Exception:
                    score = 1.0 if _run_python_tests(completion, ex.get("test", ""), ex.get("entry_point") or "") else 0.0
            elif ex.get("test"):
                score = 1.0 if _run_python_tests(completion, ex.get("test", ""), ex.get("entry_point") or "") else 0.0
            else:
                try:
                    ast.parse(completion)
                    score = 1.0
                except Exception:
                    score = 0.0

            correct += score
            seed_correct.append(float(score))

    n = max(len(dataset), 1)
    acc = correct / n
    tokens_per_correct = total_tokens / max(correct, 1e-8)
    return {
        "pass@1": float(acc),
        "tokens": float(total_tokens),
        "tokens_per_correct": float(tokens_per_correct),
        "correct_count": float(correct),
        "n": float(n),
        "example_mean": float(np.mean(seed_correct) if seed_correct else 0.0),
    }


# -----------------------------
# Unified evaluation
# -----------------------------


def run_eval(
    model: torch.nn.Module,
    tokenizer,
    benchmarks: Mapping[str, Dataset],
    *,
    seeds: Sequence[int] = (42, 123, 999),
    max_new_tokens: int = 2048,
    deterministic: bool = True,
    device: Optional[torch.device] = None,
    eval_batch_size: int = 8,
) -> Dict[str, Dict[str, float]]:
    if device is None:
        device = get_model_device(model)

    results: Dict[str, Dict[str, float]] = {}
    for name, dataset in benchmarks.items():
        seed_results: List[SeedResult] = []

        for seed in seeds:
            if name.lower().startswith("human"):
                out = eval_humaneval(
                    model,
                    tokenizer,
                    dataset,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    deterministic=deterministic,
                    seed=int(seed),
                    eval_batch_size=eval_batch_size,
                )
            else:
                out = eval_math_style(
                    model,
                    tokenizer,
                    dataset,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    deterministic=deterministic,
                    seed=int(seed),
                    eval_batch_size=eval_batch_size,
                )

            seed_results.append(
                SeedResult(
                    pass_at_1=float(out["pass@1"]),
                    tokens=float(out["tokens"]),
                    correct_count=float(out["correct_count"]),
                    n=float(out["n"]),
                )
            )

        results[name] = summarize_seed_results(seed_results)

    return results


# -----------------------------
# Benchmark construction
# -----------------------------


def build_benchmarks(config: Mapping[str, Any]) -> Dict[str, Dataset]:
    eval_cfg = config.get("evaluation", {})
    data_eval_cfg = config.get("data", {}).get("eval_datasets", {})
    subset_size = _config_get(config, "evaluation.eval_subset_size", None)

    # Known-good public HF sources for years where the data-section config
    # left source as the "custom" placeholder (which has no loader and always
    # failed). Override at data.eval_datasets.<key>.source if you have your
    # own local files instead.
    _AIME_FALLBACK_SOURCE = {
        "aime2024": "Maxwell-Jia/AIME_2024",
        "aime2025": "math-ai/aime25",
    }
    _AIME_FALLBACK_SPLIT = {
        "aime2024": "train",
        "aime2025": "test",
    }

    benchmarks: Dict[str, Dataset] = {}

    # AIME benchmarks: prefer an explicit local source from config, else fall
    # back to a known public HF dataset. Verify these IDs still resolve to the
    # schema you expect before trusting results at scale; field names across
    # AIME mirrors vary and are matched case-insensitively (see
    # data.build_dataset.normalize_problem_answer_row).
    for key in ["aime2024", "aime2025", "aime2026"]:
        spec = data_eval_cfg.get(key) or eval_cfg.get(key)
        if not spec:
            continue
        source = spec.get("source")
        if not source or source == "custom":
            source = _AIME_FALLBACK_SOURCE.get(key)
            if source is None:
                continue
        year = spec.get("year")
        split = spec.get("split") or _AIME_FALLBACK_SPLIT.get(key, "train")
        try:
            ds = load_aime_dataset(source, year=year, split=split, source_name=key)
        except Exception:
            # Don't let one bad/unreachable benchmark source take down the
            # whole eval pass; skip it and keep the rest.
            continue
        if subset_size:
            ds = ds.select(range(min(int(subset_size), len(ds))))
        benchmarks[key] = ds

    # MATH-500.
    if "math500" in data_eval_cfg or "math500" in eval_cfg:
        ds = load_math500_dataset(config)
        if subset_size:
            ds = ds.select(range(min(int(subset_size), len(ds))))
        benchmarks["math500"] = ds

    # HumanEval.
    if "humaneval" in data_eval_cfg or "humaneval" in eval_cfg:
        ds = load_humaneval_dataset(config)
        if subset_size:
            ds = ds.select(range(min(int(subset_size), len(ds))))
        benchmarks["humaneval"] = ds

    return benchmarks


# -----------------------------
# Top-level API / CLI
# -----------------------------


def main(config_or_path: str | Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    config = load_config(config_or_path)
    seed = int(_config_get(config, "experiment.seed", 42))
    set_seed(seed)

    tokenizer = load_tokenizer(config)
    model = load_eval_model(config)
    device = get_model_device(model)

    benchmarks = build_benchmarks(config)
    if not benchmarks:
        raise ValueError("No benchmarks were configured for evaluation")

    eval_cfg = config.get("evaluation", {})
    max_new_tokens = int(eval_cfg.get("generation_max_new_tokens", 2048))
    deterministic = bool(eval_cfg.get("deterministic", True))
    seeds = list(eval_cfg.get("seeds", [42, 123, 999]))

    results = run_eval(
        model,
        tokenizer,
        benchmarks,
        seeds=seeds,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        device=device,
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CRISP evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = main(args.config)
    print(json.dumps(results, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
