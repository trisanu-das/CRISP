"""
Standalone evaluation runner: `python -m crisp.eval.run_eval --config ... [--checkpoint ...]`

Batches generation (`evaluation.eval_batch_size`) instead of one example at a
time, and `evaluation.eval_subset_size` caps how many problems per benchmark
get evaluated -- useful for a quick sanity check before committing to a full
(and, for something like MATH-500, slow) eval pass.

HumanEval is handled as a special case: correctness there comes from
executing the model's code against unit tests (crisp/data/reward.py:
code_reward), not from comparing an "answer" string, and greedy decoding
with a code-oriented system prompt makes more sense than the math one. Every
other dataset goes through the generic `reward_fn(response_text, answer)`
path.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import torch

from crisp.data.build_dataset import load_eval_dataset
from crisp.data.reward import build_reward_fn, code_reward, extract_code_completion
from crisp.eval.metrics import summarize
from crisp.model.load import load_model_and_tokenizer
from crisp.training.rollout import generate_rollouts
from crisp.utils.config import load_config

_CODE_SYSTEM_PROMPT = (
    "Complete the following Python function. Respond with the full function "
    "body in a single ```python fenced code block, and nothing else."
)


@torch.no_grad()
def evaluate_dataset(model, tokenizer, dataset_name: str, dataset: list[dict], reward_fn, cfg) -> dict:
    model.eval()
    batch_size = cfg.evaluation.eval_batch_size
    subset_size = cfg.evaluation.eval_subset_size
    subset = dataset if subset_size is None else dataset[:subset_size]
    is_code = dataset_name == "humaneval"
    system_prompt = _CODE_SYSTEM_PROMPT if is_code else cfg.model.system_prompt

    results = []
    for start in range(0, len(subset), batch_size):
        batch = subset[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        rollouts = generate_rollouts(
            model, tokenizer, prompts, system_prompt,
            max_prompt_length=cfg.rollout.max_prompt_length,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            temperature=0.0, top_p=1.0, do_sample=False,  # greedy: reproducible eval
        )
        for ex, rollout in zip(batch, rollouts):
            if is_code:
                completion = extract_code_completion(rollout.response_text)
                reward = code_reward(ex, completion, timeout=cfg.reward.code_exec_timeout)
            else:
                reward = reward_fn(rollout.response_text, ex["answer"])
            results.append({"correct": reward >= 1.0, "response_tokens": len(rollout.response_ids)})
    return summarize(
        results,
        n_bootstrap=int(cfg.evaluation.n_bootstrap),
        bootstrap_seed=int(cfg.evaluation.get("bootstrap_seed", cfg.seed)),
    )


def run_eval(cfg, model=None, tokenizer=None) -> dict:
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(cfg.model)
    model.eval()
    reward_fn = build_reward_fn(cfg.reward)

    all_metrics = {}
    for name in cfg.data.eval_datasets:
        dataset = load_eval_dataset(name)
        all_metrics[name] = evaluate_dataset(model, tokenizer, name, dataset, reward_fn, cfg)
    return all_metrics


def make_eval_fn(cfg):
    """Build the periodic in-training eval hook used by training/common_loop.py."""

    def eval_fn(model, tokenizer, step: int) -> dict:
        metrics = run_eval(cfg, model=model, tokenizer=tokenizer)
        flat = {}
        for dataset_name, m in metrics.items():
            for k, v in m.items():
                flat[f"{dataset_name}/{k}"] = v
        return flat

    return eval_fn


def main(config_path: str, checkpoint: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    if checkpoint:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint)
    metrics = run_eval(cfg, model=model, tokenizer=tokenizer)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="LoRA adapter dir to load on top of the base model")
    args = parser.parse_args()
    main(args.config, checkpoint=args.checkpoint)
