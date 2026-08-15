"""GRPO training entrypoint."""
from __future__ import annotations

import argparse
import warnings
from typing import Optional

from crisp.baselines.grpo.training_step import collect_grpo_batch, score_grpo_batch
from crisp.data.build_dataset import load_train_dataset
from crisp.data.reward import build_reward_fn
from crisp.eval.run_eval import make_eval_fn
from crisp.model.load import attach_or_load_lora, load_model_and_tokenizer
from crisp.training.common_loop import run_training_loop
from crisp.utils.config import load_config
from crisp.utils.distributed import init_distributed
from crisp.utils.seeding import set_seed


def main(config_path: str, k: int = 8, overrides: Optional[dict] = None) -> None:
    cfg = load_config(config_path, overrides)
    init_distributed()
    set_seed(cfg.seed)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k == 1:
        warnings.warn(
            "k=1 has no non-trivial group-relative baseline; this run is a "
            "single-rollout global-policy-gradient control, not genuine GRPO.",
            stacklevel=2,
        )

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    initial_adapter = cfg.model.get("initial_adapter") or cfg.training.get("resume_from")
    model, _ = attach_or_load_lora(model, cfg.lora, initial_adapter=initial_adapter)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")

    rollout_batch_size = int(cfg.training.micro_batch_size)
    if rollout_batch_size % k != 0:
        raise ValueError(
            f"training.micro_batch_size ({rollout_batch_size}) must be divisible by k ({k})."
        )
    cfg.training.micro_batch_size = rollout_batch_size // k
    print(
        f"[grpo-runtime] rollout_batch={rollout_batch_size}, k={k}, "
        f"prompt_microbatch={cfg.training.micro_batch_size}, "
        f"grad_accum={cfg.training.grad_accum_steps}, "
        f"world_size={__import__("os").environ.get("WORLD_SIZE", "1")}"
    )

    train_dataset = load_train_dataset(
        cfg.data.train_dataset,
        split=cfg.data.train_split,
        train_subset_size=cfg.data.train_subset_size,
    )
    reward_fn = build_reward_fn(cfg.reward)

    def collect_fn(model_, tokenizer_, prompts, answers, global_step):
        return collect_grpo_batch(model_, tokenizer_, prompts, answers, reward_fn, cfg, k)

    def score_fn(model_, tokenizer_, collected, advantages, global_step):
        return score_grpo_batch(model_, tokenizer_, collected, advantages, params, cfg, k)

    default_name = "single_rollout_global_pg" if k == 1 else f"grpo_k{k}"
    run_training_loop(
        model=model,
        tokenizer=tokenizer,
        params=params,
        train_dataset=train_dataset,
        step_fn=None,
        collect_fn=collect_fn,
        score_fn=score_fn,
        cfg=cfg,
        run_name=cfg.logging.run_name or default_name,
        eval_fn=make_eval_fn(cfg) if cfg.data.eval_datasets else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    main(args.config, k=args.k)
