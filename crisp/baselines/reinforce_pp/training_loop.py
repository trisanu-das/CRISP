"""REINFORCE++ training entrypoint."""
from __future__ import annotations

import argparse
from typing import Optional

from crisp.baselines.reinforce_pp.training_step import (
    collect_reinforce_pp_batch,
    score_reinforce_pp_batch,
)
from crisp.data.build_dataset import load_train_dataset
from crisp.data.reward import build_reward_fn
from crisp.eval.run_eval import make_eval_fn
from crisp.model.load import attach_or_load_lora, load_model_and_tokenizer
from crisp.training.common_loop import run_training_loop
from crisp.utils.config import load_config
from crisp.utils.distributed import init_distributed
from crisp.utils.seeding import set_seed


def main(config_path: str, overrides: Optional[dict] = None) -> None:
    cfg = load_config(config_path, overrides)
    init_distributed()
    set_seed(cfg.seed)
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    initial_adapter = cfg.model.get("initial_adapter") or cfg.training.get("resume_from")
    model, _ = attach_or_load_lora(model, cfg.lora, initial_adapter=initial_adapter)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")

    train_dataset = load_train_dataset(
        cfg.data.train_dataset,
        split=cfg.data.train_split,
        train_subset_size=cfg.data.train_subset_size,
    )
    reward_fn = build_reward_fn(cfg.reward)

    def collect_fn(model_, tokenizer_, prompts, answers, global_step):
        return collect_reinforce_pp_batch(model_, tokenizer_, prompts, answers, reward_fn, cfg)

    def score_fn(model_, tokenizer_, collected, advantages, global_step):
        return score_reinforce_pp_batch(model_, tokenizer_, collected, advantages, params, cfg)

    run_training_loop(
        model=model,
        tokenizer=tokenizer,
        params=params,
        train_dataset=train_dataset,
        step_fn=None,
        collect_fn=collect_fn,
        score_fn=score_fn,
        cfg=cfg,
        run_name=cfg.logging.run_name or "reinforce_pp",
        eval_fn=make_eval_fn(cfg) if cfg.data.eval_datasets else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
