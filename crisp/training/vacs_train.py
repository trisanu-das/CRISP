"""VACS-CRISP training entry point using effective-batch normalization."""
from __future__ import annotations

import argparse
from typing import Optional

import torch

from crisp.data.build_dataset import load_train_dataset
from crisp.data.reward import build_reward_fn
from crisp.eval.run_eval import make_eval_fn
from crisp.model.load import attach_or_load_lora, load_model_and_tokenizer
from crisp.training.batch_types import CollectedBatch
from crisp.training.common_loop import run_training_loop
from crisp.training.rollout import generate_rollouts
from crisp.training.teacher_student import build_teacher_student_sequences
from crisp.training.vacs_state import VacsAdaptiveState
from crisp.training.vacs_step import vacs_step_from_arrays
from crisp.utils.config import load_config
from crisp.utils.distributed import all_reduce_scalar, init_distributed
from crisp.utils.seeding import set_seed


def collect_vacs_batch(model, tokenizer, prompts, answers, reward_fn, cfg) -> CollectedBatch:
    model.eval()
    rollouts = generate_rollouts(
        model,
        tokenizer,
        prompts,
        cfg.model.system_prompt,
        max_prompt_length=cfg.rollout.max_prompt_length,
        max_new_tokens=cfg.rollout.max_new_tokens,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
        do_sample=cfg.rollout.do_sample,
        num_return_sequences=1,
    )
    rewards = torch.tensor(
        [reward_fn(r.response_text, answer) for r, answer in zip(rollouts, answers)],
        dtype=torch.float32,
    )
    batch = CollectedBatch(
        list(prompts),
        list(answers),
        rollouts,
        rewards,
        metadata={
            "generations": [
                {"prompt": p, "response_text": r.response_text, "reward": float(reward)}
                for p, r, reward in zip(prompts, rollouts, rewards.tolist())
            ]
        },
    )
    batch.validate()
    return batch


def vacs_score_fn_factory(tokenizer, params, cfg, state):
    def score_fn(model, tokenizer_, collected, advantages, global_step):
        valid = [i for i, rollout in enumerate(collected.rollouts) if len(rollout.response_ids) > 0]
        if not valid:
            raise RuntimeError("Every VACS rollout was empty.")
        prompts = [collected.prompts[i] for i in valid]
        answers = [collected.answers[i] for i in valid]
        rollouts = [collected.rollouts[i] for i in valid]
        rewards = collected.rewards[valid]
        advantages_valid = advantages[valid]

        teacher_seqs, student_seqs = build_teacher_student_sequences(
            prompts,
            answers,
            rollouts,
            tokenizer=tokenizer_,
            sys_prompt=cfg.model.system_prompt,
            hint_template=cfg.data.teacher_hint_template,
        )
        result = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=tokenizer_.pad_token_id,
            params=params,
            rewards=rewards,
            base_lambda=cfg.training.lambda_max,
            adv_eps=cfg.training.adv_eps,
            clip_epsilon=cfg.vacs.get("token_weight_epsilon", cfg.training.clip_epsilon),
            sd_weight=cfg.vacs.sd_weight,
            soft_gate_slope=cfg.vacs.soft_gate_slope,
            reward_threshold=cfg.vacs.reward_threshold,
            credit_mix=cfg.vacs.token_credit_mix,
            use_token_credit=bool(cfg.vacs.token_credit_mix > 0.0),
            use_soft_gate=bool(cfg.vacs.get("use_soft_gate", True)),
            advantages=advantages_valid,
            use_asymmetric_projection=bool(cfg.vacs.get("use_asymmetric_projection", True)),
            state=state,
        )
        metrics = dict(result.metrics)
        metrics.update({
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "response_len_mean": sum(len(r.response_ids) for r in rollouts) / len(rollouts),
            "empty_response_fraction": 1.0 - len(valid) / max(len(collected.rollouts), 1),
            "advantage_abs_mean": float(advantages_valid.abs().mean().item()),
        })
        return result.gradients, metrics

    return score_fn


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
    state = VacsAdaptiveState(
        ema_decay=cfg.vacs.adaptive_mix.ema_decay,
        multiplier_min=cfg.vacs.adaptive_mix.multiplier_min,
        multiplier_max=cfg.vacs.adaptive_mix.multiplier_max,
        disabled=not cfg.vacs.adaptive_mix.enabled,
    )

    def collect_fn(model_, tokenizer_, prompts, answers, global_step):
        return collect_vacs_batch(model_, tokenizer_, prompts, answers, reward_fn, cfg)

    score_fn = vacs_score_fn_factory(tokenizer, params, cfg, state)

    def post_optimizer_step(model_, aggregated_step_metrics):
        if state._has_pending:
            # Every rank must use the same adaptive coefficient on the next
            # optimizer step. Aggregate all micro-batch statistics globally
            # before committing the EMA state.
            rl_sq, sd_sq, cos_value = state.pending_means()
            device = next(model_.parameters()).device
            rl_sq = all_reduce_scalar(rl_sq, device)
            sd_sq = all_reduce_scalar(sd_sq, device)
            cos_value = all_reduce_scalar(cos_value, device)
            state.commit(
                rl_grad_norm_sq=rl_sq,
                sd_grad_norm_sq=sd_sq,
                cos_value=cos_value,
            )

    def checkpoint_state() -> dict:
        return {"vacs_adaptive_state": state.state_dict()}

    def restore_method_state(payload: dict) -> None:
        if payload and "vacs_adaptive_state" in payload:
            state.load_state_dict(payload["vacs_adaptive_state"])

    run_training_loop(
        model=model,
        tokenizer=tokenizer,
        params=params,
        train_dataset=train_dataset,
        step_fn=None,
        collect_fn=collect_fn,
        score_fn=score_fn,
        cfg=cfg,
        run_name=cfg.logging.run_name or "vacs",
        eval_fn=make_eval_fn(cfg) if cfg.data.eval_datasets else None,
        post_optimizer_step=post_optimizer_step,
        checkpoint_state=checkpoint_state,
        restore_method_state=restore_method_state,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    from train_launcher import parse_overrides
    main(args.config, overrides=parse_overrides(args.override))
