"""CIBO-CRISP v2 training entry point."""
from __future__ import annotations

import argparse
from typing import Optional

import torch

from crisp.data.build_dataset import load_train_dataset
from crisp.data.reward import build_reward_fn
from crisp.eval.run_eval import make_eval_fn
from crisp.model.load import attach_or_load_lora, load_model_and_tokenizer
from crisp.training.batch_types import CollectedBatch
from crisp.training.beta_controller import BetaController
from crisp.training.cibo_v2_step import cibo_v2_step_from_arrays
from crisp.training.common_loop import run_training_loop
from crisp.training.ema_adapter import EmaAdapterState
from crisp.training.rollout import generate_rollouts
from crisp.training.teacher_student import build_teacher_student_sequences
from crisp.utils.config import load_config
from crisp.utils.distributed import all_reduce_scalar, init_distributed
from crisp.utils.seeding import set_seed


def collect_cibo_batch(model, tokenizer, prompts, answers, reward_fn, cfg) -> CollectedBatch:
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


def cibo_score_fn_factory(tokenizer, params, cfg, ema, controller):
    def score_fn(model, tokenizer_, collected, advantages, global_step):
        valid = [i for i, rollout in enumerate(collected.rollouts) if len(rollout.response_ids) > 0]
        if not valid:
            raise RuntimeError("Every CIBO rollout was empty.")
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
        gradients, metrics = cibo_v2_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=tokenizer_.pad_token_id,
            params=params,
            rewards=rewards,
            ema=ema,
            beta_controller=controller,
            credit_lambda=cfg.cibo_v2.credit_lambda,
            soft_gate_slope=cfg.cibo_v2.soft_gate_slope,
            reward_threshold=cfg.cibo_v2.reward_threshold,
            anchor_alpha=cfg.cibo_v2.anchor_alpha,
            use_anchor=cfg.cibo_v2.anchor_alpha > 0.0,
            advantages=advantages_valid,
            ib_reduction=cfg.cibo_v2.ib_reduction,
            use_soft_gate=bool(cfg.cibo_v2.get("use_soft_gate", True)),
        )
        metrics = dict(metrics)
        metrics.update({
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "response_len_mean": sum(len(r.response_ids) for r in rollouts) / len(rollouts),
            "empty_response_fraction": 1.0 - len(valid) / max(len(collected.rollouts), 1),
            "advantage_abs_mean": float(advantages_valid.abs().mean().item()),
        })
        return gradients, metrics

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
    ema = EmaAdapterState(params, decay=cfg.cibo_v2.anchor_ema_decay)
    controller = BetaController(
        beta=cfg.cibo_v2.beta,
        beta_mode=cfg.cibo_v2.beta_mode,
        beta_min=cfg.cibo_v2.beta_min,
        beta_max=cfg.cibo_v2.beta_max,
        beta_step_size=cfg.cibo_v2.beta_step_size,
        beta_ema_decay=cfg.cibo_v2.beta_ema_decay,
        beta_reference_warmup_steps=cfg.cibo_v2.beta_reference_warmup_steps,
        beta_denominator_epsilon=cfg.cibo_v2.beta_denominator_epsilon,
    )

    def collect_fn(model_, tokenizer_, prompts, answers, global_step):
        return collect_cibo_batch(model_, tokenizer_, prompts, answers, reward_fn, cfg)

    score_fn = cibo_score_fn_factory(tokenizer, params, cfg, ema, controller)

    def post_optimizer_step(model_, aggregated):
        # One globally averaged observation per optimizer update; controller
        # behavior no longer changes when grad_accum_steps changes.
        reward = all_reduce_scalar(aggregated.get("reward_mean", 0.0), model_.device)
        kl_value = all_reduce_scalar(aggregated.get("cibo/loss_ib", 0.0), model_.device)
        controller.observe(reward=reward, kl_value=kl_value)
        controller.step()
        ema.update()

    def checkpoint_state() -> dict:
        return {
            "cibo_v2_ema": ema.state_dict(),
            "cibo_v2_beta_controller": controller.state_dict(),
        }

    def restore_method_state(payload: dict) -> None:
        if not payload:
            return
        if "cibo_v2_ema" in payload:
            ema.load_state_dict(payload["cibo_v2_ema"])
        if "cibo_v2_beta_controller" in payload:
            controller.load_state_dict(payload["cibo_v2_beta_controller"])

    run_training_loop(
        model=model,
        tokenizer=tokenizer,
        params=params,
        train_dataset=train_dataset,
        step_fn=None,
        collect_fn=collect_fn,
        score_fn=score_fn,
        cfg=cfg,
        run_name=cfg.logging.run_name or "cibo-v2",
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
