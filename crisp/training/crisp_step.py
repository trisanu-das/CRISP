"""CRISP rollout collection and scoring.

The paper-correct path is two-stage: ``collect_crisp_batch`` samples and
scores rewards without autograd; ``score_crisp_batch`` receives advantages
normalized over the full effective batch from ``common_loop``.
"""
from __future__ import annotations

import torch

from crisp.training.batch_types import CollectedBatch
from crisp.training.grad_utils import compute_grads
from crisp.training.gradient_surgery import flatten_grads
from crisp.training.logprobs import forward_kl_teacher_student
from crisp.training.pcgrad import PCGradResult, pc_grad_combine
from crisp.training.rollout import generate_rollouts
from crisp.training.schedules import lambda_schedule
from crisp.training.teacher_student import (
    build_teacher_student_sequences,
    score_teacher_student,
)


def _valid_rollout_indices(rollouts) -> list[int]:
    return [i for i, rollout in enumerate(rollouts) if len(rollout.response_ids) > 0]


def collect_crisp_batch(
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    reward_fn,
    cfg,
) -> CollectedBatch:
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
    collected = CollectedBatch(
        prompts=list(prompts),
        answers=list(answers),
        rollouts=rollouts,
        rewards=rewards,
        metadata={
            "generations": [
                {"prompt": p, "response_text": r.response_text, "reward": float(reward)}
                for p, r, reward in zip(prompts, rollouts, rewards.tolist())
            ]
        },
    )
    collected.validate()
    return collected


def _naive_combine(loss_rl, loss_sd, params, lam: float, eps: float) -> PCGradResult:
    needs_sd = loss_sd is not None and loss_sd.requires_grad
    g_rl = compute_grads(loss_rl, params, retain_graph=needs_sd)
    g_sd = (
        compute_grads(loss_sd, params, retain_graph=False)
        if needs_sd
        else [torch.zeros_like(p) for p in params]
    )
    flat_rl = flatten_grads(g_rl)
    flat_sd = flatten_grads(g_sd)
    rl_norm = flat_rl.norm()
    sd_norm = flat_sd.norm()
    dot = torch.dot(flat_rl, flat_sd)
    cos = 0.0
    if float(rl_norm.item()) > 0.0 and float(sd_norm.item()) > 0.0:
        cos = float((dot / (rl_norm * sd_norm + eps)).item())
    return PCGradResult(
        combined=[a + lam * b for a, b in zip(g_rl, g_sd)],
        cos_sim=cos,
        conflicted=bool((dot < 0).item()),
        rl_grad_norm=float(rl_norm.item()),
        sd_grad_norm=float(sd_norm.item()),
    )


def score_crisp_batch(
    model,
    tokenizer,
    collected: CollectedBatch,
    advantages: torch.Tensor,
    params: list[torch.nn.Parameter],
    global_step: int,
    cfg,
) -> tuple[list[torch.Tensor], dict]:
    valid = _valid_rollout_indices(collected.rollouts)
    if not valid:
        raise RuntimeError("Every CRISP rollout was empty; no response tokens can be scored.")

    prompts = [collected.prompts[i] for i in valid]
    answers = [collected.answers[i] for i in valid]
    rollouts = [collected.rollouts[i] for i in valid]
    rewards = collected.rewards[valid]
    advantages = advantages[valid].to(model.device)

    teacher_seqs, student_seqs = build_teacher_student_sequences(
        prompts,
        answers,
        rollouts,
        tokenizer=tokenizer,
        sys_prompt=cfg.model.system_prompt,
        hint_template=cfg.data.teacher_hint_template,
    )
    scored = score_teacher_student(
        model=model,
        teacher_sequences=teacher_seqs,
        student_sequences=student_seqs,
        pad_id=tokenizer.pad_token_id,
    )

    # Plain single-pass policy gradient. A PPO ratio built from
    # ``new_lp - new_lp.detach()`` is numerically one and cannot clip.
    per_example = []
    for advantage, token_logps in zip(advantages, scored.student_token_logps):
        if token_logps.numel() == 0:
            continue
        per_example.append(-advantage * token_logps.mean())
    if not per_example:
        raise RuntimeError("No non-empty CRISP response remained after validation.")
    loss_rl = torch.stack(per_example).mean()

    kl_per_example = forward_kl_teacher_student(
        scored.teacher_row_log_probs,
        scored.student_row_log_probs,
    )
    if cfg.training.get("correctness_gate", True):
        gate = (rewards.to(model.device) < 1.0).float()
    else:
        gate = torch.ones_like(rewards, device=model.device)
    if bool(gate.sum().item() > 0):
        gated = torch.stack([kl * weight for kl, weight in zip(kl_per_example, gate)])
        loss_sd = gated.sum() / gate.sum().clamp(min=1.0)
    else:
        loss_sd = None

    lam = lambda_schedule(
        global_step,
        cfg.training.total_steps,
        cfg.training.lambda_max,
        mode=cfg.training.get("lambda_schedule", "cosine"),
    )
    if cfg.training.get("use_pcgrad", True):
        result = pc_grad_combine(
            loss_rl,
            loss_sd,
            params,
            lam,
            eps=cfg.training.pcgrad_eps,
        )
    else:
        result = _naive_combine(loss_rl, loss_sd, params, lam, cfg.training.pcgrad_eps)

    metrics = {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "loss_rl": float(loss_rl.item()),
        "loss_sd": float(loss_sd.item()) if loss_sd is not None else 0.0,
        "lambda": float(lam),
        "cos_g_rl_g_sd": result.cos_sim,
        "grad_conflicted": float(result.conflicted),
        "rl_grad_norm": result.rl_grad_norm,
        "sd_grad_norm": result.sd_grad_norm,
        "pct_gated_correct": float(1.0 - gate.mean().item()),
        "response_len_mean": sum(len(r.response_ids) for r in rollouts) / len(rollouts),
        "empty_response_fraction": 1.0 - len(valid) / max(len(collected.rollouts), 1),
        "advantage_abs_mean": float(advantages.abs().mean().item()),
    }
    return result.combined, metrics


def crisp_step(
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    reward_fn,
    params: list[torch.nn.Parameter],
    global_step: int,
    cfg,
) -> tuple[list[torch.Tensor], dict]:
    """Backward-compatible local-microbatch wrapper used by unit tests/tools."""
    collected = collect_crisp_batch(model, tokenizer, prompts, answers, reward_fn, cfg)
    rewards = collected.rewards
    advantages = (rewards - rewards.mean()) / (
        rewards.std(unbiased=False) + cfg.training.adv_eps
    )
    return score_crisp_batch(
        model, tokenizer, collected, advantages, params, global_step, cfg
    )
