"""GRPO scoring with explicit group-relative advantages for k>=2.

``k=1`` cannot define a non-trivial group-relative advantage. For backwards
compatibility it is retained as a *single-rollout global-PG control* and uses
the effective-batch advantages supplied by ``common_loop``. Paper tables
should label it accordingly rather than calling it genuine GRPO.
"""
from __future__ import annotations

import torch

from crisp.training.batch_types import CollectedBatch
from crisp.training.grad_utils import compute_grads
from crisp.training.logprobs import ScoredSequence, build_scoring_batch, sequence_log_prob
from crisp.training.rollout import generate_rollouts


def collect_grpo_batch(model, tokenizer, prompts, answers, reward_fn, cfg, k: int) -> CollectedBatch:
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
        num_return_sequences=k,
    )
    expanded_answers = [answer for answer in answers for _ in range(k)]
    expanded_prompts = [prompt for prompt in prompts for _ in range(k)]
    rewards = torch.tensor(
        [reward_fn(r.response_text, a) for r, a in zip(rollouts, expanded_answers)],
        dtype=torch.float32,
    )
    batch = CollectedBatch(
        expanded_prompts,
        expanded_answers,
        rollouts,
        rewards,
        metadata={
            "k": int(k),
            "num_prompts": len(prompts),
            "generations": [
                {"prompt": p, "response_text": r.response_text, "reward": float(reward)}
                for p, r, reward in zip(expanded_prompts, rollouts, rewards.tolist())
            ],
        },
    )
    batch.validate()
    expected = len(prompts) * k
    actual = int(batch.rewards.numel())
    if actual != expected:
        raise RuntimeError(
            f"GRPO rollout-count mismatch: prompts={len(prompts)}, k={k}, "
            f"expected_rewards={expected}, actual_rewards={actual}"
        )
    print(
        f"[grpo-collect] prompts={len(prompts)} k={k} "
        f"rollouts={len(rollouts)} rewards={actual}"
    )
    return batch


def score_grpo_batch(model, tokenizer, collected, effective_advantages, params, cfg, k: int):
    valid = [i for i, r in enumerate(collected.rollouts) if len(r.response_ids) > 0]
    if not valid:
        raise RuntimeError("Every GRPO rollout was empty.")
    rewards = collected.rewards
    num_prompts = int(collected.metadata["num_prompts"])
    if rewards.numel() != num_prompts * k:
        raise ValueError("GRPO reward count is not num_prompts * k.")
    if k >= 2:
        grouped = rewards.view(num_prompts, k)
        group_mean = grouped.mean(dim=1, keepdim=True)
        group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = ((grouped - group_mean) / (group_std + cfg.training.adv_eps)).reshape(-1)
    else:
        advantages = effective_advantages
    advantages = advantages.to(model.device)

    seqs = [
        ScoredSequence(
            ids=r.prompt_ids + r.response_ids,
            response_start=len(r.prompt_ids),
            response_len=len(r.response_ids),
        )
        for r in collected.rollouts
    ]

    chunk_size = int(cfg.training.get("grpo_score_chunk_size", k))
    if chunk_size < 1:
        raise ValueError("training.grpo_score_chunk_size must be >= 1")

    total = len(seqs)
    accum_grads = [torch.zeros_like(p) for p in params]
    loss_value = 0.0

    for lo in range(0, total, chunk_size):
        hi = min(lo + chunk_size, total)
        chunk = seqs[lo:hi]
        input_ids, attention_mask = build_scoring_batch(chunk, tokenizer.pad_token_id)
        token_log_probs, _ = sequence_log_prob(
            model, input_ids.to(model.device), attention_mask.to(model.device), chunk
        )

        chunk_losses = []
        for advantage, logps in zip(advantages[lo:hi], token_log_probs):
            if logps.numel() == 0:
                chunk_losses.append(
                    torch.zeros((), device=model.device, dtype=advantages.dtype)
                )
            else:
                chunk_losses.append(-advantage * logps.mean())

        chunk_loss = torch.stack(chunk_losses).sum() / total
        chunk_grads = compute_grads(chunk_loss, params)

        for acc, grad in zip(accum_grads, chunk_grads):
            acc.add_(grad.detach())

        loss_value += float(chunk_loss.detach().item())
        del input_ids, attention_mask, token_log_probs, chunk_losses, chunk_loss, chunk_grads

    flat = torch.cat([g.reshape(-1) for g in accum_grads])
    return accum_grads, {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "loss_rl": loss_value,
        "rl_grad_norm": float(flat.norm().item()),
        "k": float(k),
        "grpo_is_group_relative": float(k >= 2),
        "grpo_score_chunk_size": float(chunk_size),
        "response_len_mean": sum(len(r.response_ids) for r in collected.rollouts) / len(collected.rollouts),
        "empty_response_fraction": 1.0 - len(valid) / max(len(collected.rollouts), 1),
        "advantage_abs_mean": float(advantages.abs().mean().item()),
    }

def grpo_step(model, tokenizer, prompts, answers, reward_fn, params, cfg, k: int):
    collected = collect_grpo_batch(model, tokenizer, prompts, answers, reward_fn, cfg, k)
    rewards = collected.rewards
    effective = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + cfg.training.adv_eps)
    return score_grpo_batch(model, tokenizer, collected, effective, params, cfg, k)
