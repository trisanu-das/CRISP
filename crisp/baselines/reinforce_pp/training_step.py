"""REINFORCE++-style global-effective-batch policy-gradient baseline."""
from __future__ import annotations

import torch

from crisp.training.batch_types import CollectedBatch
from crisp.training.grad_utils import compute_grads
from crisp.training.logprobs import ScoredSequence, build_scoring_batch, sequence_log_prob
from crisp.training.rollout import generate_rollouts


def collect_reinforce_pp_batch(model, tokenizer, prompts, answers, reward_fn, cfg) -> CollectedBatch:
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
        [reward_fn(r.response_text, a) for r, a in zip(rollouts, answers)],
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


def score_reinforce_pp_batch(model, tokenizer, collected, advantages, params, cfg):
    valid = [i for i, r in enumerate(collected.rollouts) if len(r.response_ids) > 0]
    if not valid:
        raise RuntimeError("Every REINFORCE++ rollout was empty.")
    rollouts = [collected.rollouts[i] for i in valid]
    rewards = collected.rewards[valid]
    advantages = advantages[valid].to(model.device)

    seqs = [
        ScoredSequence(
            ids=r.prompt_ids + r.response_ids,
            response_start=len(r.prompt_ids),
            response_len=len(r.response_ids),
        )
        for r in rollouts
    ]
    input_ids, attention_mask = build_scoring_batch(seqs, tokenizer.pad_token_id)
    token_log_probs, _ = sequence_log_prob(
        model,
        input_ids.to(model.device),
        attention_mask.to(model.device),
        seqs,
    )

    losses = [
        -advantage * logps.mean()
        for advantage, logps in zip(advantages, token_log_probs)
        if logps.numel() > 0
    ]
    if not losses:
        raise RuntimeError("No non-empty REINFORCE++ response remained after validation.")
    loss = torch.stack(losses).mean()
    grads = compute_grads(loss, params)
    return grads, {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "loss_rl": float(loss.item()),
        "rl_grad_norm": float(torch.cat([g.reshape(-1) for g in grads]).norm().item()),
        "response_len_mean": sum(len(r.response_ids) for r in rollouts) / len(rollouts),
        "empty_response_fraction": 1.0 - len(valid) / max(len(collected.rollouts), 1),
        "advantage_abs_mean": float(advantages.abs().mean().item()),
    }


def reinforce_pp_step(model, tokenizer, prompts, answers, reward_fn, params, cfg):
    """Backward-compatible microbatch wrapper."""
    collected = collect_reinforce_pp_batch(model, tokenizer, prompts, answers, reward_fn, cfg)
    rewards = collected.rewards
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + cfg.training.adv_eps)
    return score_reinforce_pp_batch(model, tokenizer, collected, advantages, params, cfg)
