"""REINFORCE++ baseline step implementation for CRISP.

This module implements a critic-free policy-gradient baseline with global reward
normalization. It is designed to compare directly against CRISP and GRPO.

Core properties:
- no critic network
- single on-policy rollout per prompt by default
- global normalization across the batch
- optional ratio clipping interface for compatibility with PPO-style scaffolding

The implementation deliberately keeps the same batch contract as the other
training code in this repo:
- batch items are dicts with keys: prompt, answer
- rewards are computed with the same math verifier
- the same student prompt template is used
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from data.build_dataset import format_student_prompt
from data.reward import math_verify


# -----------------------------
# Config
# -----------------------------


@dataclass
class ReinforcePPConfig:
    """Configuration for the REINFORCE++ baseline."""

    total_steps: int = 1500
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    stop_on_eos: bool = True
    clip_ratio: float = 0.2
    advantage_eps: float = 1e-8
    reward_centering: bool = True
    reward_scaling: str = "standardize"  # standardize | center_only | none
    entropy_bonus: float = 0.0
    max_prompt_length: Optional[int] = None
    top_k: int = 0
    repetition_penalty: float = 1.15   # was implicitly 1.0 (no-op); this was the actual bug
    do_sample: bool = True


# -----------------------------
# Utilities
# -----------------------------


def _tokenize(tokenizer, texts: Sequence[str], device: torch.device, max_length: Optional[int] = None) -> Dict[str, Tensor]:
    kwargs = dict(return_tensors="pt", padding=True, truncation=True)
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(list(texts), **kwargs)
    return {k: v.to(device) for k, v in encoded.items()}


def _generate(model, tokenizer, inputs: Mapping[str, Tensor], cfg: ReinforcePPConfig) -> Tensor:
    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        return_dict_in_generate=True,
        output_scores=False,
    )
    if cfg.stop_on_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return out.sequences[:, prompt_len:]


def _sequence_logprobs(logits: Tensor, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].to(token_log_probs.dtype)
        token_log_probs = token_log_probs * shift_mask
    return token_log_probs.sum(dim=-1)


def _token_entropy(logits: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def _flatten_grads(grads: Sequence[Optional[Tensor]], params: Sequence[torch.nn.Parameter]) -> Tensor:
    flat: List[Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat) if flat else torch.zeros(0)


def _write_flat_grad(model: torch.nn.Module, flat_grad: Tensor) -> None:
    offset = 0
    for p in [p for p in model.parameters() if p.requires_grad]:
        n = p.numel()
        chunk = flat_grad[offset : offset + n].view_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def _get_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("REINFORCE++ requires at least one trainable parameter")
    return params


# -----------------------------
# Reward / advantage
# -----------------------------


def compute_rewards(batch_rollouts, batch, device):
    rewards = torch.zeros(len(batch_rollouts), dtype=torch.float32, device=device)
    for i, (pred, item) in enumerate(zip(batch_rollouts, batch)):
        rewards[i] = float(math_verify(pred, item["answer"]))
    return rewards


def compute_advantages(rewards: Tensor, cfg: ReinforcePPConfig) -> Tensor:
    if rewards.ndim != 1:
        raise ValueError(f"Expected 1D rewards, got shape {tuple(rewards.shape)}")

    if cfg.reward_scaling == "standardize":
        mean = rewards.mean()
        std = rewards.std(unbiased=False).clamp_min(cfg.advantage_eps)
        adv = (rewards - mean) / std if cfg.reward_centering else rewards / std
    elif cfg.reward_scaling == "center_only":
        mean = rewards.mean()
        adv = rewards - mean if cfg.reward_centering else rewards
    elif cfg.reward_scaling == "none":
        adv = rewards
    else:
        raise ValueError(f"Unknown reward_scaling: {cfg.reward_scaling}")

    return adv


# -----------------------------
# Main step
# -----------------------------


def reinforce_pp_step(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | ReinforcePPConfig,
    step: int,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Run one critic-free policy gradient update with global reward normalization."""
    if not batch:
        raise ValueError("reinforce_pp_step() received an empty batch")

    cfg = config if isinstance(config, ReinforcePPConfig) else ReinforcePPConfig(
        total_steps=int(config.get("total_steps", 1500)),
        max_new_tokens=int(config.get("max_new_tokens", 1024)),
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 1.0)),
        top_k=int(config.get("top_k", 0)),
        do_sample=bool(config.get("do_sample", True)),
        stop_on_eos=bool(config.get("stop_on_eos", True)),
        clip_ratio=float(config.get("clip_ratio", 0.2)),
        advantage_eps=float(config.get("advantage_eps", 1e-8)),
        reward_centering=bool(config.get("reward_centering", True)),
        reward_scaling=str(config.get("reward_scaling", "standardize")),
        entropy_bonus=float(config.get("entropy_bonus", 0.0)),
    )

    if device is None:
        device = next(model.parameters()).device

    model.train()
    params = _get_trainable_params(model)
    optimizer.zero_grad(set_to_none=True)

    prompts = [format_student_prompt(item["prompt"]) for item in batch]
    inputs = _tokenize(tokenizer, prompts, device=device, max_length=cfg.max_prompt_length)
    generated_ids = _generate(model, tokenizer, inputs, cfg)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(f"Sample generated text: {generated_texts[0]}")
    print(f"Ground truth: {batch[0]['answer']}")

    rewards = compute_rewards(generated_texts, batch, device=device)
    advantages = compute_advantages(rewards, cfg)

    rollout_input_ids = torch.cat([inputs["input_ids"], generated_ids], dim=1)
    if "attention_mask" in inputs:
        rollout_attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones_like(generated_ids, device=device)],
            dim=1,
        )
    else:
        rollout_attention_mask = None

    outputs = model(input_ids=rollout_input_ids, attention_mask=rollout_attention_mask)
    seq_logprob = _sequence_logprobs(outputs.logits, rollout_input_ids, rollout_attention_mask)

    # Bounded surrogate scaling, but no critic and no group-relative normalization.
    clipped_adv = advantages.clamp(min=-cfg.clip_ratio, max=cfg.clip_ratio)
    loss_pg = -(clipped_adv * seq_logprob).mean()

    loss = loss_pg
    entropy = _token_entropy(outputs.logits)
    if cfg.entropy_bonus != 0.0:
        loss = loss - cfg.entropy_bonus * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    logs: Dict[str, float] = {
        "reward/mean": float(rewards.mean().item()),
        "reward/std": float(rewards.std(unbiased=False).item()),
        "loss/reinforce_pp": float(loss_pg.item()),
        "advantage/variance": float(advantages.var(unbiased=False).item()),
        "policy/entropy": float(entropy.item()),
        "efficiency/tokens_generated": float(generated_ids.numel()),
        "reward/correct_frac": float(rewards.mean().item()),
        "diagnostics/reward_min": float(rewards.min().item()),
        "diagnostics/reward_max": float(rewards.max().item()),
    }
    if cfg.entropy_bonus != 0.0:
        logs["loss/entropy_bonus"] = float((-cfg.entropy_bonus * entropy).item())

    return logs


# -----------------------------
# Optional compatibility wrapper
# -----------------------------


def reinforce_pp_loss_only(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | ReinforcePPConfig,
    step: int,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """Loss-only compatibility wrapper for external trainers."""
    if device is None:
        device = next(model.parameters()).device

    dummy_opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    logs = reinforce_pp_step(model, tokenizer, batch, config, step, dummy_opt, device=device)
    return {k: torch.tensor(v, device=device) for k, v in logs.items()}
