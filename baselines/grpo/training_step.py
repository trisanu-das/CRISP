"""GRPO baselines for CRISP.

This module implements critic-free group relative policy optimization in a form
that is easy to compare against CRISP.

Included variants:
- GRPO(k=1): degenerate one-sample group baseline
- GRPO(k=8): stronger multi-sample baseline

The implementation deliberately stays close to the CRISP data/model contract:
- batch items are dicts with keys: prompt, answer
- the same student prompt template is used
- rewards come from the same verifiable reward function

This file is meant to be imported by the training loop, not executed directly.
"""

from __future__ import annotations

import math
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
class GRPOConfig:
    """Config for GRPO baseline steps."""

    k: int = 8
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
    reward_scaling: str = "group_standardize"  # group_standardize | group_center_only | none


# -----------------------------
# Utilities
# -----------------------------


def _config_value(config: Mapping[str, Any] | GRPOConfig, key: str, default: Any) -> Any:
    if isinstance(config, GRPOConfig):
        return getattr(config, key, default)
    return config.get(key, default)


def _get_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("GRPO requires at least one trainable parameter")
    return params


def _tokenize(tokenizer, texts: Sequence[str], device: torch.device) -> Dict[str, Tensor]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def _generate(model, tokenizer, inputs: Mapping[str, Tensor], cfg: GRPOConfig) -> Tensor:
    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
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


def _zero_like_params(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# -----------------------------
# Reward / advantage
# -----------------------------


def compute_reward_matrix(
    generations: Sequence[Sequence[str]],
    batch: Sequence[Mapping[str, str]],
) -> Tensor:
    """Return rewards shaped [k, batch_size]."""
    k = len(generations)
    n = len(batch)
    rewards = torch.zeros(k, n, dtype=torch.float32)
    for i in range(k):
        for j in range(n):
            rewards[i, j] = float(math_verify(generations[i][j], batch[j]["answer"]))
    return rewards


def compute_group_advantages(rewards: Tensor, cfg: GRPOConfig) -> Tensor:
    """Compute within-prompt group-relative advantages.

    rewards: [k, n]
    returns:  [k, n]
    """
    if rewards.ndim != 2:
        raise ValueError(f"Expected rewards to have shape [k, n], got {tuple(rewards.shape)}")

    if cfg.reward_scaling == "group_standardize":
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True, unbiased=False).clamp_min(cfg.advantage_eps)
        adv = (rewards - mean) / std if cfg.reward_centering else rewards / std
    elif cfg.reward_scaling == "group_center_only":
        mean = rewards.mean(dim=0, keepdim=True)
        adv = rewards - mean if cfg.reward_centering else rewards
    elif cfg.reward_scaling == "none":
        adv = rewards
    else:
        raise ValueError(f"Unknown reward_scaling: {cfg.reward_scaling}")

    return adv


# -----------------------------
# Main GRPO step
# -----------------------------


def grpo_step(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | GRPOConfig,
    step: int,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Run one GRPO update.

    This is the core baseline used for both k=1 and k=8.
    """
    if not batch:
        raise ValueError("grpo_step() received an empty batch")

    cfg = config if isinstance(config, GRPOConfig) else GRPOConfig(
        k=int(config.get("k", 8)),
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
        reward_scaling=str(config.get("reward_scaling", "group_standardize")),
    )

    if device is None:
        device = next(model.parameters()).device

    model.train()
    params = _get_trainable_params(model)
    optimizer.zero_grad(set_to_none=True)

    prompts = [format_student_prompt(item["prompt"]) for item in batch]

    # Repeat prompts k times so we can generate k on-policy rollouts per prompt.
    repeated_prompts: List[str] = []
    for _ in range(cfg.k):
        repeated_prompts.extend(prompts)

    inputs = _tokenize(tokenizer, repeated_prompts, device=device)
    generated_ids = _generate(model, tokenizer, inputs, cfg)

    # Decode generations into [k, n] structure.
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    n = len(batch)
    generations: List[List[str]] = []
    for i in range(cfg.k):
        start = i * n
        end = (i + 1) * n
        generations.append(decoded[start:end])

    rewards = compute_reward_matrix(generations, batch)  # [k, n]
    advantages = compute_group_advantages(rewards, cfg)  # [k, n]

    # Build a single flat set of rollout tokens and corresponding per-sample advantages.
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

    # seq_logprob is [k*n]; reshape to [k, n] to align with advantages.
    seq_logprob = seq_logprob.view(cfg.k, n)

    # PPO-style bounded surrogate scaling. Since we are critic-free, the current
    # policy ratio is implicitly 1; clipping is used only to avoid extreme updates.
    clipped_adv = advantages.clamp(min=-cfg.clip_ratio, max=cfg.clip_ratio)
    loss = -(clipped_adv * seq_logprob).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    reward_mean = float(rewards.mean().item())
    reward_std = float(rewards.std(unbiased=False).item())
    adv_var = float(advantages.var(unbiased=False).item())

    return {
        "reward/mean": reward_mean,
        "reward/std": reward_std,
        "loss/grpo": float(loss.item()),
        "advantage/variance": adv_var,
        "efficiency/k": float(cfg.k),
        "efficiency/tokens_generated": float(generated_ids.numel()),
        "reward/correct_frac": reward_mean,
        "diagnostics/reward_min": float(rewards.min().item()),
        "diagnostics/reward_max": float(rewards.max().item()),
    }


# -----------------------------
# Baseline convenience wrappers
# -----------------------------


def grpo_step_k1(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | GRPOConfig,
    step: int,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    cfg = config if isinstance(config, GRPOConfig) else GRPOConfig(**{**dict(config), "k": 1})
    cfg.k = 1
    return grpo_step(model, tokenizer, batch, cfg, step, optimizer, device=device)


def grpo_step_k8(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | GRPOConfig,
    step: int,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    cfg = config if isinstance(config, GRPOConfig) else GRPOConfig(**{**dict(config), "k": 8})
    cfg.k = 8
    return grpo_step(model, tokenizer, batch, cfg, step, optimizer, device=device)
