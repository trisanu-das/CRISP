"""On-policy self-distillation (OPSD) baseline step implementation for CRISP.

This module implements the distillation-only baseline used to isolate the effect
of the self-distillation branch without any RL reward term.

Design goals:
- one shared model instance for student and teacher contexts
- on-policy rollout for deciding which samples are wrong
- teacher context uses privileged answer prefix
- student context uses prompt only
- distillation loss is KL(teacher || student)
- optional correctness gating to match the CRISP proposal

Batch format:
    batch = [
        {"prompt": str, "answer": str},
        ...
    ]

This file is meant to be imported by a training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from data.build_dataset import format_student_prompt, format_teacher_prompt
from data.reward import math_verify


# -----------------------------
# Config
# -----------------------------


@dataclass
class OPSDConfig:
    """Configuration for the OPSD baseline."""

    total_steps: int = 1500
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    stop_on_eos: bool = True
    entropy_bonus: float = 0.0
    correctness_gate: bool = True
    correctness_gate_mode: str = "wrong_only"  # wrong_only | always_on | correct_only
    sd_loss_weight: float = 1.0
    distill_on_prefix_only: bool = True
    use_on_policy_rollout: bool = True
    max_prompt_length: Optional[int] = None
    max_answer_length: Optional[int] = None


# -----------------------------
# Utilities
# -----------------------------


def _tokenize(tokenizer, texts: Sequence[str], device: torch.device, max_length: Optional[int] = None) -> Dict[str, Tensor]:
    kwargs = dict(return_tensors="pt", padding=True, truncation=True)
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(list(texts), **kwargs)
    return {k: v.to(device) for k, v in encoded.items()}


def _generate(model, tokenizer, inputs: Mapping[str, Tensor], cfg: OPSDConfig) -> Tensor:
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
        raise ValueError("OPSD requires at least one trainable parameter")
    return params


def _sequence_logprobs(logits: Tensor, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].to(token_log_probs.dtype)
        token_log_probs = token_log_probs * shift_mask
    return token_log_probs.sum(dim=-1)


# -----------------------------
# Reward / gating
# -----------------------------


def compute_rewards(generations: Sequence[str], batch: Sequence[Mapping[str, str]], device: torch.device) -> Tensor:
    rewards = torch.zeros(len(generations), dtype=torch.float32, device=device)
    for i, (pred, item) in enumerate(zip(generations, batch)):
        rewards[i] = float(math_verify(pred, item["answer"]))
    return rewards


def _gate_mask(rewards: Tensor, mode: str) -> Tensor:
    if mode == "wrong_only":
        return rewards <= 0.0
    if mode == "correct_only":
        return rewards > 0.0
    if mode == "always_on":
        return torch.ones_like(rewards, dtype=torch.bool)
    raise ValueError(f"Unknown correctness_gate_mode: {mode}")


# -----------------------------
# Main step
# -----------------------------


def opsd_step(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | OPSDConfig,
    step: int,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Run one OPSD update.

    This is distillation-only: no RL loss term is used.
    """
    if not batch:
        raise ValueError("opsd_step() received an empty batch")

    cfg = config if isinstance(config, OPSDConfig) else OPSDConfig(
        total_steps=int(config.get("total_steps", 1500)),
        max_new_tokens=int(config.get("max_new_tokens", 1024)),
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 1.0)),
        top_k=int(config.get("top_k", 0)),
        do_sample=bool(config.get("do_sample", True)),
        stop_on_eos=bool(config.get("stop_on_eos", True)),
        entropy_bonus=float(config.get("entropy_bonus", 0.0)),
        correctness_gate=bool(config.get("correctness_gate", True)),
        correctness_gate_mode=str(config.get("correctness_gate_mode", "wrong_only")),
        sd_loss_weight=float(config.get("sd_loss_weight", 1.0)),
        distill_on_prefix_only=bool(config.get("distill_on_prefix_only", True)),
        use_on_policy_rollout=bool(config.get("use_on_policy_rollout", True)),
    )

    if device is None:
        device = next(model.parameters()).device

    model.train()
    params = _get_trainable_params(model)
    optimizer.zero_grad(set_to_none=True)

    student_prompts = [format_student_prompt(item["prompt"]) for item in batch]
    teacher_prompts = [format_teacher_prompt(item["prompt"], item["answer"]) for item in batch]

    sd_max_length = None
    if cfg.max_prompt_length is not None or cfg.max_answer_length is not None:
        sd_max_length = (cfg.max_prompt_length or 0) + (cfg.max_answer_length or 0) or None
    student_inputs = _tokenize(tokenizer, student_prompts, device=device, max_length=cfg.max_prompt_length)
    teacher_inputs = _tokenize(tokenizer, teacher_prompts, device=device, max_length=sd_max_length)

    # Optional on-policy rollout to determine which samples are wrong.
    if cfg.use_on_policy_rollout:
        generated_ids = _generate(model, tokenizer, student_inputs, cfg)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        rewards = compute_rewards(generated_texts, batch, device)
    else:
        generated_ids = None
        rewards = torch.zeros(len(batch), dtype=torch.float32, device=device)

    wrong_mask = _gate_mask(rewards, cfg.correctness_gate_mode) if cfg.correctness_gate else torch.ones_like(rewards, dtype=torch.bool)

    # Shared-weight forward passes.
    teacher_out = model(**teacher_inputs)
    student_out = model(**student_inputs)

    teacher_logits = teacher_out.logits
    student_logits = student_out.logits

    if cfg.distill_on_prefix_only:
        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        teacher_logits = teacher_logits[:, :min_len, :]
        student_logits = student_logits[:, :min_len, :]

    if wrong_mask.any():
        l_sd = F.kl_div(
            F.log_softmax(student_logits[wrong_mask], dim=-1),
            F.softmax(teacher_logits[wrong_mask], dim=-1),
            reduction="batchmean",
        )
    else:
        l_sd = torch.zeros((), device=device, requires_grad=True)

    loss = cfg.sd_loss_weight * l_sd
    entropy = _token_entropy(student_out.logits)
    if cfg.entropy_bonus != 0.0:
        loss = loss - cfg.entropy_bonus * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    logs: Dict[str, float] = {
        "reward/mean": float(rewards.mean().item()),
        "reward/std": float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0,
        "loss/opsd": float(l_sd.item()),
        "loss/total": float(loss.item()),
        "policy/entropy": float(entropy.item()),
        "diagnostics/wrong_frac": float(wrong_mask.float().mean().item()),
        "efficiency/tokens_generated": float(generated_ids.numel()) if generated_ids is not None else 0.0,
        "reward/correct_frac": float(rewards.mean().item()),
    }

    if cfg.entropy_bonus != 0.0:
        logs["loss/entropy_bonus"] = float((-cfg.entropy_bonus * entropy).item())

    return logs


# -----------------------------
# Compatibility wrapper
# -----------------------------


def opsd_loss_only(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | OPSDConfig,
    step: int,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """Loss-only wrapper for external trainers."""
    if device is None:
        device = next(model.parameters()).device

    dummy_opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    logs = opsd_step(model, tokenizer, batch, config, step, dummy_opt, device=device)
    return {k: torch.tensor(v, device=device) for k, v in logs.items()}
