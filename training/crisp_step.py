"""CRISP training step.

This module implements the unified RL + self-distillation update described in the
CRISP plan:

    L(theta) = L_RL(theta) + lambda(t) * L_SD(theta) + entropy_bonus

Design goals:
- Single shared model instance for both student and teacher contexts.
- On-policy rollout for the RL branch.
- Correctness-gated self-distillation.
- PC-Grad deconfliction between RL and SD gradients.
- Minimal external assumptions so the file can be adapted to veRL/OpenRLHF later.

Expected batch format:
    batch = [
        {"prompt": str, "answer": str},
        ...
    ]

Returned logs are plain Python scalars for easy WandB / JSON logging.

Notes:
- This file assumes a causal LM from HuggingFace Transformers.
- The teacher and student share weights; only the input context differs.
- For stable gradient computation, this implementation recomputes forward passes
  with gradient tracking after generation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


STUDENT_TEMPLATE = (
    "Solve the following problem step by step.\n"
    "Put your final answer inside \\boxed{}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)

TEACHER_TEMPLATE = (
    "Solve the following problem step by step.\n"
    "Put your final answer inside \\boxed{}.\n\n"
    "Problem: {problem}\n\n"
    "The correct answer is: {answer}\n\n"
    "Now write a complete step-by-step solution:"
)


@dataclass
class CrispStepConfig:
    total_steps: int = 1500
    lambda_max: float = 1.0
    lambda_schedule: str = "cosine"
    lambda_floor: float = 0.0
    correctness_gate: bool = True
    correctness_gate_mode: str = "wrong_only"
    pc_grad: bool = True
    entropy_bonus: float = 0.01
    clip_ratio: float = 0.2
    advantage_eps: float = 1e-8
    reward_centering: bool = True
    reward_scaling: str = "standardize"
    teacher_student_shared_weights: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    stop_on_eos: bool = True


# -----------------------------
# Reward / parsing utilities
# -----------------------------

_FINAL_ANSWER_PATTERNS = [
    r"\\boxed\{([^}]*)\}",
    r"final answer(?: is)?[:\s]*([^\n]+)",
    r"answer(?: is)?[:\s]*([^\n]+)",
    r"therefore[:\s]*([^\n]+)",
]


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    return text


def extract_candidate_answer(text: str) -> str:
    """Extract a likely final answer string from model output.

    This is intentionally lightweight. If you have a more formal math verifier,
    inject it by replacing reward_fn().
    """
    for pat in _FINAL_ANSWER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def reward_fn(generated: str, ground_truth: str) -> float:
    """Binary reward.

    Preferred path: math verification libraries can be plugged in later.
    Fallback here is a normalized exact-match comparison over extracted answers.
    """
    pred = _normalize_text(extract_candidate_answer(generated))
    truth = _normalize_text(extract_candidate_answer(ground_truth))
    if not pred or not truth:
        return 0.0
    return 1.0 if pred == truth else 0.0


# -----------------------------
# Lambda schedule
# -----------------------------


def schedule_lambda(step: int, total_steps: int, lambda_max: float, mode: str = "cosine", floor: float = 0.0) -> float:
    """Compute the coupling coefficient lambda(t)."""
    if lambda_max <= 0.0:
        return 0.0

    t = max(0, min(step, total_steps))
    if mode == "cosine":
        lam = lambda_max * math.cos(math.pi * t / (2.0 * max(total_steps, 1)))
    elif mode == "linear":
        lam = lambda_max * (1.0 - t / max(total_steps, 1))
    elif mode == "constant":
        lam = lambda_max
    else:
        raise ValueError(f"Unknown lambda schedule: {mode}")

    return max(float(lam), float(floor))


# -----------------------------
# Parameter / gradient helpers
# -----------------------------


def get_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_tensors(tensors: Sequence[Optional[Tensor]], params: Sequence[torch.nn.Parameter]) -> Tensor:
    flat: List[Tensor] = []
    for t, p in zip(tensors, params):
        if t is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(t.reshape(-1))
    if not flat:
        return torch.zeros(0)
    return torch.cat(flat)


def unflatten_to_grads(flat: Tensor, params: Sequence[torch.nn.Parameter]) -> None:
    """Write a flat gradient vector back into model.param.grad."""
    offset = 0
    for p in params:
        numel = p.numel()
        chunk = flat[offset : offset + numel].view_as(p)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += numel


def pc_grad_pair(g_rl: Tensor, g_sd: Tensor, eps: float = 1e-8) -> Tensor:
    """PC-Grad deconfliction for two gradients.

    If gradients conflict (dot < 0), project each gradient onto the normal
    plane of the other before summation.
    """
    if g_rl.numel() == 0:
        return g_sd
    if g_sd.numel() == 0:
        return g_rl

    g1 = g_rl.clone()
    g2 = g_sd.clone()
    dot12 = torch.dot(g1, g2)

    if dot12 < 0:
        # Project g1 away from g2.
        denom2 = torch.dot(g2, g2).clamp_min(eps)
        g1 = g1 - (dot12 / denom2) * g2

        # Recompute with updated g1, then project g2 away from g1.
        dot21 = torch.dot(g2, g1)
        denom1 = torch.dot(g1, g1).clamp_min(eps)
        g2 = g2 - (dot21 / denom1) * g1

    return g1 + g2


# -----------------------------
# Log-prob / entropy helpers
# -----------------------------


def _shift_for_causal_lm(logits: Tensor, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """Shift logits and labels for next-token prediction."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return shift_logits, shift_labels


def sequence_logprobs(logits: Tensor, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    """Per-sequence log-prob sum under a causal LM."""
    shift_logits, shift_labels = _shift_for_causal_lm(logits, input_ids)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].to(token_log_probs.dtype)
        token_log_probs = token_log_probs * shift_mask

    return token_log_probs.sum(dim=-1)


def token_entropy(logits: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


# -----------------------------
# Tokenization / generation utilities
# -----------------------------


def build_student_prompts(batch: Sequence[Mapping[str, str]]) -> List[str]:
    return [STUDENT_TEMPLATE.format(problem=item["prompt"]) for item in batch]


def build_teacher_prompts(batch: Sequence[Mapping[str, str]]) -> List[str]:
    return [TEACHER_TEMPLATE.format(problem=item["prompt"], answer=item["answer"]) for item in batch]


def _tokenize_texts(tokenizer, texts: Sequence[str], device: torch.device, max_length: Optional[int] = None) -> Dict[str, Tensor]:
    kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(list(texts), **kwargs)
    return {k: v.to(device) for k, v in encoded.items()}


def _generate_rollouts(model, tokenizer, student_inputs: Mapping[str, Tensor], cfg: CrispStepConfig) -> Tensor:
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
        out = model.generate(**student_inputs, **gen_kwargs)

    prompt_len = student_inputs["input_ids"].shape[1]
    return out.sequences[:, prompt_len:]


# -----------------------------
# Main step
# -----------------------------


def crisp_step(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | CrispStepConfig,
    step: int,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Perform one CRISP update.

    Parameters
    ----------
    model:
        Causal LM with trainable LoRA parameters.
    tokenizer:
        HuggingFace tokenizer.
    batch:
        List of dicts with keys: prompt, answer.
    config:
        Either a plain mapping (YAML-loaded config) or CrispStepConfig.
    step:
        Global training step index.
    optimizer:
        Torch optimizer. Gradients are written directly to model parameters.
    device:
        Optional device override. Defaults to model's first parameter device.

    Returns
    -------
    Dict[str, float]
        Scalar logs for reward, losses, gradient cosine, entropy, lambda, etc.
    """
    if not batch:
        raise ValueError("crisp_step() received an empty batch")

    if isinstance(config, CrispStepConfig):
        cfg = config
    else:
        # Pull from dict-like config with sensible fallbacks.
        training_cfg = config.get("training", {}) if isinstance(config, Mapping) else {}
        crisp_cfg = config.get("crisp", {}) if isinstance(config, Mapping) else {}
        rollout_cfg = config.get("rollout", {}) if isinstance(config, Mapping) else {}
        cfg = CrispStepConfig(
            total_steps=int(training_cfg.get("total_steps", 1500)),
            lambda_max=float(crisp_cfg.get("lambda_max", 1.0)),
            lambda_schedule=str(crisp_cfg.get("lambda_schedule", "cosine")),
            lambda_floor=float(crisp_cfg.get("lambda_floor", 0.0)),
            correctness_gate=bool(crisp_cfg.get("correctness_gate", True)),
            correctness_gate_mode=str(crisp_cfg.get("correctness_gate_mode", "wrong_only")),
            pc_grad=bool(crisp_cfg.get("pc_grad", True)),
            entropy_bonus=float(crisp_cfg.get("entropy_bonus", 0.01)),
            clip_ratio=float(crisp_cfg.get("clip_ratio", 0.2)),
            advantage_eps=float(crisp_cfg.get("advantage_eps", 1e-8)),
            reward_centering=bool(crisp_cfg.get("reward_centering", True)),
            reward_scaling=str(crisp_cfg.get("reward_scaling", "standardize")),
            teacher_student_shared_weights=bool(crisp_cfg.get("teacher_student_shared_weights", True)),
            max_new_tokens=int(rollout_cfg.get("max_new_tokens", 1024)),
            temperature=float(rollout_cfg.get("temperature", 0.7)),
            top_p=float(rollout_cfg.get("top_p", 1.0)),
            top_k=int(rollout_cfg.get("top_k", 0)),
            do_sample=bool(rollout_cfg.get("do_sample", True)),
            stop_on_eos=bool(rollout_cfg.get("stop_on_eos", True)),
        )

    if device is None:
        device = next(model.parameters()).device

    model.train()
    params = get_trainable_params(model)
    if not params:
        raise ValueError("No trainable parameters found on model.")

    # -------------------------
    # 1) Student rollout
    # -------------------------
    student_texts = build_student_prompts(batch)
    student_inputs = _tokenize_texts(tokenizer, student_texts, device=device)
    generated_ids = _generate_rollouts(model, tokenizer, student_inputs, cfg)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # -------------------------
    # 2) Reward computation
    # -------------------------
    rewards = torch.tensor(
        [reward_fn(g, item["answer"]) for g, item in zip(generated_texts, batch)],
        dtype=torch.float32,
        device=device,
    )

    # -------------------------
    # 3) Advantage normalization
    # -------------------------
    if cfg.reward_scaling == "standardize":
        mu = rewards.mean()
        sigma = rewards.std(unbiased=False).clamp_min(cfg.advantage_eps)
        advantages = (rewards - mu) / sigma if cfg.reward_centering else rewards / sigma
    elif cfg.reward_scaling == "center_only":
        mu = rewards.mean()
        advantages = rewards - mu
        sigma = rewards.std(unbiased=False)
    elif cfg.reward_scaling == "none":
        mu = rewards.mean()
        sigma = rewards.std(unbiased=False)
        advantages = rewards
    else:
        raise ValueError(f"Unknown reward_scaling mode: {cfg.reward_scaling}")

    # -------------------------
    # 4) RL loss (recompute with grad)
    # -------------------------
    # Build continuation log-prob objective using the sampled rollout tokens.
    student_inputs_grad = _tokenize_texts(tokenizer, student_texts, device=device)
    rollout_input_ids = torch.cat([student_inputs_grad["input_ids"], generated_ids], dim=1)

    # Attention mask must cover prompt + rollout tokens.
    if "attention_mask" in student_inputs_grad:
        rollout_attention_mask = torch.cat(
            [student_inputs_grad["attention_mask"], torch.ones_like(generated_ids, device=device)], dim=1
        )
    else:
        rollout_attention_mask = None

    rl_outputs = model(input_ids=rollout_input_ids, attention_mask=rollout_attention_mask)
    rl_logprob_sum = sequence_logprobs(rl_outputs.logits, rollout_input_ids, rollout_attention_mask)

    # PPO-style clipped surrogate approximation.
    # Here we only have the current policy, so ratio=1 and clip acts as a bounded scaling on the advantage.
    # This keeps the interface compatible with PPO-style implementations while remaining critic-free.
    clipped_adv = advantages.clamp(min=-cfg.clip_ratio, max=cfg.clip_ratio)
    l_rl = -(clipped_adv * rl_logprob_sum).mean()

    # -------------------------
    # 5) Teacher/student forward passes for self-distillation
    # -------------------------
    teacher_texts = build_teacher_prompts(batch)
    teacher_inputs = _tokenize_texts(tokenizer, teacher_texts, device=device)
    teacher_out = model(**teacher_inputs)
    student_out = model(**student_inputs_grad)

    teacher_logits = teacher_out.logits
    student_logits = student_out.logits

    # For distillation we need matching sequence lengths. We distill on the shared prefix length.
    min_len = min(student_logits.shape[1], teacher_logits.shape[1])
    student_logits = student_logits[:, :min_len, :]
    teacher_logits = teacher_logits[:, :min_len, :]

    # Correctness gating: only apply L_SD to wrong rollouts when configured.
    wrong_mask = (rewards <= 0.0)
    if cfg.correctness_gate and cfg.correctness_gate_mode == "wrong_only":
        if wrong_mask.any():
            l_sd = F.kl_div(
                F.log_softmax(student_logits[wrong_mask], dim=-1),
                F.softmax(teacher_logits[wrong_mask], dim=-1),
                reduction="batchmean",
            )
        else:
            l_sd = torch.zeros((), device=device, requires_grad=True)
    elif cfg.correctness_gate and cfg.correctness_gate_mode == "correct_only":
        right_mask = ~wrong_mask
        if right_mask.any():
            l_sd = F.kl_div(
                F.log_softmax(student_logits[right_mask], dim=-1),
                F.softmax(teacher_logits[right_mask], dim=-1),
                reduction="batchmean",
            )
        else:
            l_sd = torch.zeros((), device=device, requires_grad=True)
    else:
        l_sd = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

    # -------------------------
    # 6) Entropy bonus
    # -------------------------
    entropy = token_entropy(rl_outputs.logits)
    l_entropy = -cfg.entropy_bonus * entropy

    # -------------------------
    # 7) Compute gradients
    # -------------------------
    optimizer.zero_grad(set_to_none=True)

    # Backprop RL and SD separately for PC Grad.
    g_rl = torch.autograd.grad(l_rl, params, retain_graph=True, allow_unused=True)
    g_sd = torch.autograd.grad(l_sd, params, retain=True, allow_unused=True)
    g_entropy = torch.autograd.grad(l_entropy, params, retain_graph=False, allow_unused=True)

    g_rl_flat = flatten_tensors(g_rl, params)
    g_sd_flat = flatten_tensors(g_sd, params)
    g_entropy_flat = flatten_tensors(g_entropy, params)

    lam = schedule_lambda(
        step=step,
        total_steps=cfg.total_steps,
        lambda_max=cfg.lambda_max,
        mode=cfg.lambda_schedule,
        floor=cfg.lambda_floor,
    )

    if cfg.pc_grad:
        combined_flat = pc_grad_pair(g_rl_flat, lam * g_sd_flat)
    else:
        combined_flat = g_rl_flat + lam * g_sd_flat

    combined_flat = combined_flat + g_entropy_flat

    # -------------------------
    # 8) Write gradients and update
    # -------------------------
    unflatten_to_grads(combined_flat, params)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # -------------------------
    # 9) Logging
    # -------------------------
    def _safe_float(x: Any) -> float:
        if isinstance(x, Tensor):
            return float(x.detach().mean().item())
        return float(x)

    cos_rl_sd = float(
        F.cosine_similarity(
            g_rl_flat.unsqueeze(0),
            g_sd_flat.unsqueeze(0),
            dim=-1,
        ).item()
    ) if g_rl_flat.numel() and g_sd_flat.numel() else 0.0

    logs: Dict[str, float] = {
        "reward/mean": _safe_float(rewards.mean()),
        "reward/std": _safe_float(rewards.std(unbiased=False)),
        "loss/rl": _safe_float(l_rl),
        "loss/sd": _safe_float(l_sd),
        "loss/entropy": _safe_float(l_entropy),
        "advantage/mean": _safe_float(advantages.mean()),
        "advantage/variance": _safe_float(advantages.var(unbiased=False)),
        "gradient/cos_rl_sd": cos_rl_sd,
        "policy/entropy": _safe_float(entropy),
        "efficiency/lambda": float(lam),
        "efficiency/tokens_generated": float(generated_ids.numel()),
        "efficiency/batch_size": float(len(batch)),
        "reward/correct_frac": float(rewards.mean().item()),
        "diagnostics/wrong_frac": float(wrong_mask.float().mean().item()),
    }

    return logs


# -----------------------------
# Optional convenience wrapper
# -----------------------------


def crisp_loss_only(
    model: torch.nn.Module,
    tokenizer,
    batch: Sequence[Mapping[str, str]],
    config: Mapping[str, Any] | CrispStepConfig,
    step: int,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """Loss-only version for external training loops.

    Returns tensors instead of stepping the optimizer. Useful when a framework
    wants to own the optimizer / distributed step.
    """
    if device is None:
        device = next(model.parameters()).device

    logs = crisp_step(model, tokenizer, batch, config, step, optimizer=torch.optim.SGD(model.parameters(), lr=0.0), device=device)
    return {k: torch.tensor(v, device=device) for k, v in logs.items()}
