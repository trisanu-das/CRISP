"""Reusable objective primitives shared by VACS-CRISP and CIBO-CRISP v2.

Every function here is intentionally tiny, pure, and PyTorch-native:
  - no hidden global state,
  - no method-policy embedding (no opinion about *when* to call them,
    or about hyperparameters -- those live in the method layer),
  - all teacher-derived credit is detached at the function boundary, so
    the autograd graph never flows through the teacher side.

This is the place where mathematically sensitive token credit, the soft
reward gate, and the gated forward-KL reduction live, before either new
training method stacks them into a loss. Centralizing the math here
means each one has a single source of truth and a single dedicated test
file (tests/test_objectives.py) that can fail in isolation, instead of
being debugged through a giant stacked objective.
"""
from __future__ import annotations

from typing import Iterable

import torch


_VALID_REDUCTIONS = frozenset({"sequence_sum", "token_mean"})


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise ValueError with context if a tensor contains any non-finite values."""
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"Non-finite value in {name}; refusing to silently propagate it into the loss. "
            f"Tensor shape={tuple(tensor.shape)}; first 10 values={tensor.reshape(-1)[:10].tolist()}"
        )


def normalize_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Global-batch mean/std normalized advantages.

    `(r - mean) / (std + eps)` with a biased (population) std so the
    single-example micro-batch case still yields a finite result. The
    small `eps` keeps the denominator strictly positive when all rewards
    in a batch are identical (which would otherwise divide by zero).
    """
    if not torch.isfinite(rewards).all():
        raise ValueError("normalize_advantages received non-finite rewards.")
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def _token_credit(
    teacher_token_logps: torch.Tensor, student_token_logps: torch.Tensor
) -> torch.Tensor:
    """Per-token detached credit `sg[log p_T - log p_S]`.

    Both sides are detached. The spec is explicit that this is a pure
    stop-gradient quantity ("Teacher information appears only through
    stop-gradient weights", doc Section 3.1) -- it is a *weight* on the
    RL loss, not itself a differentiable term. The student already gets
    its proper gradient through the explicit `log pi_theta(y_t|...)`
    term in the RL loss (see vacs_step.py / cibo_v2_step.py); leaving
    the student side live here would open a *second*, unspecified
    gradient pathway to student parameters through the weight itself,
    on top of the one the loss formula actually calls for.
    """
    credit = teacher_token_logps.detach() - student_token_logps.detach()
    _assert_finite("token_credit", credit)
    return credit


def vacs_token_advantages(
    advantages: torch.Tensor,
    teacher_token_logps: torch.Tensor,
    student_token_logps: torch.Tensor,
    credit_mix: float,
    clip_epsilon: float,
) -> torch.Tensor:
    """VACS token-level advantage, doc Section 3.1:

        c_{i,t}   = sg[log p_T - log p_S]
        w_{i,t}   = exp(sign(a_i) * c_{i,t}) = clip(..., 1 - eps, 1 + eps)
        a_tilde_{i,t} = a_i * [(1 - rho) + rho * w_{i,t}]

    The `sign(a_i)` factor is load-bearing, not optional: without it, a
    teacher-endorsed token (c > 0) gets the same weight regardless of
    whether this rollout is being reinforced (a_i > 0) or penalized
    (a_i < 0) -- which defeats the entire point of differential credit
    assignment (a teacher-endorsed token inside an otherwise-wrong
    trajectory should be penalized *less*, not the same amount).

    Properties verified by tests/test_objectives.py:
      - rho = 0 -> a_tilde == a (token credit disabled).
      - c_{i,t} = 0 -> w = 1 -> a_tilde == a (no credit, no change).
      - For a_i > 0 and positive credit, a_tilde > a (reinforce where
        the student underweights the teacher).
      - For a_i < 0 and positive credit, a_tilde is LESS negative than
        a_i (penalize less where the teacher endorses this specific
        token even though the trajectory overall was wrong).
      - `teacher_token_logps` and `student_token_logps` are both
        detached inside the credit -- see `_token_credit`.
    """
    if advantages.dim() != 1:
        raise ValueError(f"advantages must be 1-D; got shape {tuple(advantages.shape)}")
    if teacher_token_logps.shape != student_token_logps.shape:
        raise ValueError(
            f"teacher/student token-log-prob shapes disagree: "
            f"{tuple(teacher_token_logps.shape)} vs {tuple(student_token_logps.shape)}"
        )
    if advantages.shape[0] != teacher_token_logps.shape[0]:
        raise ValueError(
            f"advantages first dim ({advantages.shape[0]}) must match token-log-prob first dim "
            f"({teacher_token_logps.shape[0]}); one advantage per rollout."
        )
    if not 0.0 <= credit_mix <= 1.0:
        raise ValueError(f"credit_mix must be in [0, 1]; got {credit_mix}")
    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError(f"clip_epsilon must be in [0, 1); got {clip_epsilon}")

    credit = _token_credit(teacher_token_logps, student_token_logps)
    adv_sign = torch.sign(advantages).unsqueeze(1)  # [N, 1]; sign(0) == 0 by convention
    weight = torch.exp(adv_sign * credit).clamp(min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    rho = float(credit_mix)
    per_token_adv = advantages.unsqueeze(1) * ((1.0 - rho) + rho * weight)
    _assert_finite("vacs token advantages", per_token_adv)
    return per_token_adv


def cibo_token_advantages(
    advantages: torch.Tensor,
    teacher_token_logps: torch.Tensor,
    student_token_logps: torch.Tensor,
    credit_lambda: float,
) -> torch.Tensor:
    """CIBO-CRISP v2 sign-safe token advantage.

        a_tilde_{i,t} = a_i * (1 + lambda * tanh(c_{i,t}))

    with `tanh` chosen specifically because it is bounded in (-1, 1),
    so the multiplier `(1 + lambda * tanh(c))` is bounded in
    `(1 - lambda, 1 + lambda)`. For any non-zero advantage, this means
    the token advantage can NEVER change sign -- the test enforces this
    by checking `a_tilde * a > 0` for several (a, t, s) configurations.

    `lambda` is bounded in [0, 1] at config-load time and re-checked here.
    """
    if advantages.dim() != 1:
        raise ValueError(f"advantages must be 1-D; got shape {tuple(advantages.shape)}")
    if teacher_token_logps.shape != student_token_logps.shape:
        raise ValueError(
            f"teacher/student token-log-prob shapes disagree: "
            f"{tuple(teacher_token_logps.shape)} vs {tuple(student_token_logps.shape)}"
        )
    if advantages.shape[0] != teacher_token_logps.shape[0]:
        raise ValueError(
            f"advantages first dim ({advantages.shape[0]}) must match token-log-prob first dim "
            f"({teacher_token_logps.shape[0]}); one advantage per rollout."
        )
    if not 0.0 <= credit_lambda <= 1.0:
        raise ValueError(f"credit_lambda must be in [0, 1]; got {credit_lambda}")

    credit = _token_credit(teacher_token_logps, student_token_logps)
    per_token_adv = advantages.unsqueeze(1) * (1.0 + credit_lambda * torch.tanh(credit))
    _assert_finite("cibo token advantages", per_token_adv)
    return per_token_adv


def soft_reward_gate(rewards: torch.Tensor, slope: float, threshold: float) -> torch.Tensor:
    """Soft reward gate used to down-weight self-distillation on correct rollouts.

        g_i = 1 - sigmoid(slope * (r_i - threshold))

    Higher reward -> smaller gate (the student is already getting it
    right, so the privileged-answer distillation term is suppressed).
    Lower reward -> larger gate (distillation is most useful where the
    student is failing).

    Steeper slope makes the transition sharper (closer to a step
    function); `slope=12, threshold=0.5` is the default pairing.
    """
    if not torch.isfinite(rewards).all():
        raise ValueError("soft_reward_gate received non-finite rewards.")
    gate = 1.0 - torch.sigmoid(slope * (rewards - threshold))
    return gate


def reduce_gated_kl(
    per_token_kl: Iterable[torch.Tensor],
    gate_weights: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Reduce per-token gated KL into one scalar.

    `per_token_kl[i]` is the 1-D tensor of per-token forward KL for
    rollout i (shape `[response_len_i]`), and `gate_weights[i]` is the
    soft reward gate for that rollout. Two reductions are supported:

      - "sequence_sum": sum over tokens, then mean over batch.
        (This is the CIBO v2 default per the design document, which
        describes the sum as the "documented" reduction.)
      - "token_mean": mean over tokens, then mean over batch.

    The two reductions are deliberately kept distinct: the CIBO plan
    requires that `token_mean` never silently substitute for the
    default. Empty token rows contribute zero (length-0 `per_token_kl`
    rows are preserved when the caller leaves them in the list).
    """
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {sorted(_VALID_REDUCTIONS)}; got {reduction!r}"
        )
    rows = list(per_token_kl)
    if gate_weights.dim() != 1 or gate_weights.shape[0] != len(rows):
        raise ValueError(
            f"gate_weights must be 1-D with one entry per row ({len(rows)}); got shape "
            f"{tuple(gate_weights.shape)}"
        )

    # Apply the gate first, then reduce. Empty token rows contribute zero:
    # sum([]) = 0, and mean([]) is replaced by a scalar zero so a single
    # empty row can't blow up the per-example reduction.
    gated = []
    for i, kl in enumerate(rows):
        _assert_finite(f"per_token_kl[{i}]", kl)
        gated.append(kl * float(gate_weights[i].item()))

    if reduction == "sequence_sum":
        per_example = torch.stack([g.sum() for g in gated])
    else:  # "token_mean"
        per_example = torch.stack([
            (g.sum() / max(int(g.numel()), 1)) if int(g.numel()) > 0
            else torch.zeros((), dtype=g.dtype, device=g.device)
            for g in gated
        ])

    out = per_example.mean()
    _assert_finite("reduce_gated_kl", out)
    return out
