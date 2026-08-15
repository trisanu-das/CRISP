"""
PC-Grad gradient deconfliction between the RL and self-distillation terms
(README Section 2.2).

Implemented with `torch.autograd.grad` (via grad_utils.compute_grads), never
`.backward()` + `.grad` accumulation, specifically because the two loss terms
share every LoRA parameter and need genuinely separate gradient *vectors*
before they're combined. Accumulating both into the same `.grad` buffer via
two `.backward()` calls only gives their sum, which is exactly the naive
summation this method exists to avoid -- and the reference project's own
changelog names "an invalid torch.autograd.grad kwarg" as a bug it hit,
which is this exact code path.

The projection math is unit tested against tiny hand-computed vectors in
tests/test_pcgrad.py (and was additionally hand-verified with a throwaway
numpy version during development, checking that each projected vector really
is orthogonal to the *original* other vector).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from crisp.training.grad_utils import compute_grads


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(flat[offset:offset + n].view_as(p))
        offset += n
    return out


@dataclass
class PCGradResult:
    combined: list[torch.Tensor]
    cos_sim: float
    conflicted: bool
    rl_grad_norm: float
    sd_grad_norm: float


def pc_grad_combine(
    loss_rl: torch.Tensor,
    loss_sd: Optional[torch.Tensor],
    params: list[torch.nn.Parameter],
    lam: float,
    eps: float = 1e-12,
) -> PCGradResult:
    """Compute the PC-Grad-deconflicted combined gradient for one micro-batch.

    `params` must be exactly the trainable parameters shared by both loss
    terms (the LoRA parameters), as a fixed, stable-order list -- the same
    list object should be reused every step so flatten/unflatten stay
    consistent across calls.

    `loss_sd` may be `None` (e.g. every example in the micro-batch was
    correctness-gated to zero -- see crisp_step.py) or a zero-valued tensor
    with no live graph; both are handled by skipping deconfliction and
    returning `g_rl` alone, rather than trying to differentiate a constant.

    Does NOT call `optimizer.step()` or write to `.grad` -- the caller
    accumulates/assigns `result.combined` itself (see training/common_loop.py).
    Keeping this function side-effect-free on `params[i].grad` is what makes
    it directly unit-testable without a real model.
    """
    needs_sd = loss_sd is not None and loss_sd.requires_grad
    g_rl_list = compute_grads(loss_rl, params, retain_graph=needs_sd)
    g_rl = _flatten(g_rl_list)

    if not needs_sd:
        return PCGradResult(
            combined=_unflatten(g_rl, params),
            cos_sim=0.0,
            conflicted=False,
            rl_grad_norm=g_rl.norm().item(),
            sd_grad_norm=0.0,
        )

    g_sd_list = compute_grads(loss_sd, params, retain_graph=False)
    g_sd = _flatten(g_sd_list)

    rl_norm = g_rl.norm()
    sd_norm = g_sd.norm()
    dot = torch.dot(g_rl, g_sd)
    cos_sim = (dot / (rl_norm * sd_norm + eps)).item()

    conflicted = bool((dot < 0.0).item())
    if conflicted:
        # Both projections are computed from the *original* g_rl / g_sd
        # (captured above, before either is modified) rather than
        # sequentially -- projecting g_rl first and then using the
        # already-modified g_rl to project g_sd would make the result depend
        # on an arbitrary processing order for what should be a symmetric
        # operation between exactly two tasks.
        g_rl_proj = g_rl - (dot / (sd_norm ** 2 + eps)) * g_sd
        g_sd_proj = g_sd - (dot / (rl_norm ** 2 + eps)) * g_rl
    else:
        g_rl_proj, g_sd_proj = g_rl, g_sd

    combined = g_rl_proj + lam * g_sd_proj
    return PCGradResult(
        combined=_unflatten(combined, params),
        cos_sim=cos_sim,
        conflicted=conflicted,
        rl_grad_norm=rl_norm.item(),
        sd_grad_norm=sd_norm.item(),
    )
