"""Asymmetric PC-Grad for VACS-CRISP: protect the RL gradient, project only the auxiliary.

The legacy symmetric PC-Grad (crisp/training/pcgrad.py) projects BOTH
gradient vectors away from each other on conflict -- `g_rl_proj = g_rl -
projection(g_rl, g_sd)` and vice versa. That is the right behaviour for
classic PC-Grad: the two losses are peers.

VACS is not symmetric. The RL term is the primary objective; the
self-distillation term is auxiliary. Under conflict, VACS requires:

  - The RL gradient is preserved bit-for-bit.
  - The auxiliary (SD) gradient is projected away from the ORIGINAL RL
    gradient (not a modified RL gradient, since RL is untouched).

That is what this module implements. It returns:

  - `combined = g_rl + lambda_eff * g_sd_projected`,
  - a `cos_sim` between the original g_rl and g_sd,
  - whether the projections actually happened (`conflicted`),
  - the two gradient norms.

The function is side-effect-free on `params[i].grad` and on the input
gradient lists, mirroring the legacy helper's discipline. That keeps
it directly unit-testable without a real model and makes the asymmetric
behaviour explicitly attributable to this code path -- not to the
caller's `param.grad` accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def flatten_grads(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of per-parameter gradient tensors into one flat vector."""
    return torch.cat([g.reshape(-1) for g in grads])


def unflatten_grads(
    flat: torch.Tensor, params: Sequence[torch.nn.Parameter]
) -> list[torch.Tensor]:
    """Split a flat gradient vector back into per-parameter tensors.

    Each output tensor has the shape and dtype of its corresponding
    parameter. The split is purely positional: the caller's params must
    come in the same stable order they were passed to
    `asymmetric_pcgrad_combine`.
    """
    out: list[torch.Tensor] = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(flat[offset:offset + n].view_as(p))
        offset += n
    return out


@dataclass
class AsymmetricPCGradResult:
    """Result of one asymmetric PC-Grad combine call.

    Attributes mirror the legacy `PCGradResult` plus an explicit
    `auxiliary_projected` so the caller can audit / log the projected
    SD gradient independently of the combined vector.
    """

    combined: list[torch.Tensor]
    auxiliary_projected: list[torch.Tensor]
    cos_sim: float
    conflicted: bool
    rl_grad_norm: float
    sd_grad_norm: float


def asymmetric_pcgrad_combine(
    g_rl_list: Sequence[torch.Tensor],
    g_sd_list: Sequence[torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    lam: float,
    eps: float = 1e-12,
) -> AsymmetricPCGradResult:
    """Combine RL and SD gradients with asymmetric PC-Grad surgery.

    The combined output is `g_rl + lam * g_sd_projected`. The projection
    is computed against the ORIGINAL `g_rl` -- `g_rl` itself is never
    modified. This is the structural difference from symmetric PC-Grad.

    Edge cases:
      - `g_sd` is the zero vector: no projection (division by `eps` is
        safe), `conflicted = False`, combined = g_rl.
      - `g_rl` is the zero vector: no projection (the conflict flag is
        the sign of the dot product, which is 0; not <0), the
        auxiliary passes through unchanged, combined = lam * g_sd.
      - All inputs must be on the same device/dtype; mismatches are the
        caller's responsibility, not this helper's.
    """
    if len(g_rl_list) != len(g_sd_list):
        raise ValueError(
            f"g_rl_list / g_sd_list length mismatch: {len(g_rl_list)} vs {len(g_sd_list)}"
        )
    if len(g_rl_list) != len(params):
        raise ValueError(
            f"gradient list and params length mismatch: {len(g_rl_list)} vs {len(params)}"
        )

    g_rl_flat = flatten_grads(g_rl_list)
    g_sd_flat = flatten_grads(g_sd_list)

    rl_norm = g_rl_flat.norm()
    sd_norm = g_sd_flat.norm()
    dot = torch.dot(g_rl_flat, g_sd_flat)
    cos_sim = (dot / (rl_norm * sd_norm + eps)).item()

    conflicted = bool((dot < 0.0).item())

    # RL is preserved bit-for-bit. SD is projected away from the
    # ORIGINAL g_rl only when the projections are valid (g_rl has
    # nonzero norm) and the dot product is negative (conflict).
    if conflicted and float(rl_norm.item()) > 0.0:
        # Hand-computed: g_sd_proj = g_sd - (dot(g_rl, g_sd) / |g_rl|^2) * g_rl
        g_sd_proj = g_sd_flat - (dot / (rl_norm ** 2 + eps)) * g_rl_flat
    else:
        g_sd_proj = g_sd_flat

    combined_flat = g_rl_flat + lam * g_sd_proj

    return AsymmetricPCGradResult(
        combined=unflatten_grads(combined_flat, params),
        auxiliary_projected=unflatten_grads(g_sd_proj, params),
        cos_sim=cos_sim,
        conflicted=conflicted,
        rl_grad_norm=rl_norm.item(),
        sd_grad_norm=sd_norm.item(),
    )
