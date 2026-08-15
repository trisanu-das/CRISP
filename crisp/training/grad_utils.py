"""
Small shared helpers for computing per-parameter gradients via
`torch.autograd.grad` (never `.backward()` + reading `.grad` back off) and
turning the result into plain tensor lists that are safe to add together.

Every method in this codebase (CRISP and all three baselines) ultimately
returns a `list[Tensor]` aligned 1:1 with a fixed `params` list from its step
function -- see training/common_loop.py for why that uniform interface is
worth keeping even for the baselines, which don't need PC-Grad's gradient
surgery.
"""
from __future__ import annotations

import torch


def grad_or_zeros(grads, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """`torch.autograd.grad(..., allow_unused=True)` returns `None` for any
    parameter a particular loss's graph never touched (e.g. a LoRA adapter on
    a layer that happened not to fire for every token in a short response).
    Downstream code (accumulation, PC-Grad's dot products) needs real zero
    tensors there instead, so every parameter stays aligned across calls.
    """
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def compute_grads(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool = False) -> list[torch.Tensor]:
    """One loss -> one gradient per parameter in `params`, in order.

    Deliberately does not touch `param.grad`: computing g_rl and g_sd as
    independent vectors (rather than both accumulating into the same `.grad`
    buffer via two `.backward()` calls) is the entire point -- PC-Grad needs
    to inspect and reconcile them *before* anything is applied to the
    parameters. See training/pcgrad.py.
    """
    raw = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return grad_or_zeros(raw, params)
