"""Small torch.distributed helpers for manual-gradient training.

CRISP and VACS use ``torch.autograd.grad`` plus gradient surgery, so wrapping
the model in DDP is not sufficient by itself. This module instead synchronizes
reward statistics and final per-parameter gradients explicitly.
"""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed() -> bool:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False
    if not dist.is_available():
        raise RuntimeError("WORLD_SIZE>1 but torch.distributed is unavailable.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return True


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0")) if is_initialized() else 0


def is_main_process() -> bool:
    return rank() == 0


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def gather_1d_equal(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather equal-length 1-D tensors and concatenate them."""
    if not is_initialized():
        return tensor
    if tensor.dim() != 1:
        raise ValueError(f"Expected a 1-D tensor, got {tuple(tensor.shape)}")
    gathered = [torch.empty_like(tensor) for _ in range(world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def all_reduce_gradients(grads: list[torch.Tensor]) -> list[torch.Tensor]:
    if not is_initialized():
        return grads
    size = float(world_size())
    for grad in grads:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(size)
    return grads


def all_reduce_scalar(value: float, device: torch.device | str) -> float:
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(world_size())
    return float(t.item())
