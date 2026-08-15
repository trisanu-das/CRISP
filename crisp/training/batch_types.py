"""Shared rollout-batch records for two-stage RL training.

Stage A collects on-policy rollouts without retaining autograd graphs. Stage B
scores those stored rollouts after reward statistics have been computed over
the entire effective batch (all gradient-accumulation micro-batches and, when
distributed training is enabled, all ranks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CollectedBatch:
    prompts: list[str]
    answers: list[str]
    rollouts: list[Any]
    rewards: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.rewards.dim() != 1:
            raise ValueError(f"rewards must be 1-D, got {tuple(self.rewards.shape)}")
        if len(self.rollouts) != int(self.rewards.numel()):
            raise ValueError(
                f"rollout/reward count mismatch: {len(self.rollouts)} vs {self.rewards.numel()}"
            )
        if not torch.isfinite(self.rewards).all():
            raise ValueError("Collected rewards contain non-finite values.")


@dataclass(frozen=True)
class AdvantageStats:
    mean: float
    std: float
    count: int

    def normalize(self, rewards: torch.Tensor, eps: float) -> torch.Tensor:
        if not torch.isfinite(rewards).all():
            raise ValueError("Cannot normalize non-finite rewards.")
        return (rewards - float(self.mean)) / (float(self.std) + float(eps))
