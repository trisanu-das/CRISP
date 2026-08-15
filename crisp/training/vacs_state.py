"""Adaptive gradient-energy state for VACS.

The implementation aggregates every micro-batch observation in an optimizer
step before committing one EMA update. Earlier versions used "last write
wins", making the controller depend on ``grad_accum_steps`` and discarding
most of the applied gradient statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def observe_grad_stats(
    g_rl: torch.Tensor, g_sd: torch.Tensor, eps: float = 1e-12
) -> tuple[float, float]:
    rl_norm_sq = float(g_rl.detach().pow(2).sum().item())
    rl_norm = float(g_rl.detach().norm().item())
    sd_norm = float(g_sd.detach().norm().item())
    cos_value = 0.0
    if rl_norm > 0.0 and sd_norm > 0.0:
        cos_value = float(
            torch.dot(g_rl.detach(), g_sd.detach()).item()
            / (rl_norm * sd_norm + eps)
        )
        cos_value = max(-1.0, min(1.0, cos_value))
    return rl_norm_sq, cos_value


@dataclass
class VacsAdaptiveState:
    ema_decay: float = 0.95
    multiplier_min: float = 0.10
    multiplier_max: float = 5.0
    disabled: bool = False

    rl_grad_norm_sq_ema: float = 0.0
    sd_grad_norm_sq_ema: float = 0.0
    cos_ema: float = 0.0

    _pending_rl_sum: float = 0.0
    _pending_sd_sum: float = 0.0
    _pending_cos_sum: float = 0.0
    _pending_count: int = 0

    def __post_init__(self):
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1]; got {self.ema_decay}")
        if self.multiplier_min <= 0.0:
            raise ValueError("multiplier_min must be positive")
        if self.multiplier_max < self.multiplier_min:
            raise ValueError("multiplier_max must be >= multiplier_min")

    # Compatibility accessors used by older hooks/tests.
    @property
    def _pending_rl_norm_sq(self) -> float:
        return self.pending_means()[0]

    @property
    def _pending_sd_norm_sq(self) -> float:
        return self.pending_means()[1]

    @property
    def _pending_cos(self) -> float:
        return self.pending_means()[2]

    @property
    def _has_pending(self) -> bool:
        return self._pending_count > 0

    def observe(
        self,
        norm_sq: float,
        cos_value: float,
        sd_norm_sq: float | None = None,
    ) -> None:
        self._pending_rl_sum += float(norm_sq)
        self._pending_sd_sum += float(norm_sq if sd_norm_sq is None else sd_norm_sq)
        self._pending_cos_sum += float(cos_value)
        self._pending_count += 1

    def pending_means(self) -> tuple[float, float, float]:
        if self._pending_count == 0:
            return 0.0, 0.0, 0.0
        count = float(self._pending_count)
        return (
            self._pending_rl_sum / count,
            self._pending_sd_sum / count,
            self._pending_cos_sum / count,
        )

    def clear_pending(self) -> None:
        self._pending_rl_sum = 0.0
        self._pending_sd_sum = 0.0
        self._pending_cos_sum = 0.0
        self._pending_count = 0

    def commit(
        self,
        rl_grad_norm_sq: float | None = None,
        sd_grad_norm_sq: float | None = None,
        cos_value: float | None = None,
    ) -> None:
        if rl_grad_norm_sq is None or sd_grad_norm_sq is None or cos_value is None:
            rl_grad_norm_sq, sd_grad_norm_sq, cos_value = self.pending_means()
        if float(rl_grad_norm_sq) > 0.0:
            self.rl_grad_norm_sq_ema = (
                self.ema_decay * self.rl_grad_norm_sq_ema
                + (1.0 - self.ema_decay) * float(rl_grad_norm_sq)
            )
        if float(sd_grad_norm_sq) > 0.0:
            self.sd_grad_norm_sq_ema = (
                self.ema_decay * self.sd_grad_norm_sq_ema
                + (1.0 - self.ema_decay) * float(sd_grad_norm_sq)
            )
        self.cos_ema = (
            self.ema_decay * self.cos_ema
            + (1.0 - self.ema_decay) * float(cos_value)
        )
        self.clear_pending()

    def state_dict(self) -> dict:
        return {
            "ema_decay": self.ema_decay,
            "multiplier_min": self.multiplier_min,
            "multiplier_max": self.multiplier_max,
            "disabled": self.disabled,
            "rl_grad_norm_sq_ema": self.rl_grad_norm_sq_ema,
            "sd_grad_norm_sq_ema": self.sd_grad_norm_sq_ema,
            "cos_ema": self.cos_ema,
        }

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        self.clear_pending()


def compute_lambda_effective(state: VacsAdaptiveState, base_lambda: float) -> float:
    if state.disabled:
        return float(base_lambda)
    eps = 1e-12
    ratio = state.rl_grad_norm_sq_ema / (state.sd_grad_norm_sq_ema + eps)
    raw = math.sqrt(max(ratio, 0.0))
    multiplier = max(state.multiplier_min, min(state.multiplier_max, raw))
    cosine_factor = (1.0 + max(-1.0, min(1.0, state.cos_ema))) / 2.0
    return float(base_lambda) * float(multiplier) * float(cosine_factor)
