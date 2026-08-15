"""CIBO v2 beta controller.

The CIBO v2 design supports two beta modes that are deliberately
disjoint:

  - `fixed`: beta is constant across steps. Useful as a baseline and
    as the "documented" mode that matches the original fixed-beta
    theory (Theorems 1/2 only hold for fixed beta -- see doc Section 7,
    "Guarantee status").

  - `adaptive_sign`: doc Section 7's rate-distortion rule. During
    warm-up, the controller estimates a (frozen-after-warmup) reference
    KL EMA. After warm-up, on each `step()` it tracks the *change* in a
    fast reward EMA and a fast "baselined KL" (= KL EMA - reference KL)
    since the previous step, forms their ratio, and nudges beta by a
    small bounded multiplicative step based only on the SIGN of
    (ratio - 1):

        baselined_kl = kl_ema - reference_kl
        signal        = delta_baselined_kl - delta_reward_ema
        beta          = clip(beta * (1 + step_size * sign(signal)),
                              beta_min, beta_max)

    This increases the penalty when leakage grows faster than reward and
    decreases it when reward improves faster than leakage. Reward is not optional here: a controller that reacts to
    KL alone doesn't implement this rule, it implements a different
    (pure KL-trend) one.

    Safety properties this module guarantees:

      - `beta` is always in `[beta_min, beta_max]`.
      - The reference EMA is frozen after warm-up completes.
      - Only the SIGN of (ratio - 1) drives the update, never its
        magnitude, so a near-zero denominator (guarded by `+eps`) can
        only ever produce a bounded, small step -- never a blow-up.
      - No update is attempted until there are two post-warmup
        observations to form a delta from.

State is JSON-serializable so it can ride in the project checkpoint.
"""
from __future__ import annotations

from typing import Optional


class BetaController:
    """Two-mode beta controller for CIBO v2's IB coefficient."""

    def __init__(
        self,
        beta: float,
        beta_mode: str,
        beta_min: float,
        beta_max: float,
        beta_step_size: float,
        beta_ema_decay: float,
        beta_reference_warmup_steps: int,
        beta_denominator_epsilon: float = 1.0e-6,
    ):
        if beta_mode not in ("fixed", "adaptive_sign"):
            raise ValueError(
                f"beta_mode must be 'fixed' or 'adaptive_sign'; got {beta_mode!r}"
            )
        # beta_min can be 0 to support `beta=0` as a documented "no IB"
        # ablation. beta_min is then a lower-bound for the bound, which
        # for fixed mode is moot; for adaptive_sign it means the
        # adaptive updates can drive beta all the way down to 0.
        if not (0.0 <= beta_min <= beta <= beta_max):
            raise ValueError(
                f"beta bounds must be 0 < beta_min <= beta <= beta_max; "
                f"got beta_min={beta_min} beta={beta} beta_max={beta_max}"
            )
        if not 0.0 <= beta_step_size:
            raise ValueError(f"beta_step_size must be non-negative; got {beta_step_size}")
        if not 0.0 <= beta_ema_decay <= 1.0:
            raise ValueError(f"beta_ema_decay must be in [0, 1]; got {beta_ema_decay}")
        if beta_reference_warmup_steps < 0:
            raise ValueError(
                f"beta_reference_warmup_steps must be non-negative; got {beta_reference_warmup_steps}"
            )
        if beta_denominator_epsilon <= 0.0:
            raise ValueError(
                f"beta_denominator_epsilon must be positive; got {beta_denominator_epsilon}"
            )

        self.beta = float(beta)
        self.beta_mode = beta_mode
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_step_size = float(beta_step_size)
        self.beta_ema_decay = float(beta_ema_decay)
        self.warmup_remaining = int(beta_reference_warmup_steps)
        self.beta_denominator_epsilon = float(beta_denominator_epsilon)

        # Reference KL EMA (frozen after warm-up completes).
        self.reference_kl: float = 0.0
        # Fast EMAs, updated on every observe() call (not just warm-up).
        self.kl_ema: Optional[float] = None
        self.reward_ema: Optional[float] = None
        # Last observation's raw KL, kept for diagnostics/metrics.
        self.last_kl: Optional[float] = None
        # Snapshots from the previous post-warmup step(), needed to form
        # a delta -- there is no meaningful "change" on the very first
        # post-warmup observation.
        self.prev_reward_ema: Optional[float] = None
        self.prev_baselined_kl: Optional[float] = None

    def observe(self, reward: float, kl_value: float) -> None:
        """Record one effective-batch observation before ``step()``.

        The CIBO trainer calls this exactly once per optimizer update after
        aggregating micro-batch metrics, so controller dynamics do not depend
        on ``grad_accum_steps``.
        """
        self.last_kl = float(kl_value)
        self.kl_ema = (
            float(kl_value) if self.kl_ema is None
            else self.beta_ema_decay * self.kl_ema + (1.0 - self.beta_ema_decay) * float(kl_value)
        )
        self.reward_ema = (
            float(reward) if self.reward_ema is None
            else self.beta_ema_decay * self.reward_ema + (1.0 - self.beta_ema_decay) * float(reward)
        )
        if self.beta_mode == "adaptive_sign" and self.warmup_remaining > 0:
            # Reference EMA only accumulates KL during warm-up, then freezes.
            self.reference_kl = (
                float(kl_value) if self.reference_kl == 0.0
                else self.beta_ema_decay * self.reference_kl + (1.0 - self.beta_ema_decay) * float(kl_value)
            )

    def step(self) -> None:
        """Apply one controller step. Called once per optimizer step (after the
        optimizer has already stepped).
        """
        if self.beta_mode == "fixed":
            return
        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
            return
        if self.kl_ema is None or self.reward_ema is None:
            return  # no observation yet this run

        baselined_kl = self.kl_ema - self.reference_kl

        if self.prev_reward_ema is None or self.prev_baselined_kl is None:
            # First post-warmup step: nothing to take a delta against yet.
            self.prev_reward_ema = self.reward_ema
            self.prev_baselined_kl = baselined_kl
            return

        delta_reward = self.reward_ema - self.prev_reward_ema
        delta_baselined_kl = baselined_kl - self.prev_baselined_kl

        # Constraint-controller interpretation: increase beta when leakage
        # grows faster than reward, decrease it when reward improves faster
        # than leakage, and hold it when neither changes materially. This is
        # sign-only and bounded, avoiding the unstable near-zero ratio used
        # by the earlier implementation.
        signal = delta_baselined_kl - delta_reward
        tol = self.beta_denominator_epsilon
        sign = 1.0 if signal > tol else (-1.0 if signal < -tol else 0.0)

        new_beta = self.beta * (1.0 + sign * self.beta_step_size)
        new_beta = max(self.beta_min, min(self.beta_max, new_beta))
        self.beta = float(new_beta)

        self.prev_reward_ema = self.reward_ema
        self.prev_baselined_kl = baselined_kl

    def metrics(self) -> dict:
        """Return the controller's current state as the metrics the plan calls out."""
        baselined_kl = (
            (self.kl_ema - self.reference_kl)
            if self.kl_ema is not None
            else 0.0
        )
        return {
            "cibo/beta": self.beta,
            "cibo/beta_mode": self.beta_mode,
            "cibo/beta_reference_kl": self.reference_kl,
            "cibo/baselined_kl": baselined_kl,
            "cibo/reward_ema": self.reward_ema if self.reward_ema is not None else 0.0,
        }

    def state_dict(self) -> dict:
        return {
            "beta": self.beta,
            "beta_mode": self.beta_mode,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "beta_step_size": self.beta_step_size,
            "beta_ema_decay": self.beta_ema_decay,
            "warmup_remaining": self.warmup_remaining,
            "beta_denominator_epsilon": self.beta_denominator_epsilon,
            "reference_kl": self.reference_kl,
            "kl_ema": self.kl_ema,
            "reward_ema": self.reward_ema,
            "last_kl": self.last_kl,
            "prev_reward_ema": self.prev_reward_ema,
            "prev_baselined_kl": self.prev_baselined_kl,
        }

    def load_state_dict(self, sd: dict) -> None:
        for key, value in sd.items():
            setattr(self, key, value)