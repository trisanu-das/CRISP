"""Pure-math training schedules."""
from __future__ import annotations

import math


def lambda_schedule(
    step: int,
    total_steps: int,
    lambda_max: float,
    mode: str = "cosine",
) -> float:
    """Return a bounded auxiliary-loss coefficient.

    Supported modes are ``cosine``, ``linear``, and ``constant``. The step is
    clamped to ``[0, total_steps]`` so an extended/resumed run never flips the
    coefficient negative.
    """
    if total_steps <= 0 or lambda_max <= 0.0:
        return 0.0
    t = min(max(int(step), 0), int(total_steps))
    if mode == "cosine":
        value = lambda_max * math.cos(math.pi * t / (2.0 * total_steps))
    elif mode == "linear":
        value = lambda_max * (1.0 - t / total_steps)
    elif mode == "constant":
        value = lambda_max
    else:
        raise ValueError(
            f"Unknown lambda schedule {mode!r}; expected cosine, linear, or constant."
        )
    return max(0.0, min(float(lambda_max), float(value)))
