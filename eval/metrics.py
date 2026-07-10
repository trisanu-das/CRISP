"""Shared evaluation metrics for CRISP.

This module is intentionally small and pure so both the training/eval runner
and any future analysis notebooks can reuse the exact same calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SeedResult:
    """Per-seed summary for a single benchmark."""
    pass_at_1: float
    tokens: float
    correct_count: float
    n: float

    @property
    def tokens_per_correct(self) -> float:
        return float(self.tokens / max(self.correct_count, 1e-8))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if len(values) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    stats = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        stats[i] = float(sample.mean())

    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi


def summarize_seed_results(seed_results: Sequence[SeedResult]) -> dict[str, float]:
    """Aggregate per-seed benchmark results into the standard reporting schema."""
    if len(seed_results) == 0:
        raise ValueError("seed_results must not be empty")

    accs = np.asarray([r.pass_at_1 for r in seed_results], dtype=np.float64)
    tokens = np.asarray([r.tokens for r in seed_results], dtype=np.float64)
    tpc = np.asarray([r.tokens_per_correct for r in seed_results], dtype=np.float64)
    correct = np.asarray([r.correct_count for r in seed_results], dtype=np.float64)

    ci_lo, ci_hi = bootstrap_ci(accs.tolist())

    return {
        "pass@1_mean": float(accs.mean()),
        "pass@1_std": float(accs.std()),
        "tokens_mean": float(tokens.mean()),
        "tokens_per_correct": float(tpc.mean()),
        "correct_mean": float(correct.mean()),
        "bootstrap_ci_95_low": float(ci_lo),
        "bootstrap_ci_95_high": float(ci_hi),
    }


def summarize_raw_results(
    *,
    pass_at_1: Sequence[float],
    tokens: Sequence[float],
    correct_count: Sequence[float],
    n: Sequence[float],
) -> dict[str, float]:
    """Convenience helper for callers that have raw per-seed arrays."""
    if not (len(pass_at_1) == len(tokens) == len(correct_count) == len(n)):
        raise ValueError("All input sequences must have the same length")

    seed_results = [
        SeedResult(
            pass_at_1=float(a),
            tokens=float(t),
            correct_count=float(c),
            n=float(nn),
        )
        for a, t, c, nn in zip(pass_at_1, tokens, correct_count, n)
    ]
    return summarize_seed_results(seed_results)
