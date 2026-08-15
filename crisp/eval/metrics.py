"""
Evaluation metrics: pass@1 with a bootstrap confidence interval, and
token-efficiency accounting (README Section 5.4's "tokens per correct
answer"). Pure Python/stdlib -- no torch dependency, since none of this
needs a model, only a list of already-computed correctness flags.
"""
from __future__ import annotations

import random


def pass_at_1(correct_flags: list[bool]) -> float:
    if not correct_flags:
        return 0.0
    return sum(correct_flags) / len(correct_flags)


def bootstrap_ci(
    correct_flags: list[bool], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI on pass@1. Meant for small eval sets (AIME's
    ~30 problems) where a bare point estimate is close to meaningless --
    with n=30, one extra correct answer moves pass@1 by more than 3 points.
    """
    if not correct_flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct_flags)
    samples = []
    for _ in range(n_bootstrap):
        resampled = [correct_flags[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(resampled) / n)
    samples.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = min(int((1 + ci) / 2 * n_bootstrap) - 1, n_bootstrap - 1)
    return samples[lo_idx], samples[hi_idx]


def tokens_per_correct(total_tokens: int, num_correct: int) -> float:
    if num_correct == 0:
        return float("inf")
    return total_tokens / num_correct


def summarize(results: list[dict], n_bootstrap: int = 1000, bootstrap_seed: int = 0) -> dict:
    """`results`: list of `{"correct": bool, "response_tokens": int}`, one per problem."""
    if not results:
        return {
            "pass@1_mean": 0.0, "item_correctness_std": 0.0, "pass@1_std": 0.0, "pass@1_ci_lo": 0.0, "pass@1_ci_hi": 0.0,
            "total_tokens": 0, "tokens_per_correct": float("inf"), "n": 0,
        }
    correct_flags = [r["correct"] for r in results]
    total_tokens = sum(r["response_tokens"] for r in results)
    num_correct = sum(correct_flags)
    mean = pass_at_1(correct_flags)
    variance = sum((float(f) - mean) ** 2 for f in correct_flags) / len(correct_flags)
    lo, hi = bootstrap_ci(correct_flags, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    return {
        "pass@1_mean": mean,
        "item_correctness_std": variance ** 0.5,
        # Deprecated compatibility alias; this is NOT cross-seed std.
        "pass@1_std": variance ** 0.5,
        "pass@1_ci_lo": lo,
        "pass@1_ci_hi": hi,
        "total_tokens": total_tokens,
        "tokens_per_correct": tokens_per_correct(total_tokens, num_correct),
        "n": len(results),
    }
