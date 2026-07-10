"""Verifiable reward function for CRISP.

This module provides a binary math reward that tries to use the `math_verify`
library when available, and falls back to deterministic string matching when it
is not.

Why this design:
- Math-Verify supports `parse()` / `verify()` for mathematically equivalent
  expressions, including LaTeX and plain expressions, and its parser exposes
  LatexExtractionConfig, ExprExtractionConfig, and StringExtractionConfig. ([github.com](https://github.com/huggingface/Math-Verify))
- The parser's default timeout is not ideal in threaded environments; the
  upstream source notes that `parsing_timeout=None` avoids the signal-based
  timeout path. ([github.com](https://github.com/huggingface/Math-Verify/blob/main/src/math_verify/parser.py))

The public entrypoint is:
    math_verify(prediction: str, ground_truth: str) -> float
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence


# -----------------------------
# Optional Math-Verify imports
# -----------------------------

try:
    from math_verify import parse, verify
    from math_verify.parser import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        StringExtractionConfig,
    )

    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover
    parse = None
    verify = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    StringExtractionConfig = None
    _HAS_MATH_VERIFY = False


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for the verifier."""

    parsing_timeout: Optional[int] = None
    allow_strings_for_mcq: bool = True
    lowercase_strings: bool = True
    treat_parse_failure_as_wrong: bool = True
    normalize_whitespace: bool = True
    use_boxed_priority: bool = True


DEFAULT_REWARD_CONFIG = RewardConfig()


# -----------------------------
# Text normalization / extraction
# -----------------------------


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_ANSWER_RES = [
    re.compile(r"final answer(?: is)?[:\s]*([^\n]+)", re.IGNORECASE),
    re.compile(r"answer(?: is)?[:\s]*([^\n]+)", re.IGNORECASE),
    re.compile(r"therefore[:\s]*([^\n]+)", re.IGNORECASE),
]


def _normalize_text(text: str, *, lowercase: bool = False) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    return text


def _find_boxed_content(text: str) -> Optional[str]:
    """Find the content of the last \\boxed{...} in text, correctly handling
    nested braces (\\frac{1}{2}, \\sqrt{3}, x^{2}, etc.), unlike a naive
    [^}]* regex which stops at the first inner closing brace.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)  # last occurrence: the final answer, not an intermediate step
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    content_start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unbalanced braces, malformed generation
    return text[content_start : i - 1]


def _extract_candidate_answer(text: str) -> str:
    """Best-effort extraction of a final answer string from free-form text."""
    if not text:
        return ""

    boxed = _find_boxed_content(text)
    if boxed is not None:
        return boxed.strip()

    for pat in _FINAL_ANSWER_RES:
        m = pat.search(text)
        if m:
            return m.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return text.strip()

def _looks_like_multiple_choice(answer: str) -> bool:
    s = answer.strip().upper()
    return s in {"A", "B", "C", "D", "E"}


@lru_cache(maxsize=512)
def _build_extraction_config(
    *,
    gold: str,
    pred: str,
    allow_strings_for_mcq: bool,
    lowercase_strings: bool,
):
    """Return a tuple of Math-Verify extraction configs tuned to the inputs."""
    configs: list[Any] = []

    # Math-Verify recommends LatexExtractionConfig + ExprExtractionConfig for
    # most prediction strings, and StringExtractionConfig for MCQ targets.
    if _looks_like_multiple_choice(gold) and allow_strings_for_mcq:
        configs.append(
            StringExtractionConfig(
                strings=("A", "B", "C", "D", "E"),
                try_extract_without_anchor=True,
                lowercase=lowercase_strings,
            )
        )

    # For gold answers, prefer the narrowest reasonable extractor first.
    if any(tok in gold for tok in ("\\", "{", "}", "$", "[", "]")):
        configs.append(LatexExtractionConfig())
        configs.append(ExprExtractionConfig())
    else:
        configs.append(ExprExtractionConfig())
        configs.append(LatexExtractionConfig())

    # Prediction side should be robust to free-form outputs.
    if not _looks_like_multiple_choice(gold):
        configs.append(StringExtractionConfig(strings=("A", "B", "C", "D", "E"), try_extract_without_anchor=True, lowercase=lowercase_strings))

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for cfg in configs:
        key = type(cfg), getattr(cfg, "strings", None), getattr(cfg, "lowercase", None)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return tuple(deduped)


# -----------------------------
# Core reward API
# -----------------------------


def _math_verify_score(
    prediction: str,
    ground_truth: str,
    *,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Use Math-Verify if it is installed, otherwise fall back gracefully."""
    if not _HAS_MATH_VERIFY:
        return _fallback_exact_match_reward(prediction, ground_truth, config=config)

    pred_text = _extract_candidate_answer(prediction)
    gold_text = _extract_candidate_answer(ground_truth)

    try:
        extraction_config = _build_extraction_config(
            gold=gold_text,
            pred=pred_text,
            allow_strings_for_mcq=config.allow_strings_for_mcq,
            lowercase_strings=config.lowercase_strings,
        )

        # Parse both sides using the same library-supported mechanics.
        # The upstream parser supports `parsing_timeout=None` to avoid signal
        # based timeout logic in threaded environments.
        gold_parsed = parse(
            gold_text,
            extraction_config=extraction_config,
            parsing_timeout=config.parsing_timeout,
            raise_on_error=False,
        )
        pred_parsed = parse(
            pred_text,
            extraction_config=extraction_config,
            parsing_timeout=config.parsing_timeout,
            raise_on_error=False,
        )

        if not gold_parsed or not pred_parsed:
            return 0.0 if config.treat_parse_failure_as_wrong else float(bool(gold_parsed and pred_parsed))

        result = verify(gold_parsed, pred_parsed)
        return 1.0 if bool(result) else 0.0
    except Exception:
        return _fallback_exact_match_reward(prediction, ground_truth, config=config)


def _fallback_exact_match_reward(
    prediction: str,
    ground_truth: str,
    *,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Deterministic fallback when Math-Verify is unavailable."""
    pred = _extract_candidate_answer(prediction)
    gold = _extract_candidate_answer(ground_truth)
    pred = _normalize_text(pred, lowercase=config.lowercase_strings)
    gold = _normalize_text(gold, lowercase=config.lowercase_strings)
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold else 0.0


def math_verify(prediction: str, ground_truth: str, *, config: RewardConfig = DEFAULT_REWARD_CONFIG) -> float:
    """Binary reward in [0, 1].

    Returns 1.0 when the prediction is mathematically equivalent to the ground
    truth according to Math-Verify, otherwise 0.0.
    """
    return _math_verify_score(prediction, ground_truth, config=config)


# Backwards-compatible alias used by some training code.
reward_fn = math_verify


# -----------------------------
# Batch helpers
# -----------------------------


def batch_math_verify(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    *,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> list[float]:
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")
    return [math_verify(p, g, config=config) for p, g in zip(predictions, ground_truths)]


# -----------------------------
# CLI smoke test
# -----------------------------


def _main() -> None:
    examples = [
        (r"The answer is \boxed{1/2}", r"\frac{1}{2}"),
        ("final answer is B", "B"),
        ("4", "4"),
    ]
    for pred, gold in examples:
        print(math_verify(pred, gold))


if __name__ == "__main__":
    _main()
