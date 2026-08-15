"""
Verifiable reward functions.

Math: extract a final answer from model output (prefer `\\boxed{}`, then a
few fallback textual patterns), then compare against the gold answer using
`math_verify` if it's installed (this is the verifier the reference README
names). If `math_verify` isn't available, falls back to a self-contained
sympy-based comparator so the codebase still runs correctly -- just with a
somewhat less forgiving comparator -- in an environment where that optional
dependency couldn't be installed.

Code: executes generated code against HumanEval-style unit tests in a
*subprocess*, with a timeout. This runs arbitrary model-generated code --
only do this in an isolated/disposable environment (container, VM, throwaway
sandbox), never on a machine with access to anything sensitive. Reward is
purely pass/fail against the provided tests; nothing here inspects, teaches,
or optimizes for *how* to write unsafe code, and the subprocess has no
special privileges beyond whatever the calling process already has.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def extract_last_boxed(text: str) -> Optional[str]:
    """Content of the LAST `\\boxed{...}` in `text`, handling nested braces
    (e.g. `\\boxed{\\frac{1}{2}}`). Returns None if no complete boxed
    expression is found.
    """
    key = "\\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    depth = 1
    buf: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf) if depth == 0 else None


_FINAL_ANSWER_PATTERNS = [
    re.compile(r"final answer is[:\s]*\$?([^\n$]+?)\$?[.\s]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"answer\s*:\s*\$?([^\n$]+)\$?", re.IGNORECASE),
    re.compile(r"ANSWER:\s*\$?([^\n$]+)\$?"),
]


def extract_final_answer(text: str) -> Optional[str]:
    """Best-effort final-answer extraction: `\\boxed{}` first (the format the
    default system prompt asks for), then a few common textual fallbacks,
    then -- as a last resort -- the last standalone number in the text.
    Returns None only if nothing at all could be extracted (an empty or
    completely non-numeric/non-answer-shaped response).
    """
    boxed = extract_last_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def _normalize_for_fallback(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = s.replace(",", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    # \frac{a}{b} -> (a)/(b); handles one level of nesting, which covers the
    # overwhelming majority of real model outputs without pulling in a full
    # LaTeX parser (math_verify is the intended tool for anything gnarlier).
    s = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    if s.endswith("."):
        s = s[:-1]
    if s.endswith("%"):
        s = f"({s[:-1]})/100"
    return s


def _fallback_compare(pred: str, gold: str) -> bool:
    p, g = _normalize_for_fallback(pred), _normalize_for_fallback(gold)
    if p == g:
        return True
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) < 1e-6 * max(1.0, abs(gf))
    except ValueError:
        pass
    try:
        import sympy

        p_expr, g_expr = sympy.sympify(p), sympy.sympify(g)
        diff = sympy.simplify(p_expr - g_expr)
        if diff == 0:
            return True
        # Exact symbolic equality failed, but one side is very often a
        # decimal approximation of the other (a model writing "0.333" for a
        # gold answer of "1/3", say) rather than a genuinely different value
        # -- Python's float() alone can't compare those two strings (float("1/3")
        # raises), which is why this is handled here via sympy's numeric
        # evaluation rather than earlier. Tolerance is intentionally tight:
        # this is meant to catch decimal-rounding, not near-miss answers.
        numeric_diff = complex(sympy.N(diff))
        denom = max(1.0, abs(complex(sympy.N(g_expr))))
        return abs(numeric_diff) < 1e-4 * denom
    except Exception:
        return False


def math_reward(response_text: str, gold_answer: str) -> float:
    pred = extract_final_answer(response_text)
    if pred is None:
        return 0.0
    try:
        from math_verify import parse, verify

        gold_parsed = parse(f"${gold_answer}$")
        pred_parsed = parse(f"${pred}$")
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except ImportError:
        return 1.0 if _fallback_compare(pred, gold_answer) else 0.0
    except Exception:
        # math_verify can legitimately fail to parse malformed model output
        # (truncated LaTeX, a generation that never reached a boxed answer,
        # ...); that is a wrong/ungraded answer, not a crash.
        return 0.0


def extract_code_completion(response_text: str) -> str:
    """Pull the first fenced code block out of a model response; fall back to
    the raw text if the model didn't fence its answer.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
    return match.group(1) if match else response_text


def code_reward(problem: dict, completion: str, timeout: float = 10.0) -> float:
    """`problem` needs HumanEval-schema `prompt`, `test`, `entry_point`.

    Runs `prompt + completion + test + check(entry_point)` as a standalone
    script in a fresh subprocess. Reward is 1.0 iff the script exits 0
    (i.e. every assertion in `test` passed) within `timeout` seconds; any
    non-zero exit, exception, or timeout is reward 0.0. See module docstring
    for the sandboxing caveat -- this executes arbitrary generated code.
    """
    program = "\n".join([
        problem["prompt"],
        completion,
        problem["test"],
        f"check({problem['entry_point']})",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(program)
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if result.returncode == 0 else 0.0


def build_reward_fn(reward_cfg):
    """Build a `(response_text, gold_answer) -> float` reward function for
    training. Code-execution reward (HumanEval) is intentionally not wired
    in here -- it needs the whole problem dict (prompt/test/entry_point), not
    just a gold-answer string, and is only used from eval/run_eval.py where
    that dict is available. See that module if you want to train against a
    code-execution reward directly.
    """
    task = reward_cfg.get("task", "math")
    if task == "math":
        return math_reward
    raise ValueError(f"Unknown reward task '{task}' for build_reward_fn (code tasks are eval-only; see eval/run_eval.py)")
