"""
Unit tests for crisp/data/reward.py.

Deliberately torch-free: answer extraction/comparison and the code-exec
sandbox are pure-Python (+ sympy) and should be checkable in any environment,
including one where torch/transformers aren't installed yet.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.data.reward import (  # noqa: E402
    _fallback_compare,
    _normalize_for_fallback,
    code_reward,
    extract_code_completion,
    extract_final_answer,
    extract_last_boxed,
    math_reward,
)


class TestExtractLastBoxed(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(extract_last_boxed("The answer is \\boxed{42}."), "42")

    def test_nested_braces(self):
        self.assertEqual(extract_last_boxed("So x = \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")

    def test_takes_last_occurrence(self):
        text = "First I guessed \\boxed{1} but actually \\boxed{2}"
        self.assertEqual(extract_last_boxed(text), "2")

    def test_no_boxed_returns_none(self):
        self.assertIsNone(extract_last_boxed("no boxed answer here"))

    def test_unclosed_boxed_returns_none(self):
        self.assertIsNone(extract_last_boxed("\\boxed{unterminated"))

    def test_deeply_nested(self):
        text = "\\boxed{\\frac{\\sqrt{2}}{3}}"
        self.assertEqual(extract_last_boxed(text), "\\frac{\\sqrt{2}}{3}")


class TestExtractFinalAnswer(unittest.TestCase):
    def test_prefers_boxed(self):
        text = "The answer is 5, wait let me box it: \\boxed{7}"
        self.assertEqual(extract_final_answer(text), "7")

    def test_falls_back_to_answer_colon(self):
        text = "Let's work through this.\nAnswer: 17"
        self.assertEqual(extract_final_answer(text), "17")

    def test_falls_back_to_last_number(self):
        text = "We compute 3 + 4 to get 7 apples."
        self.assertEqual(extract_final_answer(text), "7")

    def test_empty_text_returns_none(self):
        self.assertIsNone(extract_final_answer(""))


class TestFallbackCompare(unittest.TestCase):
    def test_exact_string_match(self):
        self.assertTrue(_fallback_compare("42", "42"))

    def test_numeric_tolerance(self):
        self.assertTrue(_fallback_compare("0.3333333", "1/3"))

    def test_integer_vs_float_string(self):
        self.assertTrue(_fallback_compare("4.0", "4"))

    def test_fraction_normalization(self):
        self.assertEqual(_normalize_for_fallback("\\frac{1}{2}"), "(1)/(2)")

    def test_sympy_equivalence(self):
        # 2*sqrt(2) written two different ways.
        self.assertTrue(_fallback_compare("sqrt(8)", "2*sqrt(2)"))

    def test_mismatched_values_are_rejected(self):
        self.assertFalse(_fallback_compare("3", "4"))

    def test_decimal_approximation_of_fraction_is_accepted(self):
        self.assertTrue(_fallback_compare("0.3333333", "1/3"))

    def test_loose_decimal_rounding_is_still_rejected(self):
        # off by ~0.033, i.e. nowhere near float rounding error of 1/3
        self.assertFalse(_fallback_compare("0.3", "1/3"))

    def test_comma_thousands_separator(self):
        self.assertTrue(_fallback_compare("1,234", "1234"))

    def test_percentage(self):
        self.assertTrue(_fallback_compare("50%", "1/2"))


class TestMathReward(unittest.TestCase):
    """These exercise the *fallback* comparator specifically, by testing
    through math_reward with math_verify uninstalled being the expected
    environment for this sandbox. If math_verify IS installed, math_reward
    will use it instead -- both paths should agree on these simple cases,
    which is itself a useful cross-check once torch/math_verify are present.
    """

    def test_correct_boxed_answer(self):
        self.assertEqual(math_reward("Steps... \\boxed{12}", "12"), 1.0)

    def test_incorrect_boxed_answer(self):
        self.assertEqual(math_reward("Steps... \\boxed{13}", "12"), 0.0)

    def test_no_extractable_answer(self):
        self.assertEqual(math_reward("I am not sure how to solve this.", "12"), 0.0)

    def test_equivalent_fraction_forms(self):
        self.assertEqual(math_reward("\\boxed{\\frac{2}{4}}", "1/2"), 1.0)


class TestExtractCodeCompletion(unittest.TestCase):
    def test_fenced_python_block(self):
        text = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```"
        self.assertEqual(extract_code_completion(text), "def f(x):\n    return x + 1\n")

    def test_no_fence_returns_raw_text(self):
        text = "def f(x):\n    return x + 1"
        self.assertEqual(extract_code_completion(text), text)


class TestCodeReward(unittest.TestCase):
    def _problem(self):
        return {
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
            "entry_point": "add",
        }

    def test_passing_solution(self):
        completion = "    return a + b\n"
        self.assertEqual(code_reward(self._problem(), completion, timeout=5.0), 1.0)

    def test_failing_solution(self):
        completion = "    return a - b\n"
        self.assertEqual(code_reward(self._problem(), completion, timeout=5.0), 0.0)

    def test_infinite_loop_times_out_as_failure(self):
        problem = {
            "prompt": "def f():\n",
            "test": "def check(candidate):\n    assert candidate() == 1\n",
            "entry_point": "f",
        }
        completion = "    while True:\n        pass\n"
        self.assertEqual(code_reward(problem, completion, timeout=1.0), 0.0)

    def test_syntax_error_is_failure_not_crash(self):
        completion = "    this is not valid python(((\n"
        self.assertEqual(code_reward(self._problem(), completion, timeout=5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
