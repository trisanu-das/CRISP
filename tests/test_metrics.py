from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.eval.metrics import bootstrap_ci, pass_at_1, summarize, tokens_per_correct  # noqa: E402


class TestPassAt1(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(pass_at_1([True, True, True]), 1.0)

    def test_all_wrong(self):
        self.assertEqual(pass_at_1([False, False]), 0.0)

    def test_mixed(self):
        self.assertAlmostEqual(pass_at_1([True, False, True, False]), 0.5)

    def test_empty(self):
        self.assertEqual(pass_at_1([]), 0.0)


class TestBootstrapCI(unittest.TestCase):
    def test_all_correct_gives_tight_ci_at_one(self):
        lo, hi = bootstrap_ci([True] * 30, n_bootstrap=500, seed=0)
        self.assertEqual(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_ci_bounds_contain_point_estimate(self):
        flags = [True, True, False, True, False, True, True, False, True, True]
        lo, hi = bootstrap_ci(flags, n_bootstrap=2000, seed=0)
        point = pass_at_1(flags)
        self.assertLessEqual(lo, point + 1e-9)
        self.assertGreaterEqual(hi, point - 1e-9)

    def test_deterministic_given_seed(self):
        flags = [True, False, True, True, False]
        r1 = bootstrap_ci(flags, n_bootstrap=200, seed=42)
        r2 = bootstrap_ci(flags, n_bootstrap=200, seed=42)
        self.assertEqual(r1, r2)

    def test_empty_input_does_not_crash(self):
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0))

    def test_small_n_produces_wider_ci_than_large_n(self):
        # A sanity check on the *shape* of the behavior rather than exact
        # numbers: fewer problems -> more sampling noise -> wider interval.
        small = [True, False, True]
        large = [True, False, True] * 20
        lo_s, hi_s = bootstrap_ci(small, n_bootstrap=2000, seed=1)
        lo_l, hi_l = bootstrap_ci(large, n_bootstrap=2000, seed=1)
        self.assertGreater(hi_s - lo_s, hi_l - lo_l)


class TestTokensPerCorrect(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokens_per_correct(1000, 10), 100.0)

    def test_zero_correct_is_infinite_not_a_crash(self):
        self.assertEqual(tokens_per_correct(1000, 0), float("inf"))


class TestSummarize(unittest.TestCase):
    def test_basic_aggregation(self):
        results = [
            {"correct": True, "response_tokens": 100},
            {"correct": False, "response_tokens": 200},
            {"correct": True, "response_tokens": 150},
        ]
        summary = summarize(results)
        self.assertAlmostEqual(summary["pass@1_mean"], 2 / 3)
        self.assertEqual(summary["total_tokens"], 450)
        self.assertAlmostEqual(summary["tokens_per_correct"], 450 / 2)
        self.assertEqual(summary["n"], 3)

    def test_empty_results_does_not_crash(self):
        summary = summarize([])
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["pass@1_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
