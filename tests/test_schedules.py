from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.training.schedules import lambda_schedule  # noqa: E402


class TestLambdaSchedule(unittest.TestCase):
    def test_starts_at_lambda_max(self):
        self.assertAlmostEqual(lambda_schedule(0, 1000, 1.0), 1.0, places=6)

    def test_reaches_zero_at_total_steps(self):
        self.assertAlmostEqual(lambda_schedule(1000, 1000, 1.0), 0.0, places=6)

    def test_matches_cosine_formula_at_midpoint(self):
        # lambda(T/2) = lambda_max * cos(pi/4) = lambda_max / sqrt(2)
        result = lambda_schedule(500, 1000, 1.0)
        self.assertAlmostEqual(result, 1.0 / math.sqrt(2), places=6)

    def test_scales_with_lambda_max(self):
        self.assertAlmostEqual(lambda_schedule(0, 1000, 2.5), 2.5, places=6)
        self.assertAlmostEqual(lambda_schedule(500, 1000, 2.5), 2.5 / math.sqrt(2), places=6)

    def test_monotonically_decreasing(self):
        steps = [0, 100, 250, 500, 750, 900, 1000]
        values = [lambda_schedule(s, 1000, 1.0) for s in steps]
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(a, b)

    def test_never_negative_past_total_steps(self):
        # Without clipping, cos(pi * t / (2T)) goes negative for t > T --
        # this is exactly the bug the clipping guards against.
        for step in [1000, 1500, 2000, 10_000]:
            self.assertGreaterEqual(lambda_schedule(step, 1000, 1.0), 0.0)

    def test_negative_step_treated_as_zero(self):
        self.assertAlmostEqual(lambda_schedule(-5, 1000, 1.0), lambda_schedule(0, 1000, 1.0), places=6)

    def test_zero_total_steps_returns_zero_not_a_crash(self):
        self.assertEqual(lambda_schedule(0, 0, 1.0), 0.0)

    def test_zero_lambda_max_is_always_zero(self):
        for step in [0, 100, 1000]:
            self.assertEqual(lambda_schedule(step, 1000, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
