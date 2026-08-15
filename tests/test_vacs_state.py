"""Unit tests for crisp/training/vacs_state.py.

VACS's adaptive mixing state must satisfy four properties the plan calls
out explicitly:

  1. Deterministic EMA update (no randomness, no time-based state).
  2. Multiplier clamping inside [multiplier_min, multiplier_max].
  3. State serialization round-trips losslessly.
  4. The rule that a micro-batch cannot update the coefficient used in
     its own optimizer step. This is the timing invariant: VACS state
     reads PRIOR-step norms to compute `lambda_eff`, then writes the
     CURRENT-step norm into a `pending` slot that is committed only
     after the optimizer step. The state object must enforce that
     separation by API, not just by documentation.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from crisp.training.vacs_state import (
        VacsAdaptiveState,
        compute_lambda_effective,
        observe_grad_stats,
    )


def _make_state(**kwargs):
    defaults = dict(
        ema_decay=0.95,
        multiplier_min=0.10,
        multiplier_max=5.0,
    )
    defaults.update(kwargs)
    return VacsAdaptiveState(**defaults)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestEmaUpdate(unittest.TestCase):
    def test_ema_update_matches_decay_formula(self):
        # ema_new = decay * ema_old + (1 - decay) * new_observation.
        # With default ema_decay=0.95 and an initial ema of 0:
        # first commit: 0.95 * 0 + 0.05 * 4 = 0.20
        # second commit: 0.95 * 0.20 + 0.05 * 9 = 0.19 + 0.45 = 0.64
        state = _make_state()
        state.observe(norm_sq=4.0, cos_value=0.0)
        state.commit(rl_grad_norm_sq=4.0, sd_grad_norm_sq=4.0, cos_value=0.0)
        self.assertAlmostEqual(state.rl_grad_norm_sq_ema, 0.20, places=6)
        self.assertAlmostEqual(state.cos_ema, 0.0, places=6)

        state.observe(norm_sq=9.0, cos_value=1.0)
        state.commit(rl_grad_norm_sq=9.0, sd_grad_norm_sq=9.0, cos_value=1.0)
        self.assertAlmostEqual(state.rl_grad_norm_sq_ema, 0.64, places=6)
        self.assertAlmostEqual(state.cos_ema, 0.95 * 0.0 + 0.05 * 1.0, places=6)

    def test_ema_decay_one_means_no_update(self):
        # decay=1 -> (1 - decay) = 0 -> the EMA formula's contribution
        # from any new observation is zero. The committed EMA therefore
        # stays at whatever value it had before the commit. With a fresh
        # state (initial ema = 0), a single commit leaves the ema at 0
        # regardless of the observation magnitude.
        state = _make_state(ema_decay=1.0)
        state.observe(norm_sq=2.0, cos_value=0.5)
        state.commit(rl_grad_norm_sq=2.0, sd_grad_norm_sq=2.0, cos_value=0.5)
        self.assertAlmostEqual(state.rl_grad_norm_sq_ema, 0.0, places=6,
                               msg="decay=1 freezes the EMA at its prior value (here, initial 0)")
        self.assertAlmostEqual(state.cos_ema, 0.0, places=6)

        # A subsequent commit also leaves the EMA unchanged.
        state.observe(norm_sq=99.0, cos_value=-1.0)
        state.commit(rl_grad_norm_sq=99.0, sd_grad_norm_sq=99.0, cos_value=-1.0)
        self.assertAlmostEqual(state.rl_grad_norm_sq_ema, 0.0, places=6)
        self.assertAlmostEqual(state.cos_ema, 0.0, places=6)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestMultiplierClamping(unittest.TestCase):
    def test_multiplier_clamped_to_range(self):
        state = _make_state(multiplier_min=0.10, multiplier_max=5.0)
        # Drive the EMAs to known values via direct assignment so the
        # test is not sensitive to the EMA decay rate. The structural
        # fact under test is "the multiplier is clipped to [min, max]".
        state.rl_grad_norm_sq_ema = 1e-6
        state.sd_grad_norm_sq_ema = 1.0
        state.cos_ema = 1.0  # -> cosine_factor = 1.0
        m = compute_lambda_effective(state, base_lambda=1.0)
        # variance_factor = sqrt(1e-6/1) = 1e-3 -> clipped to 0.10 (min)
        # lambda_eff = 1.0 * 0.10 * 1.0 = 0.10
        self.assertAlmostEqual(m, 0.10, places=5)

    def test_multiplier_inverse_clamped_at_max(self):
        state = _make_state(multiplier_min=0.10, multiplier_max=5.0)
        state.rl_grad_norm_sq_ema = 100.0
        state.sd_grad_norm_sq_ema = 1e-6
        state.cos_ema = 1.0  # -> cosine_factor = 1.0
        m = compute_lambda_effective(state, base_lambda=2.0)
        # variance_factor = sqrt(100/1e-6) ~ 1e4 -> clipped to 5.0 (max)
        # lambda_eff = 2.0 * 5.0 * 1.0 = 10.0
        self.assertAlmostEqual(m, 10.0, places=5)

    def test_multiplier_uses_min_when_rl_norm_zero(self):
        # Numerical guard: |g_rl|^2 = 0 should not produce NaN or a huge ratio.
        state = _make_state(multiplier_min=0.10, multiplier_max=5.0)
        state.rl_grad_norm_sq_ema = 0.0
        state.sd_grad_norm_sq_ema = 1.0
        state.cos_ema = 0.0  # -> cosine_factor = 0.5
        m = compute_lambda_effective(state, base_lambda=1.0)
        # sqrt(0/1) = 0 -> clipped to 0.10 (min)
        # lambda_eff = 1.0 * 0.10 * 0.5 = 0.05
        self.assertAlmostEqual(m, 0.05, places=5)

    def test_adaptive_mix_can_be_disabled(self):
        # When disabled, multiplier is exactly 1.0 -> lambda_eff = base_lambda.
        state = _make_state(multiplier_min=0.10, multiplier_max=5.0)
        # Disable at the boundary: state.disabled should suppress the adaptive math.
        state.disabled = True
        state.commit(rl_grad_norm_sq=4.0, sd_grad_norm_sq=1.0, cos_value=0.0)
        m = compute_lambda_effective(state, base_lambda=1.5)
        self.assertAlmostEqual(m, 1.5, places=6)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestObserveVsCommitTiming(unittest.TestCase):
    def test_observe_does_not_change_committed_value(self):
        # The plan's structural rule: a micro-batch cannot update the
        # coefficient used in its own optimizer step. `observe` writes
        # into the "pending" slot only; `compute_lambda_effective` must
        # continue to use the previously-committed value until commit()
        # is called explicitly (which the loop does AFTER optimizer.step).
        state = _make_state(ema_decay=0.5)
        # Initial commit: ema = some value
        state.commit(rl_grad_norm_sq=4.0, sd_grad_norm_sq=1.0, cos_value=0.0)
        rl_before = state.rl_grad_norm_sq_ema
        # Observe a wildly different norm (this micro-batch).
        state.observe(norm_sq=10000.0, cos_value=1.0)
        # The committed value MUST NOT have changed yet.
        self.assertAlmostEqual(state.rl_grad_norm_sq_ema, rl_before, places=6)
        # Calling compute_lambda_effective now should still use the
        # pre-observation committed value.
        m_before = compute_lambda_effective(state, base_lambda=1.0)
        # Now the optimizer step has happened; commit the observation.
        state.commit(rl_grad_norm_sq=10000.0, sd_grad_norm_sq=1.0, cos_value=1.0)
        m_after = compute_lambda_effective(state, base_lambda=1.0)
        # The two values MUST differ because the committed EMA changed.
        self.assertNotAlmostEqual(m_before, m_after, places=4)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestStateSerialization(unittest.TestCase):
    def test_state_dict_round_trips_losslessly(self):
        state = _make_state()
        # Build up some non-default state.
        state.observe(norm_sq=4.0, cos_value=0.5)
        state.commit(rl_grad_norm_sq=4.0, sd_grad_norm_sq=1.0, cos_value=0.5)
        snapshot = state.state_dict()

        # Reconstruct from snapshot
        restored = _make_state()
        restored.load_state_dict(snapshot)

        self.assertAlmostEqual(restored.rl_grad_norm_sq_ema, state.rl_grad_norm_sq_ema, places=6)
        self.assertAlmostEqual(restored.cos_ema, state.cos_ema, places=6)
        self.assertAlmostEqual(restored.disabled, state.disabled, places=6)
        self.assertEqual(restored.ema_decay, state.ema_decay)
        self.assertEqual(restored.multiplier_min, state.multiplier_min)
        self.assertEqual(restored.multiplier_max, state.multiplier_max)

    def test_state_dict_handles_disabled_flag(self):
        state = _make_state()
        state.disabled = True
        snapshot = state.state_dict()
        restored = _make_state()
        restored.load_state_dict(snapshot)
        self.assertTrue(restored.disabled)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestDefaultMultiplierFloor(unittest.TestCase):
    def test_min_above_max_is_rejected(self):
        # Sanity: bounds must be ordered (the central config validation
        # already enforces this; the state object re-checks defensively).
        with self.assertRaises(ValueError):
            _make_state(multiplier_min=5.0, multiplier_max=0.5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestObserveGradStatsHelper(unittest.TestCase):
    def test_helper_extracts_norm_sq_and_cos_from_gradients(self):
        g_rl = torch.tensor([3.0, 4.0])  # |g_rl|^2 = 25
        g_sd = torch.tensor([4.0, -3.0])  # |g_sd|^2 = 25, dot=0 -> cos=0
        norm_sq, cos = observe_grad_stats(g_rl, g_sd)
        self.assertAlmostEqual(norm_sq, 25.0, places=4)
        self.assertAlmostEqual(cos, 0.0, places=4)

    def test_helper_handles_zero_norm_gracefully(self):
        g_rl = torch.zeros(4)
        g_sd = torch.tensor([1.0, 2.0, 3.0, 4.0])
        norm_sq, cos = observe_grad_stats(g_rl, g_sd)
        self.assertAlmostEqual(norm_sq, 0.0, places=6)
        self.assertTrue(math.isfinite(cos))


if __name__ == "__main__":
    unittest.main()
