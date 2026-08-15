"""Unit tests for crisp/training/beta_controller.py.

CIBO v2's beta controller holds the running IB coefficient. Two
modes are documented:

  - `fixed`: beta never changes from `beta` regardless of observations.
  - `adaptive_sign`: during the warm-up phase the controller estimates
    the KL reference EMA; after warm-up, the controller computes a
    bounded sign-based update on each post-step observation.

The plan calls out specific properties that MUST hold:

  - In fixed mode, beta is never changed.
  - In adaptive mode, the reference EMA is frozen after warm-up.
  - Near-zero |baselined_kl| -> beta is held stable (no division by ~0).
  - Sign updates are bounded by `beta_step_size` and clipped to
    `[beta_min, beta_max]`.
  - State round-trips losslessly through state_dict / load_state_dict.
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
    from crisp.training.beta_controller import BetaController


def _make_controller(**kwargs):
    defaults = dict(
        beta=0.1,
        beta_mode="fixed",
        beta_min=0.001,
        beta_max=10.0,
        beta_step_size=0.05,
        beta_ema_decay=0.95,
        beta_reference_warmup_steps=20,
        beta_denominator_epsilon=1.0e-6,
    )
    defaults.update(kwargs)
    return BetaController(**defaults)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestFixedMode(unittest.TestCase):
    def test_fixed_mode_never_changes_beta(self):
        ctrl = _make_controller(beta_mode="fixed", beta=0.1)
        # Even after many observations and step() calls, beta must be 0.1.
        for _ in range(50):
            ctrl.observe(reward=0.0, kl_value=1.0)
            ctrl.step()
        self.assertAlmostEqual(ctrl.beta, 0.1, places=6)

    def test_fixed_mode_never_warms_up(self):
        # Fixed mode is a complete no-op: warmup_remaining is irrelevant
        # and reference_kl stays at its initial 0.
        ctrl = _make_controller(beta_mode="fixed", beta=0.1)
        for _ in range(1000):
            ctrl.observe(reward=0.0, kl_value=1.0)
            ctrl.step()
        # Reference stays at initial 0 -- fixed mode never builds one.
        self.assertEqual(ctrl.reference_kl, 0.0)
        # And beta is unchanged.
        self.assertAlmostEqual(ctrl.beta, 0.1, places=6)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestAdaptiveMode(unittest.TestCase):
    def test_adaptive_mode_freezes_reference_after_warmup(self):
        ctrl = _make_controller(
            beta_mode="adaptive_sign",
            beta_reference_warmup_steps=4,
        )
        # During warm-up: reference is updated each observation.
        for kl in [1.0, 2.0, 3.0, 4.0]:
            ctrl.observe(reward=0.0, kl_value=kl)
            ctrl.step()
        # After warm-up (4 steps): reference must NOT change.
        ref_after_warmup = ctrl.reference_kl
        for _ in range(10):
            ctrl.observe(reward=0.0, kl_value=99.0)
            ctrl.step()
        self.assertAlmostEqual(ctrl.reference_kl, ref_after_warmup, places=6)

    def test_near_zero_kl_delta_holds_beta_stable(self):
        ctrl = _make_controller(
            beta_mode="adaptive_sign",
            beta_reference_warmup_steps=2,
            beta_step_size=0.05,
        )
        # Warm up with kl=1.0 so the reference is locked at ~1.0.
        ctrl.observe(reward=0.0, kl_value=1.0)
        ctrl.step()
        ctrl.observe(reward=0.0, kl_value=1.0)
        ctrl.step()
        beta_before = ctrl.beta
        # Now observe kl values that are essentially equal to the
        # reference (delta -> 0). beta MUST stay exactly the same.
        for _ in range(20):
            ctrl.observe(reward=0.0, kl_value=1.0 + 1e-15)
            ctrl.step()
        self.assertAlmostEqual(ctrl.beta, beta_before, places=6)

    def test_positive_and_negative_sign_updates_are_bounded(self):
        # Two runs that differ ONLY in the sign of (kl - reference):
        # one with kl much larger than ref -> beta grows (up to beta_max).
        # one with kl much smaller than ref -> beta shrinks (down to beta_min).
        # In both, the per-step change is bounded by beta_step_size.
        def make():
            return _make_controller(
                beta_mode="adaptive_sign",
                beta_reference_warmup_steps=2,
                beta=0.1,
                beta_step_size=0.01,
                beta_min=0.01,
                beta_max=10.0,
            )
        # Growing branch
        ctrl_pos = make()
        ctrl_pos.observe(reward=0.0, kl_value=1.0); ctrl_pos.step()
        ctrl_pos.observe(reward=0.0, kl_value=1.0); ctrl_pos.step()
        beta_before = ctrl_pos.beta
        for _ in range(50):
            ctrl_pos.observe(reward=0.0, kl_value=100.0)
            ctrl_pos.step()
        self.assertGreater(ctrl_pos.beta, beta_before)
        # The per-step change is bounded: over N steps the total change
        # is bounded by N * beta_step_size in either direction.
        self.assertLessEqual(ctrl_pos.beta - beta_before, 50 * 0.01 + 1e-9)

        # Shrinking branch
        ctrl_neg = make()
        ctrl_neg.observe(reward=0.0, kl_value=1.0); ctrl_neg.step()
        ctrl_neg.observe(reward=0.0, kl_value=1.0); ctrl_neg.step()
        beta_before = ctrl_neg.beta
        for _ in range(50):
            ctrl_neg.observe(reward=0.0, kl_value=0.0001)
            ctrl_neg.step()
        self.assertLess(ctrl_neg.beta, beta_before)
        self.assertLessEqual(beta_before - ctrl_neg.beta, 50 * 0.01 + 1e-9)

    def test_beta_always_within_bounds(self):
        ctrl = _make_controller(
            beta_mode="adaptive_sign",
            beta_reference_warmup_steps=2,
            beta=0.1, beta_min=0.05, beta_max=2.0,
            beta_step_size=0.5,
        )
        ctrl.observe(reward=0.0, kl_value=1.0); ctrl.step()
        ctrl.observe(reward=0.0, kl_value=1.0); ctrl.step()
        # Hammer with extreme observations.
        for _ in range(200):
            ctrl.observe(reward=0.0, kl_value=1e6)
            ctrl.step()
        self.assertGreaterEqual(ctrl.beta, ctrl.beta_min - 1e-9)
        self.assertLessEqual(ctrl.beta, ctrl.beta_max + 1e-9)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestStateRoundTrip(unittest.TestCase):
    def test_state_round_trip_preserves_beta_and_emas(self):
        ctrl = _make_controller(
            beta_mode="adaptive_sign",
            beta_reference_warmup_steps=2,
        )
        ctrl.observe(reward=0.0, kl_value=2.0); ctrl.step()
        ctrl.observe(reward=0.0, kl_value=3.0); ctrl.step()
        snap = ctrl.state_dict()

        # New controller with the SAME beta bounds (the validator
        # enforces beta_min <= beta <= beta_max on construction), then
        # load the snapshot -- load_state_dict overwrites all of beta,
        # beta_mode, reference_kl, warmup_remaining, etc.
        new_ctrl = _make_controller(beta_mode="fixed", beta=0.5)
        new_ctrl.load_state_dict(snap)
        self.assertAlmostEqual(new_ctrl.beta, ctrl.beta, places=6)
        self.assertAlmostEqual(new_ctrl.reference_kl, ctrl.reference_kl, places=6)
        self.assertEqual(new_ctrl.beta_mode, "adaptive_sign")
        self.assertEqual(new_ctrl.warmup_remaining, ctrl.warmup_remaining)

    def test_metrics_expose_required_keys(self):
        # The plan calls out: cibo/beta, cibo/beta_mode,
        # cibo/beta_reference_kl, cibo/baselined_kl. Verify those are
        # surfaced via the controller's metrics dict.
        ctrl = _make_controller(beta_mode="adaptive_sign", beta_reference_warmup_steps=2)
        ctrl.observe(reward=0.0, kl_value=1.5); ctrl.step()
        ctrl.observe(reward=0.0, kl_value=2.5); ctrl.step()
        # Step past warmup so reference_kl is meaningful.
        ctrl.observe(reward=0.0, kl_value=2.0); ctrl.step()
        metrics = ctrl.metrics()
        for key in ("cibo/beta", "cibo/beta_mode", "cibo/beta_reference_kl",
                    "cibo/baselined_kl"):
            self.assertIn(key, metrics, msg=f"missing metric {key}")


if __name__ == "__main__":
    unittest.main()