"""Unit tests for crisp/training/gradient_surgery.py (asymmetric PC-Grad).

These tests are about VACS's "protect the RL direction" requirement:
when the RL and self-distillation gradients conflict, the asymmetric
projection must leave the RL gradient **bit-for-bit unchanged** and
project only the auxiliary (SD) gradient away from the original RL
direction. This is the structural difference from symmetric PC-Grad
(where both gradients are projected away from each other).

Tests hand-derive expected vectors in plain math rather than reusing
the same tensor ops the production code uses, since this is the kind
of code where a one-character bug (e.g. subtracting instead of adding)
silently corrupts training.
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
    from crisp.training.gradient_surgery import (
        AsymmetricPCGradResult,
        asymmetric_pcgrad_combine,
        flatten_grads,
        unflatten_grads,
    )
    # Sanity: legacy symmetric PC-Grad must still be importable from the
    # original module. The new asymmetric helpers must not have silently
    # changed the legacy algorithm.
    from crisp.training.pcgrad import pc_grad_combine, PCGradResult


def _make_params():
    # Two parameter tensors of different shapes, like LoRA's A/B matrices
    # across different modules -- exercises flatten/unflatten across more
    # than one tensor, not just one.
    return [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))]


def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestFlattenUnflattenRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_shape_and_values(self):
        params = _make_params()
        # Use distinct non-zero values so a silent zero-out is detectable.
        params[0].data = torch.tensor([1.5, -2.0])
        params[1].data = torch.tensor([3.0, 4.0, -5.0])
        grads = [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0, 50.0])]

        flat = flatten_grads(grads)
        self.assertEqual(flat.shape, (5,))
        self.assertTrue(torch.allclose(flat, torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])))

        rebuilt = unflatten_grads(flat, params)
        self.assertEqual(len(rebuilt), len(grads))
        for got, want in zip(rebuilt, grads):
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.allclose(got, want))

    def test_round_trip_with_three_params(self):
        params = [
            torch.nn.Parameter(torch.zeros(2)),
            torch.nn.Parameter(torch.zeros(4)),
            torch.nn.Parameter(torch.zeros(1)),
        ]
        grads = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0, 5.0, 6.0]),
            torch.tensor([7.0]),
        ]
        rebuilt = unflatten_grads(flatten_grads(grads), params)
        for got, want in zip(rebuilt, grads):
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.allclose(got, want))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestAsymmetricPCGradConflicting(unittest.TestCase):
    def test_rl_gradient_is_unchanged_under_conflict(self):
        # The defining property of VACS's asymmetric PC-Grad: g_rl itself
        # must NOT be modified. The combined output is `g_rl + lam *
        # g_sd_proj`, so each combined slot is `g_rl[i] + lam *
        # g_sd_proj[i]`. After projection, g_sd_proj is orthogonal to g_rl
        # -- but that does NOT mean combined == g_rl in every slot. In
        # this specific test, g_sd_proj slot 0 is [0, 0] (the projection
        # removed the conflicting direction entirely) so combined[0]
        # equals g_rl[0] there. Slot 1 has all-zero g_rl and g_sd_proj,
        # so combined[1] is all zeros.
        params = _make_params()
        # g_rl: [1, 0, 0, 0, 0] (only the first param has nonzero grad)
        # g_sd: [-1, 1, 0, 0, 0] -- dot(g_rl, g_sd) = -1 < 0 -> conflict
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])]

        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0, eps=1e-12)
        self.assertTrue(result.conflicted)
        # Hand-derived: g_sd_proj = g_sd - (dot/|g_rl|^2)*g_rl =
        #              [-1,1] - (-1)*[1,0] = [0,1]. So combined slot 0 is
        #              [1,0] + 1*[0,1] = [1, 1]. Combined slot 1 is all zeros.
        self.assertTrue(torch.allclose(result.combined[0], torch.tensor([1.0, 1.0]), atol=1e-5),
                        msg=f"got combined[0]={result.combined[0]}")
        self.assertTrue(torch.allclose(result.combined[1], torch.tensor([0.0, 0.0, 0.0])),
                        msg=f"got combined[1]={result.combined[1]}")
        # And the structural invariant the test is really checking: g_rl
        # would have been zero in slot 0 only if the helper had projected
        # it (which it must not). Confirm g_rl was preserved by checking
        # the auxiliary_projected is orthogonal to the ORIGINAL g_rl.
        self.assertAlmostEqual(
            torch.dot(_flat(result.auxiliary_projected), _flat(g_rl_list)).item(),
            0.0, places=5,
        )

    def test_asymmetric_projection_makes_auxiliary_orthogonal_to_original_rl(self):
        # After asymmetric projection, g_sd_proj must be orthogonal to the
        # ORIGINAL g_rl (not to a modified g_rl, since RL is untouched).
        params = _make_params()
        g_rl = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        g_sd = torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0])
        g_rl_list = [g_rl[:2], g_rl[2:]]
        g_sd_list = [g_sd[:2], g_sd[2:]]

        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0, eps=1e-12)
        g_sd_proj = _flat(result.auxiliary_projected)
        # Hand-computed: g_sd_proj = g_sd - (dot(g_rl, g_sd) / |g_rl|^2) * g_rl
        # dot = -1, |g_rl|^2 = 1 -> g_sd_proj = [-1,1,0,0,0] - (-1)*[1,0,0,0,0]
        #                              = [-1+1, 1, 0, 0, 0] = [0, 1, 0, 0, 0]
        expected = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(g_sd_proj, expected, atol=1e-5),
                        msg=f"got g_sd_proj={g_sd_proj} expected={expected}")
        # And orthogonal to g_rl (the original one -- this is what makes
        # the projection asymmetric, vs symmetric PC-Grad where both
        # gradients are modified).
        self.assertAlmostEqual(torch.dot(g_sd_proj, g_rl).item(), 0.0, places=5)

    def test_combined_gradient_is_g_rl_plus_lam_times_g_sd_projected(self):
        params = _make_params()
        g_rl = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        g_sd = torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0])
        g_rl_list = [g_rl[:2], g_rl[2:]]
        g_sd_list = [g_sd[:2], g_sd[2:]]
        lam = 0.5

        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=lam, eps=1e-12)
        # g_sd_proj = [0, 1, 0, 0, 0]
        # combined = g_rl + lam * g_sd_proj = [1, 0, 0, 0, 0] + 0.5 * [0, 1, 0, 0, 0]
        #          = [1, 0.5, 0, 0, 0]
        expected = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(_flat(result.combined), expected, atol=1e-5),
                        msg=f"got combined={_flat(result.combined)} expected={expected}")


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestAsymmetricPCGradAligned(unittest.TestCase):
    def test_keeps_auxiliary_unchanged_when_gradients_align(self):
        # Aligned gradients -> no projection -> g_sd_proj == g_sd.
        params = _make_params()
        g_rl = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        g_sd = torch.tensor([0.5, 0.25, 0.0, 0.0, 0.0])  # dot = 0.5 > 0 -> aligned
        g_rl_list = [g_rl[:2], g_rl[2:]]
        g_sd_list = [g_sd[:2], g_sd[2:]]

        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0, eps=1e-12)
        self.assertFalse(result.conflicted)
        expected = g_rl + g_sd
        self.assertTrue(torch.allclose(_flat(result.combined), expected, atol=1e-5))
        # auxiliary_projected == g_sd
        self.assertTrue(torch.allclose(_flat(result.auxiliary_projected), g_sd, atol=1e-5))

    def test_cos_sim_sign_matches_conflict_flag(self):
        params = _make_params()
        # Conflict case
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        self.assertTrue(result.conflicted)
        self.assertLess(result.cos_sim, 0.0)
        # Aligned case
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([0.5, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        self.assertFalse(result.conflicted)
        self.assertGreater(result.cos_sim, 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestAsymmetricPCGradEdgeCases(unittest.TestCase):
    def test_zero_norm_auxiliary_is_safe_and_finite(self):
        # If g_sd is the zero vector, conflict is False, cos_sim is well-defined
        # (eps in denominator), and the combined gradient equals g_rl.
        params = _make_params()
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        self.assertFalse(result.conflicted)
        self.assertAlmostEqual(result.sd_grad_norm, 0.0, places=6)
        self.assertAlmostEqual(result.cos_sim, 0.0, places=6)
        # combined = g_rl + lam * 0 = g_rl
        self.assertTrue(torch.allclose(_flat(result.combined), _flat(g_rl_list), atol=1e-6))

    def test_zero_norm_rl_keeps_auxiliary_as_is(self):
        # If g_rl is zero, no projection can happen (any dot is 0), and the
        # projection denominator |g_rl|^2 is 0 -> safe behavior is "no
        # projection", i.e. auxiliary passes through unchanged.
        params = _make_params()
        g_rl_list = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        # No conflict because the conflict flag is computed from the
        # sign of the dot product, and 0 < 0 is False.
        self.assertFalse(result.conflicted)
        # combined = g_rl + lam * g_sd = g_sd
        self.assertTrue(torch.allclose(_flat(result.combined), _flat(g_sd_list), atol=1e-6))

    def test_auxiliary_projected_can_be_zero_under_extreme_conflict(self):
        # If g_sd is exactly anti-parallel to g_rl with the same magnitude,
        # g_sd_proj collapses to zero.
        params = _make_params()
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        self.assertTrue(result.conflicted)
        # combined = g_rl + lam * 0 = g_rl
        self.assertTrue(torch.allclose(_flat(result.combined), _flat(g_rl_list), atol=1e-6))

    def test_does_not_mutate_input_grad_lists(self):
        # The helper takes already-separated gradient lists and must NOT
        # mutate them in place -- the caller owns the lists and may reuse
        # them (e.g. for logging or accumulated micro-batches).
        params = _make_params()
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])]
        rl_before = [g.clone() for g in g_rl_list]
        sd_before = [g.clone() for g in g_sd_list]
        _ = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        for got, want in zip(g_rl_list, rl_before):
            self.assertTrue(torch.allclose(got, want))
        for got, want in zip(g_sd_list, sd_before):
            self.assertTrue(torch.allclose(got, want))

    def test_does_not_mutate_param_grad(self):
        # Like pc_grad_combine, this helper must be side-effect free on
        # `param.grad` -- the caller assigns the combined result itself.
        params = _make_params()
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])]
        asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        for p in params:
            self.assertIsNone(p.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestAsymmetricPCGradMetrics(unittest.TestCase):
    def test_metrics_report_rl_and_sd_norms(self):
        params = _make_params()
        g_rl_list = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.0, 0.0])]  # |g_rl|=5
        g_sd_list = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]    # |g_sd|=0
        result = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)
        self.assertAlmostEqual(result.rl_grad_norm, 5.0, places=4)
        self.assertAlmostEqual(result.sd_grad_norm, 0.0, places=6)

    def test_lambda_scales_only_the_projected_auxiliary(self):
        # Doubling lambda should NOT change g_sd_proj -- it only scales
        # the auxiliary contribution to the combined vector.
        params = _make_params()
        g_rl_list = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
        g_sd_list = [torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])]

        r_low = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=0.25)
        r_high = asymmetric_pcgrad_combine(g_rl_list, g_sd_list, params, lam=1.0)

        self.assertTrue(torch.allclose(
            _flat(r_low.auxiliary_projected),
            _flat(r_high.auxiliary_projected),
            atol=1e-6,
        ))
        # Combined = g_rl + lam * g_sd_proj, so doubling lam changes
        # the combined proportionally (where the projected component is nonzero).
        expected_high_minus_low = (1.0 - 0.25) * _flat(r_low.auxiliary_projected)
        diff = _flat(r_high.combined) - _flat(r_low.combined)
        self.assertTrue(torch.allclose(diff, expected_high_minus_low, atol=1e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestLegacyPCGradUnchanged(unittest.TestCase):
    """The asymmetric helper is a NEW addition. The legacy symmetric
    pc_grad_combine must continue to project BOTH gradients away from
    each other (not just the auxiliary). These regression tests lock in
    that behavior so a future refactor can't silently switch algorithms.
    """

    def test_symmetric_pcgrad_projects_both_gradients_under_conflict(self):
        # Re-derived from the legacy code path: under conflict, BOTH
        # g_rl and g_sd are projected away from the *original* other
        # vector. This is what the asymmetric helper explicitly does NOT
        # do for g_rl.
        params = _make_params()
        a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # g_rl
        b = torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0])  # g_sd
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()
        loss_sd = (params[0] * b[:2]).sum() + (params[1] * b[2:]).sum()

        result = pc_grad_combine(loss_rl, loss_sd, params, lam=1.0)
        self.assertTrue(result.conflicted)
        flat = _flat(result.combined)
        # Hand-computed: dot=-1, |a|^2=1, |b|^2=2.
        # g_rl_proj = a - (dot/|b|^2)*b = [1, 0] - (-1/2)*[-1, 1]
        #           = [1, 0] - [0.5, -0.5] = [0.5, 0.5]
        # g_sd_proj = b - (dot/|a|^2)*a = [-1, 1] - (-1)*[1, 0]
        #           = [-1, 1] + [1, 0] = [0, 1]
        # combined = g_rl_proj + 1 * g_sd_proj = [0.5, 0.5] + [0, 1] = [0.5, 1.5]
        expected = torch.tensor([0.5, 1.5, 0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(flat, expected, atol=1e-5),
                        msg=f"symmetric pcgrad combined={flat} expected={expected}")
        # Critically, combined[0] must NOT be 1.0 (which would mean RL was
        # preserved): the asymmetric variant preserves RL; symmetric does
        # not. This is the structural difference the asymmetric helper
        # explicitly enforces.
        self.assertNotAlmostEqual(flat[0].item(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
