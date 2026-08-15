"""
Unit tests for crisp/training/pcgrad.py.

Requires torch (skipped automatically if it isn't installed -- this sandbox
used to develop the codebase doesn't have it; run this for real once torch
is available, e.g. `pip install -r requirements.txt && python -m pytest
tests/test_pcgrad.py -v`).

Losses below are deliberately simple linear functions of the parameters
(`loss = (param * constant_vector).sum()`), so `d(loss)/d(param)` is exactly
`constant_vector` -- this lets every expected gradient be written down by
hand and checked exactly, the same way the projection formula itself was
hand-verified with a throwaway numpy script before this module was written.
"""
from __future__ import annotations

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
    from crisp.training.pcgrad import pc_grad_combine


def _make_params():
    # Two parameter tensors of different shapes, like LoRA's A/B matrices
    # across different modules -- exercises flatten/unflatten across more
    # than one tensor, not just one.
    return [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))]


def _combined_flat(result):
    return torch.cat([g.reshape(-1) for g in result.combined])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestPCGradCombine(unittest.TestCase):
    def test_conflicting_gradients_are_projected_orthogonal_to_originals(self):
        params = _make_params()
        a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])   # g_rl, padded to full param width
        b = torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0])  # g_sd; dot(a, b) = -1 < 0 -> conflict
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()
        loss_sd = (params[0] * b[:2]).sum() + (params[1] * b[2:]).sum()

        result = pc_grad_combine(loss_rl, loss_sd, params, lam=1.0)

        self.assertTrue(result.conflicted)
        dot = torch.dot(a, b)
        g_rl_proj = a - (dot / b.norm() ** 2) * b
        g_sd_proj = b - (dot / a.norm() ** 2) * a
        expected = g_rl_proj + 1.0 * g_sd_proj

        self.assertTrue(torch.allclose(_combined_flat(result), expected, atol=1e-5))
        # The defining property of PC-Grad: each projected vector is
        # orthogonal to the *original* other vector, not the projected one.
        self.assertAlmostEqual(torch.dot(g_rl_proj, b).item(), 0.0, places=4)
        self.assertAlmostEqual(torch.dot(g_sd_proj, a).item(), 0.0, places=4)

    def test_non_conflicting_gradients_pass_through_unmodified(self):
        params = _make_params()
        a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0])  # dot = 1 > 0 -> no conflict
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()
        loss_sd = (params[0] * b[:2]).sum() + (params[1] * b[2:]).sum()

        result = pc_grad_combine(loss_rl, loss_sd, params, lam=1.0)

        self.assertFalse(result.conflicted)
        expected = a + 1.0 * b
        self.assertTrue(torch.allclose(_combined_flat(result), expected, atol=1e-5))

    def test_lambda_scales_only_the_sd_component(self):
        params = _make_params()
        a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0])
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()
        loss_sd = (params[0] * b[:2]).sum() + (params[1] * b[2:]).sum()

        result = pc_grad_combine(loss_rl, loss_sd, params, lam=0.25)
        expected = a + 0.25 * b
        self.assertTrue(torch.allclose(_combined_flat(result), expected, atol=1e-5))

    def test_loss_sd_none_returns_g_rl_alone(self):
        params = _make_params()
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()

        result = pc_grad_combine(loss_rl, None, params, lam=1.0)

        self.assertFalse(result.conflicted)
        self.assertEqual(result.sd_grad_norm, 0.0)
        self.assertTrue(torch.allclose(_combined_flat(result), a, atol=1e-5))

    def test_parameter_unused_by_a_loss_contributes_zero_not_a_crash(self):
        # params[1] never appears in loss_rl at all -> torch.autograd.grad
        # returns None for it (allow_unused=True); grad_utils.grad_or_zeros
        # must turn that into a real zero tensor of the right shape.
        params = _make_params()
        loss_rl = (params[0] * torch.tensor([2.0, 3.0])).sum()
        loss_sd = (params[1] * torch.tensor([1.0, 1.0, 1.0])).sum()

        result = pc_grad_combine(loss_rl, loss_sd, params, lam=1.0)

        self.assertEqual(result.combined[0].shape, params[0].shape)
        self.assertEqual(result.combined[1].shape, params[1].shape)
        self.assertTrue(torch.allclose(result.combined[0], torch.tensor([2.0, 3.0])))

    def test_cos_sim_sign_matches_conflict_flag(self):
        params = _make_params()
        a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        b_conflict = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0])
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()
        loss_sd = (params[0] * b_conflict[:2]).sum() + (params[1] * b_conflict[2:]).sum()
        result = pc_grad_combine(loss_rl, loss_sd, params, lam=1.0)
        self.assertTrue(result.conflicted)
        self.assertLess(result.cos_sim, 0.0)

    def test_does_not_mutate_param_grad(self):
        # pc_grad_combine must be side-effect free on `.grad` -- the caller
        # (training/common_loop.py) is responsible for assignment, which is
        # what makes this function unit-testable without a real optimizer.
        params = _make_params()
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_rl = (params[0] * a[:2]).sum() + (params[1] * a[2:]).sum()
        pc_grad_combine(loss_rl, None, params, lam=1.0)
        for p in params:
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
