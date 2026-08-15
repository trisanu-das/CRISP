"""Unit tests for crisp/training/objectives.py.

The functions here are mathematically sensitive primitives shared by VACS-CRISP
and CIBO-CRISP v2. They are deliberately tiny and pure so each one can be
hand-verified against independent arithmetic (plain Python math.log / manual
softmax), rather than re-derived via the same tensor ops the production code
uses (which only proves the code agrees with itself).

Test style:
  - hand-computed expected values from independent math,
  - property-style coverage of positive/negative/zero advantages plus
    extreme log-prob gaps (where numerical issues are most likely to hide),
  - explicit detaching/finite-value guards, since a non-finite credit
    silently poisoning a gradient is exactly the kind of bug these helpers
    exist to prevent.
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
    from crisp.training.objectives import (
        cibo_token_advantages,
        normalize_advantages,
        reduce_gated_kl,
        soft_reward_gate,
        vacs_token_advantages,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestNormalizeAdvantages(unittest.TestCase):
    def test_zero_variance_returns_zero(self):
        # All-equal rewards -> zero std -> (r - mean) / (std + eps) = 0.
        r = torch.tensor([0.5, 0.5, 0.5])
        adv = normalize_advantages(r, eps=1e-8)
        self.assertEqual(adv.shape, r.shape)
        for v in adv.tolist():
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_known_mean_and_std_match_hand_math(self):
        # rewards = [1, 2, 3, 4, 5] -> mean=3, std (biased)=sqrt(2)
        r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        adv = normalize_advantages(r, eps=1e-8)
        mean = r.mean().item()
        std = r.std(unbiased=False).item()
        expected = ((r - mean) / (std + 1e-8)).tolist()
        for got, want in zip(adv.tolist(), expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_output_dtype_matches_input(self):
        r = torch.tensor([0.0, 1.0], dtype=torch.float64)
        adv = normalize_advantages(r, eps=1e-8)
        self.assertEqual(adv.dtype, torch.float64)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestVacsTokenAdvantages(unittest.TestCase):
    def test_zero_credit_returns_original_advantage(self):
        # teacher and student log-probs identical -> credit = 0 ->
        # weight = clip(exp(0), 1-eps, 1+eps) = 1 -> token advantage == advantage.
        adv = torch.tensor([0.7])
        t = torch.tensor([[0.0, 0.0]])   # shape [1, 2]
        s = torch.tensor([[0.0, 0.0]])
        token_adv = vacs_token_advantages(adv, t, s, credit_mix=0.25, clip_epsilon=0.2)
        self.assertEqual(token_adv.shape, (1, 2))
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, 0.7, places=5)

    def test_positive_credit_pushes_token_advantage_toward_teacher(self):
        # teacher has strictly higher log-prob than student for every token,
        # so credit > 0 and weight > 1, so token advantage > global advantage.
        adv = torch.tensor([0.5])
        t = torch.tensor([[1.0, 1.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = vacs_token_advantages(adv, t, s, credit_mix=1.0, clip_epsilon=0.2)
        # credit = sg[log p_T - log p_S] = 1.0 (detached)
        # weight = clip(exp(1.0), 1-eps, 1+eps) = clip(e, 0.8, 1.2) = 1.2
        # token_adv = 0.5 * [(1-1) + 1 * 1.2] = 0.5 * 1.2 = 0.6
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, 0.6, places=5)

    def test_weight_is_clipped_within_one_minus_eps_one_plus_eps(self):
        # Extreme credit gaps must be clipped, not allowed to blow up.
        adv = torch.tensor([1.0])
        t = torch.tensor([[10.0, 10.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = vacs_token_advantages(adv, t, s, credit_mix=1.0, clip_epsilon=0.2)
        # weight = clip(exp(10), 0.8, 1.2) = 1.2
        # token_adv = 1.0 * [(1-1) + 1 * 1.2] = 1.2
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, 1.2, places=5)

    def test_negative_advantage_with_positive_credit_yields_smaller_magnitude(self):
        # VACS uses sign(advantage) inside the exponential. For a negative
        # trajectory advantage and positive teacher credit, the weight is
        # below one, so a teacher-endorsed token is penalized less strongly.
        adv = torch.tensor([-0.5])
        t = torch.tensor([[1.0, 1.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = vacs_token_advantages(adv, t, s, credit_mix=1.0, clip_epsilon=0.2)
        # sign(a)=-1, so weight=clip(exp(-1), 0.8, 1.2)=0.8.
        # token_adv = -0.5 * 0.8 = -0.4.
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, -0.4, places=5)

    def test_credit_mix_zero_disables_token_credit(self):
        # rho = 0 -> token advantage == global advantage.
        adv = torch.tensor([0.3])
        t = torch.tensor([[1.0, 1.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = vacs_token_advantages(adv, t, s, credit_mix=0.0, clip_epsilon=0.2)
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, 0.3, places=5)

    def test_credit_weight_is_fully_stop_gradient(self):
        t_param = torch.nn.Parameter(torch.tensor([[1.0, 1.0]]))
        s_param = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
        adv = torch.tensor([0.5])
        token_adv = vacs_token_advantages(
            adv, t_param, s_param, credit_mix=1.0, clip_epsilon=0.2
        )
        # The specification uses sg[log p_T - log p_S]. The credit weight is
        # therefore detached from both contexts; the policy gradient flows
        # separately through the explicit response-token log-probabilities.
        self.assertFalse(token_adv.requires_grad)
        self.assertIsNone(t_param.grad)
        self.assertIsNone(s_param.grad)

    def test_nonfinite_credit_rejected_with_context(self):
        # NaN in teacher log-probs -> the credit becomes NaN -> the helper
        # MUST raise rather than silently produce NaN gradients.
        adv = torch.tensor([0.5])
        t = torch.tensor([[float("nan"), 1.0]])
        s = torch.tensor([[0.0, 0.0]])
        with self.assertRaises(ValueError) as cm:
            vacs_token_advantages(adv, t, s, credit_mix=1.0, clip_epsilon=0.2)
        self.assertIn("credit", str(cm.exception).lower())


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestCiboTokenAdvantages(unittest.TestCase):
    def test_credit_lambda_zero_returns_original_advantage(self):
        adv = torch.tensor([0.7])
        t = torch.tensor([[1.0, 1.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = cibo_token_advantages(adv, t, s, credit_lambda=0.0)
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, 0.7, places=5)

    def test_tanh_credit_never_changes_nonzero_advantage_sign(self):
        # For nonzero advantage, tanh credit is bounded in (-1, 1) and the
        # lambda is in [0, 1] -> (1 + lambda * tanh(c)) > 0 -> token
        # advantage must have the SAME sign as the global advantage.
        for a_val, t_val, s_val in [(0.5, 1.0, 0.0), (-0.5, 1.0, 0.0), (0.5, 0.0, 1.0), (-0.5, 0.0, 1.0)]:
            adv = torch.tensor([a_val])
            t = torch.tensor([[t_val, t_val]])
            s = torch.tensor([[s_val, s_val]])
            token_adv = cibo_token_advantages(adv, t, s, credit_lambda=0.25)
            got_sign = (token_adv[0, 0].item() >= 0)
            want_sign = (a_val >= 0)
            self.assertEqual(got_sign, want_sign,
                             msg=f"sign flip for a={a_val} t={t_val} s={s_val}: "
                                 f"got token_adv={token_adv[0,0].item()}, want sign {want_sign}")

    def test_tanh_credit_matches_hand_math(self):
        # tanh(c) = tanh(log p_T - log p_S). With t=1, s=0 -> c=1 ->
        # tanh(1) ~ 0.761594. lambda=0.25 -> token_adv = a*(1 + 0.25*tanh(1)).
        adv = torch.tensor([0.5])
        t = torch.tensor([[1.0, 1.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = cibo_token_advantages(adv, t, s, credit_lambda=0.25)
        expected = 0.5 * (1.0 + 0.25 * math.tanh(1.0))
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, expected, places=5)

    def test_extreme_credit_gap_still_bounded_by_lambda(self):
        # Even with an enormous log-prob gap, tanh saturates -> token
        # advantage is bounded by a*(1+lambda).
        adv = torch.tensor([1.0])
        t = torch.tensor([[100.0, 100.0]])
        s = torch.tensor([[0.0, 0.0]])
        token_adv = cibo_token_advantages(adv, t, s, credit_lambda=0.25)
        # tanh(100) ~ 1, so token_adv ~ 1 * (1 + 0.25 * 1) = 1.25
        for v in token_adv.tolist():
            for x in v:
                self.assertAlmostEqual(x, 1.25, places=4)

    def test_credit_weight_is_fully_stop_gradient(self):
        t_param = torch.nn.Parameter(torch.tensor([[1.0, 1.0]]))
        s_param = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
        adv = torch.tensor([0.5])
        token_adv = cibo_token_advantages(
            adv, t_param, s_param, credit_lambda=0.25
        )
        self.assertFalse(token_adv.requires_grad)
        self.assertIsNone(t_param.grad)
        self.assertIsNone(s_param.grad)

    def test_lambda_above_unit_interval_is_rejected(self):
        # Documented: 0 <= lambda_c <= 1. Out-of-range lambda must fail
        # loudly at config time, not silently saturate.
        with self.assertRaises(ValueError):
            cibo_token_advantages(torch.tensor([0.5]), torch.tensor([[1.0]]), torch.tensor([[0.0]]), credit_lambda=1.5)
        with self.assertRaises(ValueError):
            cibo_token_advantages(torch.tensor([0.5]), torch.tensor([[1.0]]), torch.tensor([[0.0]]), credit_lambda=-0.1)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestSoftRewardGate(unittest.TestCase):
    def test_gate_is_monotonic_in_reward(self):
        # Higher reward -> larger sigmoid argument -> larger gate complement
        # -> SMALLER weight on the SD term (down-weighting distillation
        # when the student already gets it right).
        # Use a gentle slope so the test stays out of sigmoid saturation
        # and adjacent entries differ measurably.
        rewards = torch.tensor([-5.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        g = soft_reward_gate(rewards, slope=1.0, threshold=0.5)
        # gate = 1 - sigmoid(slope * (reward - threshold))
        expected = [1.0 - 1.0 / (1.0 + math.exp(-1.0 * (r - 0.5))) for r in rewards.tolist()]
        for got, want in zip(g.tolist(), expected):
            self.assertAlmostEqual(got, want, places=5)
        # Strict monotonicity across the gentle-slope range.
        for a, b in zip(g.tolist(), g.tolist()[1:]):
            self.assertGreater(a, b, msg=f"non-monotonic at boundary {a} -> {b}")

    def test_gate_downweights_high_reward(self):
        # Reward at the threshold -> gate = 1 - sigmoid(0) = 0.5.
        # Reward far above threshold -> gate -> 0 (distillation off).
        # Reward far below threshold -> gate -> 1 (full distillation).
        r_low = torch.tensor([-5.0])
        r_at = torch.tensor([0.5])
        r_high = torch.tensor([5.0])
        g_low = soft_reward_gate(r_low, slope=12.0, threshold=0.5).item()
        g_at = soft_reward_gate(r_at, slope=12.0, threshold=0.5).item()
        g_high = soft_reward_gate(r_high, slope=12.0, threshold=0.5).item()
        self.assertAlmostEqual(g_at, 0.5, places=5)
        self.assertGreater(g_low, g_at)
        self.assertGreater(g_at, g_high)

    def test_steeper_slope_makes_gate_sharper(self):
        r = torch.tensor([0.7])
        g_mild = soft_reward_gate(r, slope=1.0, threshold=0.5).item()
        g_steep = soft_reward_gate(r, slope=20.0, threshold=0.5).item()
        # Above-threshold reward -> gate below 0.5; steeper slope pushes it
        # closer to 0.
        self.assertLess(g_steep, g_mild)
        self.assertGreater(g_steep, 0.0)
        self.assertLess(g_mild, 0.5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestReduceGatedKL(unittest.TestCase):
    def test_sequence_sum_sums_per_token_kl(self):
        # Sequence-sum reduction: total KL = sum over tokens, mean over batch.
        per_token = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
        gate = torch.tensor([1.0, 1.0])
        out = reduce_gated_kl(per_token, gate, reduction="sequence_sum")
        # sum1 = 0.6, sum2 = 0.9, mean = (0.6 + 0.9) / 2 = 0.75
        self.assertAlmostEqual(out.item(), 0.75, places=6)

    def test_token_mean_averages_per_token_kl(self):
        # Token-mean reduction: per-example mean -> mean over batch.
        per_token = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
        gate = torch.tensor([1.0, 1.0])
        out = reduce_gated_kl(per_token, gate, reduction="token_mean")
        # mean1 = 0.2, mean2 = 0.45, mean = (0.2 + 0.45) / 2 = 0.325
        self.assertAlmostEqual(out.item(), 0.325, places=6)

    def test_gate_zeros_out_an_example(self):
        # gate[i] = 0 -> that example contributes zero to the per-example
        # sum/mean, so its contribution to the batch mean is also zero
        # (we include every example in the average -- not skip them --
        # which is the consistent behaviour for both the soft-gate case
        # and the hard-gate case documented here).
        per_token = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
        gate = torch.tensor([1.0, 0.0])
        out_sum = reduce_gated_kl(per_token, gate, reduction="sequence_sum")
        out_mean = reduce_gated_kl(per_token, gate, reduction="token_mean")
        # Row 0 contributes 0.6 (sum) and 0.2 (mean); row 1 contributes 0 (gate=0).
        # batch mean over both rows:
        self.assertAlmostEqual(out_sum.item(), 0.6 / 2, places=6)
        self.assertAlmostEqual(out_mean.item(), 0.2 / 2, places=6)

    def test_sequence_sum_and_token_mean_are_distinct_reductions(self):
        # For unequal-length responses, the two reductions MUST give
        # different results -- otherwise they are not separate reductions.
        per_token = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])]
        gate = torch.tensor([1.0, 1.0])
        out_sum = reduce_gated_kl(per_token, gate, reduction="sequence_sum").item()
        out_mean = reduce_gated_kl(per_token, gate, reduction="token_mean").item()
        self.assertNotAlmostEqual(out_sum, out_mean, places=4)

    def test_invalid_reduction_rejected(self):
        per_token = [torch.tensor([0.1, 0.2])]
        gate = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            reduce_gated_kl(per_token, gate, reduction="mean_over_examples")

    def test_empty_token_row_returns_zero(self):
        per_token = [torch.tensor([0.1, 0.2]), torch.zeros(0), torch.tensor([0.4])]
        gate = torch.tensor([1.0, 1.0, 1.0])
        out = reduce_gated_kl(per_token, gate, reduction="sequence_sum")
        # row1 sum=0.3, row2 sum=0.0 (empty), row3 sum=0.4 -> mean=0.7/3
        self.assertAlmostEqual(out.item(), (0.3 + 0.0 + 0.4) / 3, places=6)

    def test_nonfinite_per_token_kl_rejected(self):
        per_token = [torch.tensor([float("nan"), 0.2])]
        gate = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            reduce_gated_kl(per_token, gate, reduction="sequence_sum")


if __name__ == "__main__":
    unittest.main()
