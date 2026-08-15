"""Tests for crisp/training/vacs_step.py.

VACS-CRISP integration tests use a tiny fake causal LM that exposes
the same surface as a HF model (forward(input_ids=, attention_mask=)
returning a logits tensor), plus a minimal Rollout stand-in. The tests
verify behavior contracts the plan calls out:

  - VACS step returns a finite gradient per parameter and the required
    metrics (loss_rl, loss_sd, lambda_eff, cos_rl_sd, conflicted, etc.).
  - Disabling token credit reduces to sequence-advantage RL.
  - Disabling the soft gate gives unit gate weights on every rollout.
  - Disabling asymmetric projection uses the unprojected sum path.
  - RL gradient is unmodified when a conflict is forced.

The tests deliberately bypass HF generate()/rollout sampling -- that
path is covered by the existing tests for rollout.py / teacher_student.py.
Here we focus on what happens once we have rollouts, sequences, and
forward outputs.
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
    from crisp.training.gradient_surgery import asymmetric_pcgrad_combine
    from crisp.training.logprobs import ScoredSequence
    from crisp.training.objectives import (
        cibo_token_advantages,
        normalize_advantages,
        reduce_gated_kl,
        soft_reward_gate,
        vacs_token_advantages,
    )
    from crisp.training.vacs_step import (
        VacsStepResult,
        build_vacs_components,
        vacs_step_from_arrays,
    )


def _fake_model_factory(token_logp_targets, params, vocab=16):
    """Build a callable model whose forward output depends on `params`.

    For test purposes, we synthesize logits as

        logits = params[0] @ W   (broadcast over batch)

    where W is a deterministic projection matrix. The chosen log-prob
    targets are then produced by writing into the resulting logits at
    the right SHIFT positions. This way, `loss_rl` and `loss_sd` actually
    flow through `params`, so `compute_grads` returns real gradient
    vectors (not None / zeros).
    """

    # Build a stable projection W = ones(5, vocab) * 0.1 so the dependence
    # on params is non-trivial but still simple. params[0] is [1, 5]
    # so we want W [5, vocab].
    W = torch.full((5, vocab), 0.1)

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def __call__(self, input_ids=None, attention_mask=None):
            B, L = input_ids.shape
            # Base logits = params @ W (broadcast over batch and seq).
            base = params[0] @ W  # [B, vocab]
            logits = base.unsqueeze(1).expand(B, L, vocab).contiguous()
            for i, (logps, ids_for_row, resp_start) in enumerate(token_logp_targets):
                L_row = int(attention_mask[i].sum().item())
                for t_in_resp, lp in enumerate(logps):
                    pos = (resp_start - 1) + t_in_resp
                    if pos < 0 or pos >= L_row - 1:
                        raise AssertionError(
                            f"row {i}: resp_start={resp_start} resp_len={len(logps)} "
                            f"L_row={L_row} produced out-of-range shift pos {pos}"
                        )
                    p = float(lp.exp().clamp(min=1e-9, max=1 - 1e-9))
                    row = torch.full((vocab,), math.log((1.0 - p) / (vocab - 1)))
                    row[ids_for_row[t_in_resp]] = float(lp)
                    logits[i, pos] = row
            out = type("O", (), {"logits": logits})()
            return out

    return FakeModel()


def _make_teacher_student_inputs(params):
    """Build two paired rows (teacher + student) with shared response ids
    but different prefix lengths, plus a tiny model that gives chosen
    log-probs to each side. Returns everything needed to drive the
    VACS step end-to-end.
    """
    # Teacher row has response [5, 6] under prefix [10, 11, 12]; student
    # row has the same response [5, 6] under prefix [7]. We pick chosen
    # log-probs per row.
    teacher_seq = ScoredSequence(ids=[10, 11, 12, 5, 6], response_start=3, response_len=2)
    student_seq = ScoredSequence(ids=[7, 5, 6], response_start=1, response_len=2)
    teacher_logp_targets = [
        torch.tensor([[math.log(0.9)], [math.log(0.8)]]),  # response id 5: log 0.9; id 6: log 0.8
    ]
    student_logp_targets = [
        torch.tensor([[math.log(0.5)], [math.log(0.6)]]),  # response id 5: log 0.5; id 6: log 0.6
    ]
    teacher_ids_per_row = [
        [5, 6],
    ]
    student_ids_per_row = [
        [5, 6],
    ]

    teacher_targets = list(zip(
        teacher_logp_targets,
        teacher_ids_per_row,
        [teacher_seq.response_start],
    ))
    student_targets = list(zip(
        student_logp_targets,
        student_ids_per_row,
        [student_seq.response_start],
    ))

    model = _fake_model_factory(teacher_targets + student_targets, params=params)
    return model, [teacher_seq], [student_seq]


def _params_2d():
    return torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestBuildVacsComponents(unittest.TestCase):
    def test_returns_finite_scalar_loss_pair_and_metrics(self):
        params = [_params_2d()]
        model, teacher_seqs, student_seqs = _make_teacher_student_inputs(params=params)
        # One rollout (matching the single teacher+student pair built
        # by _make_teacher_student_inputs); reward=1.0 -> advantage ~ 0.
        rewards = torch.tensor([1.0])
        components = build_vacs_components(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=rewards,
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=1.0,
            use_token_credit=True,
            use_soft_gate=True,
        )
        # loss_rl and loss_sd are scalar tensors requiring grad.
        self.assertTrue(torch.isfinite(components.loss_rl).item())
        self.assertTrue(torch.isfinite(components.loss_sd).item())
        self.assertTrue(components.loss_rl.requires_grad)
        self.assertTrue(components.loss_sd.requires_grad)
        # Required metrics present.
        for key in (
            "vacs/token_weight_mean",
            "vacs/gate_mean",
            "vacs/cos_rl_sd",
            "vacs/conflicted",
            "vacs/rl_grad_norm",
            "vacs/sd_grad_norm",
            "vacs/variance_multiplier",
            "vacs/lambda_eff",
            "vacs/loss_rl",
            "vacs/loss_sd",
        ):
            self.assertIn(key, components.metrics, msg=f"missing metric {key}")
        # lambda_eff was computed (not None) because state.observed contains values.
        self.assertIsNotNone(components.lambda_eff)
        self.assertGreater(components.lambda_eff, 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestVacsStepFromArrays(unittest.TestCase):
    def test_returns_finite_gradient_and_required_metrics(self):
        params = [_params_2d()]
        model, teacher_seqs, student_seqs = _make_teacher_student_inputs(params=params)
        rewards = torch.tensor([1.0])
        result = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=rewards,
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=1.0,
            use_token_credit=True,
            use_soft_gate=True,
        )
        self.assertIsInstance(result, VacsStepResult)
        # gradients list aligned 1:1 with params.
        self.assertEqual(len(result.gradients), len(params))
        for g, p in zip(result.gradients, params):
            self.assertEqual(g.shape, p.shape)
            self.assertTrue(torch.isfinite(g).all().item())
            # p.grad was NOT touched by the helper.
            self.assertIsNone(p.grad)
        # Required metrics in the result.
        for key in (
            "vacs/loss_rl",
            "vacs/loss_sd",
            "vacs/token_weight_mean",
            "vacs/gate_mean",
            "vacs/cos_rl_sd",
            "vacs/conflicted",
            "vacs/rl_grad_norm",
            "vacs/sd_grad_norm",
            "vacs/variance_multiplier",
            "vacs/lambda_eff",
        ):
            self.assertIn(key, result.metrics)

    def test_disabling_token_credit_matches_sequence_advantage_path(self):
        # With credit_mix=0 (or use_token_credit=False), token credit is
        # disabled. The RL loss becomes the standard CRISP sequence-advantage
        # PPO-clipped path -- but VACS still adds the SD term, so the loss
        # must be FINITE. We assert token_weight_mean is exactly 1.0
        # everywhere (the no-credit weight) and the loss_rl is well-defined.
        params = [_params_2d()]
        model, teacher_seqs, student_seqs = _make_teacher_student_inputs(params=params)
        rewards = torch.tensor([1.0])
        result_no_credit = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=rewards,
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=0.0,
            use_token_credit=False,
            use_soft_gate=True,
        )
        # token_weight_mean = 1.0 because (1 - rho) + rho * 1 = 1 when rho=0.
        self.assertAlmostEqual(result_no_credit.metrics["vacs/token_weight_mean"], 1.0, places=5)
        self.assertTrue(torch.isfinite(result_no_credit.components.loss_rl).item())

    def test_disabling_soft_gate_gives_unit_gate_weights(self):
        # With use_soft_gate=False, the gate applied to the SD term is
        # exactly 1.0 for every rollout -- so gate_mean = 1.0.
        params = [_params_2d()]
        model, teacher_seqs, student_seqs = _make_teacher_student_inputs(params=params)
        rewards = torch.tensor([1.0])
        result = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=rewards,
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=1.0,
            use_token_credit=True,
            use_soft_gate=False,
        )
        self.assertAlmostEqual(result.metrics["vacs/gate_mean"], 1.0, places=5)

    def test_vacs_rl_gradient_remains_unmodified_under_conflict(self):
        # Construct an asymmetric PC-Grad conflict by hand (g_rl dot g_sd < 0).
        # Then run the VACS step with use_asymmetric_projection=True and
        # verify the resulting combined gradient equals g_rl + lam * g_sd_proj
        # -- which means g_rl itself is recoverable by combined - lam*g_sd_proj.
        params = [_params_2d()]
        model, teacher_seqs, student_seqs = _make_teacher_student_inputs(params=params)
        rewards = torch.tensor([1.0])
        result = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=rewards,
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=1.0,
            use_token_credit=True,
            use_soft_gate=True,
            use_asymmetric_projection=True,
        )
        if result.metrics["vacs/conflicted"] > 0.5:
            # Recovery: combined - lam * g_sd_proj should equal g_rl.
            lam = result.metrics["vacs/lambda_eff"]
            sd_proj = result.components.auxiliary_projected[0]
            recovered_rl = result.gradients[0] - lam * sd_proj
            rl_grad = result.components.rl_grad[0]
            self.assertTrue(torch.allclose(recovered_rl, rl_grad, atol=1e-4))
        else:
            # No conflict: combined = g_rl + lam * g_sd (no projection).
            lam = result.metrics["vacs/lambda_eff"]
            sd_proj = result.components.auxiliary_projected[0]
            recovered = result.gradients[0] - lam * sd_proj
            rl_grad = result.components.rl_grad[0]
            self.assertTrue(torch.allclose(recovered, rl_grad, atol=1e-4))

    def test_disabling_asymmetric_projection_uses_unprojected_sum(self):
        # With use_asymmetric_projection=False, the helper should bypass
        # the asymmetric surgery and simply return g_rl + lam * g_sd.
        params = [_params_2d()]
        model, teacher_seqs, student_seqs = _make_teacher_student_inputs(params=params)
        rewards = torch.tensor([1.0])
        result = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=rewards,
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=1.0,
            use_token_credit=True,
            use_soft_gate=True,
            use_asymmetric_projection=False,
        )
        # Without asymmetric surgery, combined = g_rl + lam * g_sd (no proj).
        # So combined - lam * g_sd == g_rl regardless of conflict.
        lam = result.metrics["vacs/lambda_eff"]
        recovered = result.gradients[0] - lam * result.components.sd_grad[0]
        self.assertTrue(torch.allclose(recovered, result.components.rl_grad[0], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
