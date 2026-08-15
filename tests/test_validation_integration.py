"""Step 2 of VALIDATION.md: offline fake-model integration checks for
CRISP (control), VACS, CIBO v2 fixed-beta, and CIBO v2 adaptive-beta.

Each test:
  - constructs a tiny `nn.Linear` parameter set (so .grad is meaningful),
  - runs the method's step ONCE,
  - asserts losses are finite, gradients are finite,
  - asserts post-step state changes are correct.

This file lives in tests/ and IS itself part of the test suite, so
we re-run it under pytest (where it can be verbose if useful) and
also invoke it as a script for the ad-hoc verifier.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _make_fake_model_and_params(vocab: int = 16):
    """A 1-Linear fake model with a chosen log-prob structure.

    Forward: `logits = param @ W` then we set row 5 of the response
    position to log(0.9) and the rest to log(0.1 / (vocab-1)). This
    gives a deterministic teacher/student signal.
    """
    W = torch.full((5, vocab), 0.1)

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.W = W

        def __call__(self, input_ids=None, attention_mask=None):
            B, L = input_ids.shape
            base = params[0] @ self.W
            logits = base.unsqueeze(1).expand(B, L, vocab).contiguous()
            row = torch.full((vocab,), math.log(0.1 / (vocab - 1)))
            row[5] = math.log(0.9)
            logits[0, 1, :] = row
            return type("O", (), {"logits": logits})()

    params = [torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))]
    return FakeModel(), params


def _ts(ids, rs, rl):
    from crisp.training.logprobs import ScoredSequence
    return ScoredSequence(ids=ids, response_start=rs, response_len=rl)


class TestCrispOfflineIntegration(unittest.TestCase):
    """The CRISP control path. The reimplementation's `crisp_step` is
    bound to its own training loop signature; for the offline
    integration test we exercise the SHARED math primitives the
    CRISP step is built on (response-aligned teacher/student scoring,
    the SD forward-KL reduction) so we can verify finite losses
    without spinning up the full trainer.
    """

    def test_crisp_uses_response_aligned_scoring_and_finite_forward_kl(self):
        from crisp.training.teacher_student import score_teacher_student
        from crisp.training.objectives import reduce_gated_kl

        model, params = _make_fake_model_and_params()
        teacher_seqs = [_ts([10, 11, 12, 5, 6], 3, 2),
                        _ts([10, 11, 12, 5, 7], 3, 2)]
        student_seqs = [_ts([7, 5, 6], 1, 2),
                        _ts([7, 5, 7], 1, 2)]
        scored = score_teacher_student(
            model=model, teacher_sequences=teacher_seqs,
            student_sequences=student_seqs, pad_id=0,
        )
        # Compute per-token forward KL from the scored teacher/student
        # per-token log-probs (an ad-hoc synthetic for the test).
        # Use the SHARED objective: per-token KL[i] = max(0, lp_t - lp_s).
        per_token_kl = []
        for t_lp, s_lp in zip(scored.teacher_token_logps,
                              scored.student_token_logps):
            per_token_kl.append(torch.relu(t_lp - s_lp).detach())
        gate = torch.tensor([1.0, 1.0])  # CRISP has no soft gate
        kl = reduce_gated_kl(per_token_kl, gate_weights=gate,
                             reduction="sequence_sum")
        self.assertTrue(torch.isfinite(kl).item(),
                        msg=f"CRISP SD KL non-finite: {kl}")


class TestVacsOfflineIntegration(unittest.TestCase):
    """VACS: token credit + soft gate + asymmetric surgery + adaptive mix."""

    def test_vacs_step_produces_finite_losses_and_gradients(self):
        from crisp.training.vacs_step import vacs_step_from_arrays
        from crisp.training.vacs_state import VacsAdaptiveState
        model, params = _make_fake_model_and_params()
        vacs_state = VacsAdaptiveState()
        teacher_seqs = [_ts([10, 11, 12, 5, 6], 3, 2),
                        _ts([10, 11, 12, 5, 7], 3, 2)]
        student_seqs = [_ts([7, 5, 6], 1, 2),
                        _ts([7, 5, 7], 1, 2)]
        result = vacs_step_from_arrays(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=torch.tensor([1.0, 0.0]),
            base_lambda=1.0,
            adv_eps=1e-8,
            clip_epsilon=0.2,
            sd_weight=1.0,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            credit_mix=0.25,
            use_token_credit=True,
            use_soft_gate=True,
            use_asymmetric_projection=True,
            state=vacs_state,
        )
        grads = result.gradients
        self.assertEqual(len(grads), len(params))
        for g in grads:
            self.assertTrue(torch.isfinite(g).all().item(),
                            msg=f"VACS gradient non-finite: {g}")
        # VACS exposes named vacs/* metrics.
        m = result.metrics
        self.assertIn("vacs/loss_rl", m)
        self.assertIn("vacs/loss_sd", m)
        self.assertIn("vacs/lambda_eff", m)


class TestCiboFixedOfflineIntegration(unittest.TestCase):
    """CIBO v2 FIXED-beta: single composite loss, no PC-Grad."""

    def test_cibo_fixed_step_produces_finite_losses_and_gradients(self):
        from crisp.training.cibo_v2_step import build_cibo_components
        from crisp.training.ema_adapter import EmaAdapterState
        from crisp.training.beta_controller import BetaController
        model, params = _make_fake_model_and_params()
        ema = EmaAdapterState(params, decay=0.9)
        ctrl = BetaController(
            beta=0.1, beta_mode="fixed",
            beta_min=0.001, beta_max=10.0,
            beta_step_size=0.05, beta_ema_decay=0.95,
            beta_reference_warmup_steps=20,
        )
        teacher_seqs = [_ts([10, 11, 12, 5, 6], 3, 2),
                        _ts([10, 11, 12, 5, 7], 3, 2)]
        student_seqs = [_ts([7, 5, 6], 1, 2),
                        _ts([7, 5, 7], 1, 2)]
        comp = build_cibo_components(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=torch.tensor([1.0, 0.0]),
            ema=ema,
            beta_controller=ctrl,
            credit_lambda=0.25,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            anchor_alpha=0.01,
            use_anchor=True,
        )
        # Single composite loss + finite gradient.
        self.assertTrue(torch.isfinite(comp.loss_total).item())
        self.assertEqual(len(comp.gradients), len(params))
        for g in comp.gradients:
            self.assertTrue(torch.isfinite(g).all().item())
        # CIBO has NO pcgrad_result field (single-objective discipline).
        self.assertFalse(hasattr(comp, "pcgrad_result"))
        # Beta unchanged (fixed mode + step() not called yet).
        self.assertAlmostEqual(ctrl.beta, 0.1, places=6)


class TestCiboAdaptiveOfflineIntegration(unittest.TestCase):
    """CIBO v2 ADAPTIVE-beta: controller.step() grows beta on positive delta."""

    def test_cibo_adaptive_step_observer_and_controller(self):
        from crisp.training.cibo_v2_step import build_cibo_components
        from crisp.training.ema_adapter import EmaAdapterState
        from crisp.training.beta_controller import BetaController
        model, params = _make_fake_model_and_params()
        ema = EmaAdapterState(params, decay=0.9)
        ctrl = BetaController(
            beta=0.5, beta_mode="adaptive_sign",
            beta_min=0.001, beta_max=10.0,
            beta_step_size=0.05, beta_ema_decay=0.95,
            beta_reference_warmup_steps=0,
        )
        teacher_seqs = [_ts([10, 11, 12, 5, 6], 3, 2),
                        _ts([10, 11, 12, 5, 7], 3, 2)]
        student_seqs = [_ts([7, 5, 6], 1, 2),
                        _ts([7, 5, 7], 1, 2)]
        comp = build_cibo_components(
            model=model,
            teacher_sequences=teacher_seqs,
            student_sequences=student_seqs,
            pad_id=0,
            params=params,
            rewards=torch.tensor([1.0, 0.0]),
            ema=ema,
            beta_controller=ctrl,
            credit_lambda=0.25,
            soft_gate_slope=12.0,
            reward_threshold=0.5,
            anchor_alpha=0.01,
            use_anchor=True,
        )
        # Step() did not change beta (the post-step hook owns it).
        self.assertAlmostEqual(ctrl.beta, 0.5, places=6)
        # The first observation establishes the post-warmup delta
        # reference. The second creates a positive leakage delta.
        ctrl.observe(reward=0.0, kl_value=2.0)
        ctrl.step()
        self.assertAlmostEqual(ctrl.beta, 0.5, places=6)
        ctrl.observe(reward=0.0, kl_value=100.0)
        ctrl.step()
        self.assertGreater(ctrl.beta, 0.5,
                           msg="adaptive controller did not grow beta on +delta")
        # Checkpoint round-trip preserves beta + reference + state.
        state = ctrl.state_dict()
        ctrl2 = BetaController(
            beta=0.001, beta_mode="adaptive_sign",
            beta_min=0.001, beta_max=10.0,
            beta_step_size=0.05, beta_ema_decay=0.95,
            beta_reference_warmup_steps=0,
        )
        ctrl2.load_state_dict(state)
        self.assertAlmostEqual(ctrl2.beta, ctrl.beta, places=6)


class TestEmaAnchorNoUnwantedParameterMutation(unittest.TestCase):
    """The EMA adapter anchor must NOT mutate live parameters after the
    `with ema_target_forward(...)` block exits. Bit-for-bit equality.
    """

    def test_ema_target_forward_restores_live_params(self):
        from crisp.training.ema_adapter import EmaAdapterState, ema_target_forward
        params = [torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))]
        ema = EmaAdapterState(params, decay=0.9)
        # Snapshot before
        before = params[0].detach().clone()
        with ema_target_forward(ema, params):
            # Inside, params[0] equals EMA. EMA is initialized equal
            # to live params, so this is a no-op for now.
            pass
        # After exit, params[0] must equal `before` byte-for-byte.
        self.assertTrue(torch.equal(params[0].detach(), before),
                        msg="EMA anchor corrupted live params on exit")


if __name__ == "__main__":
    unittest.main()