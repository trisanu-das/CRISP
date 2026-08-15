"""Unit tests for crisp/training/ema_adapter.py.

The EMA adapter is CIBO v2's anchor: an EMA of the trainable adapter
parameters that produces a frozen "previous-step" student for the
anchor cross-entropy term. The plan requires these properties:

  - Initialization: the EMA is exactly the current trainable params.
  - Update: the decay formula  ema = decay * ema + (1 - decay) * p
  - Target forward context manager: swaps in the EMA values, runs the
    caller-provided forward, then RESTORES the live values -- both on
    the happy path AND on exception.
  - Anchor target has no gradient; the student loss computed after the
    context manager exits DOES carry gradient through the live params.
  - State dict round-trips losslessly.

The fake causal LM here uses the same construction as
tests/test_teacher_student.py / tests/test_vacs_step.py: a callable
class whose forward produces logits whose log_softmax at the chosen
response token ids equals a target log-prob, AND whose output depends
on `params` (so a real autograd graph flows through).
"""
from __future__ import annotations

import math
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from crisp.training.ema_adapter import EmaAdapterState, ema_target_forward


def _make_fake_model(params, vocab=16):
    """Fake model whose forward depends on `params`."""
    W = torch.full((5, vocab), 0.1)

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, input_ids=None, attention_mask=None):
            B, L = input_ids.shape
            base = params[0] @ W  # [1, vocab]
            logits = base.unsqueeze(1).expand(B, L, vocab).contiguous()
            # Mark target response positions with a chosen log-prob.
            for t in range(min(L - 1, 2)):
                p = 0.9
                row = torch.full((vocab,), math.log((1.0 - p) / (vocab - 1)))
                row[5] = math.log(p)
                logits[0, t, :] = row
            return type("O", (), {"logits": logits})()

    return FakeModel()


def _make_params():
    return torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestEmaInitialization(unittest.TestCase):
    def test_ema_initializes_from_trainable_parameters(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=0.9)
        for live, stored in zip(params, ema.ema_values):
            self.assertTrue(torch.allclose(live.detach(), stored))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestEmaUpdate(unittest.TestCase):
    def test_ema_update_matches_decay_formula(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=0.9)
        original = params[0].detach().clone()
        # Mutate the live parameter to a new value.
        with torch.no_grad():
            params[0].fill_(10.0)
        ema.update()
        # ema_new = 0.9 * original + 0.1 * new
        # original was [1, 2, 3, 4, 5]; new is [10, 10, 10, 10, 10].
        expected = 0.9 * original + 0.1 * torch.full_like(original, 10.0)
        self.assertTrue(torch.allclose(ema.ema_values[0], expected, atol=1e-6))

    def test_decay_one_holds_old_value(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=1.0)
        original = ema.ema_values[0].clone()
        with torch.no_grad():
            params[0].fill_(99.0)
        ema.update()
        # decay=1 -> 1-decay=0 -> EMA is unchanged from observations.
        self.assertTrue(torch.allclose(ema.ema_values[0], original, atol=1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestEmaTargetForward(unittest.TestCase):
    def test_target_forward_restores_live_adapter_values_on_success(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=0.9)
        original = [p.detach().clone() for p in params]
        ema_snapshot = [v.clone() for v in ema.ema_values]

        # During the swap, params must equal ema_values.
        with ema_target_forward(ema, params):
            for live, snap in zip(params, ema_snapshot):
                self.assertTrue(torch.allclose(live.detach(), snap),
                                msg="during ema_target_forward, params must equal EMA values")

        # After exit, params are restored exactly.
        for live, orig in zip(params, original):
            self.assertTrue(torch.allclose(live.detach(), orig),
                            msg="after ema_target_forward, params must be restored exactly")

    def test_target_forward_restores_live_adapter_values_on_exception(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=0.9)
        original = [p.detach().clone() for p in params]

        try:
            with ema_target_forward(ema, params):
                # Mutate live to a non-EMA value to detect missing restore.
                with torch.no_grad():
                    params[0].fill_(42.0)
                raise RuntimeError("simulated failure inside the context")
        except RuntimeError:
            pass
        # Even on exception, params must be restored.
        for live, orig in zip(params, original):
            self.assertTrue(torch.allclose(live.detach(), orig),
                            msg="on exception, params must still be restored")

    def test_anchor_target_has_no_gradient_and_student_loss_does(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=0.9)
        model = _make_fake_model(params)
        # Compute a target log-prob via the EMA anchor (no grad).
        target_logp = torch.tensor(0.0)
        with ema_target_forward(ema, params):
            out = model(input_ids=torch.tensor([[1, 2, 3, 5, 6]]),
                         attention_mask=torch.tensor([[1, 1, 1, 1, 1]]))
            target_logp = torch.nn.functional.log_softmax(out.logits[0, 2:4, :].float(), dim=-1)[
                :, 5
            ].sum()
        self.assertFalse(target_logp.requires_grad,
                         msg="anchor target log-prob MUST have no grad")
        # Now compute a student loss through the live params.
        out = model(input_ids=torch.tensor([[1, 2, 3, 5, 6]]),
                     attention_mask=torch.tensor([[1, 1, 1, 1, 1]]))
        student_logp = torch.nn.functional.log_softmax(out.logits[0, 2:4, :].float(), dim=-1)[
            :, 5
        ].sum()
        self.assertTrue(student_logp.requires_grad,
                        msg="student log-prob MUST carry grad through live params")
        # The student grad reaches params.
        student_logp.sum().backward()
        self.assertIsNotNone(params[0].grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestEmaStateSerialization(unittest.TestCase):
    def test_ema_state_round_trips_through_state_dict(self):
        params = [_make_params()]
        ema = EmaAdapterState(params, decay=0.9)
        # Mutate and update to get a non-trivial state.
        with torch.no_grad():
            params[0].fill_(7.0)
        ema.update()
        snapshot = ema.state_dict()

        # Build a new EMA from scratch and load the snapshot.
        new_params = [_make_params()]
        new_ema = EmaAdapterState(new_params, decay=0.5)  # different decay
        new_ema.load_state_dict(snapshot)
        # The EMA values are exactly the snapshotted ones (decay info
        # comes from the snapshot, not the new constructor).
        self.assertTrue(torch.allclose(new_ema.ema_values[0], ema.ema_values[0]))
        self.assertEqual(new_ema.decay, 0.9)


if __name__ == "__main__":
    unittest.main()