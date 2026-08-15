"""Tests for the lifecycle hooks added to crisp/training/common_loop.py.

The plan requires two optional hooks the loop calls once per optimizer
step (NOT once per micro-batch, when `grad_accum_steps > 1`):

  - `post_optimizer_step(model, aggregated_step_metrics)` -- lets a
    method commit observations to its state AFTER the optimizer step
    (so the next step sees the freshly-stepped state, not the pre-step
    values). For VACS this is when `VacsAdaptiveState.commit` runs.

  - `checkpoint_state() -> dict` -- lets the method serialize any
    additional state the project checkpoint should carry alongside
    the model + optimizer state.

Both must default to "do nothing" so legacy CRISP and baselines
(GRPO / REINFORCE++ / OPSD) keep their current behavior.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from crisp.training.common_loop import run_training_loop


def _make_minimal_dataset(n: int = 8):
    return [
        {"prompt": f"What is {i} + 1?", "answer": str(i + 1)} for i in range(n)
    ]


def _make_minimal_cfg(tmp_dir: Path, *, total_steps: int, accum_steps: int):
    """A bare-bones DotDict config that satisfies the loop's reads."""
    from crisp.utils.config import DotDict

    return DotDict({
        "seed": 0,
        "training": {
            "output_dir": str(tmp_dir),
            "micro_batch_size": 2,
            "grad_accum_steps": accum_steps,
            "total_steps": total_steps,
            "save_every": 0,
            "eval_every": 0,
            "log_every": 1,
            "lambda_max": 1.0,
            "clip_epsilon": 0.2,
            "adv_eps": 1e-8,
            "pcgrad_eps": 1e-12,
        },
        "rollout": {
            "max_prompt_length": 16,
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
        },
        "model": {"system_prompt": "Be brief."},
        "data": {"eval_datasets": []},
        "logging": {"backend": "none", "run_name": "smoke", "wandb_project": "x"},
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "max_grad_norm": 1.0,
            "warmup_steps": 1,
        },
        "checkpointing": {"save_adapter_only": False},
    })


class _FakeParams:
    """Minimal param list that supports the loop's optimizer API."""

    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(3))
        self.call_count = 0
        self.optimizer = None

    def parameters(self):
        return iter([self.p])


class _FakeModel:
    """A trivial model that supports optimizer/scheduler bookkeeping."""

    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(3))
        self.device = torch.device("cpu")

    def eval(self):
        pass

    def parameters(self):
        return iter([self.p])

    def state_dict(self):
        return {"p": self.p.detach()}

    def __call__(self, *args, **kwargs):
        # Not used in this test path (step_fn is mocked).
        raise NotImplementedError


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestLifecycleHooks(unittest.TestCase):
    def test_hooks_default_to_do_nothing(self):
        # Without any hooks set, the loop completes without errors and
        # behaves exactly like the legacy code path.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_minimal_cfg(Path(tmp), total_steps=2, accum_steps=2)
            dataset = _make_minimal_dataset(n=4)
            model = _FakeModel()
            params = list(model.parameters())
            tokenizer = MagicMock()

            # step_fn: produce a tiny zero gradient (the loop will
            # divide by accum_steps and step the optimizer -- with
            # zero grads this is a no-op for params).
            def step_fn(model, tokenizer, prompts, answers, global_step):
                zeros = [torch.zeros_like(p) for p in params]
                metrics = {"loss": 0.0}
                return zeros, metrics

            # No hooks passed -- must not crash, must not require them.
            run_training_loop(
                model=model,
                tokenizer=tokenizer,
                params=params,
                train_dataset=dataset,
                step_fn=step_fn,
                cfg=cfg,
                run_name="default_no_hooks",
                eval_fn=None,
            )
            # If we got here, the loop ran to completion without hooks.

    def test_post_optimizer_step_called_once_per_optimizer_step(self):
        # Critical timing invariant: with grad_accum_steps > 1, the hook
        # must be called ONCE per optimizer step, NOT once per micro-batch.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_minimal_cfg(Path(tmp), total_steps=3, accum_steps=2)
            dataset = _make_minimal_dataset(n=4)
            model = _FakeModel()
            params = list(model.parameters())
            tokenizer = MagicMock()

            post_calls = []

            def post_optimizer_step(model, aggregated_step_metrics):
                post_calls.append(aggregated_step_metrics)

            def step_fn(model, tokenizer, prompts, answers, global_step):
                zeros = [torch.zeros_like(p) for p in params]
                metrics = {"loss": float(global_step)}
                return zeros, metrics

            run_training_loop(
                model=model,
                tokenizer=tokenizer,
                params=params,
                train_dataset=dataset,
                step_fn=step_fn,
                cfg=cfg,
                run_name="post_hook",
                eval_fn=None,
                post_optimizer_step=post_optimizer_step,
            )
            self.assertEqual(len(post_calls), 3,
                             msg=f"expected 3 post-step calls (one per optimizer step), got {len(post_calls)}")

    def test_checkpoint_state_hook_is_optional(self):
        # Without checkpoint_state, the existing save_checkpoint behavior
        # must continue to work and not crash.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_minimal_cfg(Path(tmp), total_steps=1, accum_steps=1)
            dataset = _make_minimal_dataset(n=4)
            model = _FakeModel()
            params = list(model.parameters())
            tokenizer = MagicMock()

            def step_fn(model, tokenizer, prompts, answers, global_step):
                zeros = [torch.zeros_like(p) for p in params]
                metrics = {"loss": 0.0}
                return zeros, metrics

            run_training_loop(
                model=model,
                tokenizer=tokenizer,
                params=params,
                train_dataset=dataset,
                step_fn=step_fn,
                cfg=cfg,
                run_name="no_ckpt_hook",
                eval_fn=None,
                # no checkpoint_state hook
            )
            # Should have produced a "final" checkpoint via the existing
            # save_checkpoint path.
            final_dir = Path(tmp) / "final"
            self.assertTrue(final_dir.exists(),
                            msg="save_checkpoint did not produce 'final' directory without checkpoint_state hook")

    def test_post_optimizer_step_called_once_per_optimizer_step_with_grad_accum(self):
        # Critical timing invariant the plan calls out: EMA / beta
        # controller updates must execute ONCE per optimizer step, not
        # once per micro-batch. We simulate this by maintaining a
        # counter inside the hook and a separate counter inside step_fn.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_minimal_cfg(Path(tmp), total_steps=4, accum_steps=3)
            dataset = _make_minimal_dataset(n=8)
            model = _FakeModel()
            params = list(model.parameters())
            tokenizer = MagicMock()

            counters = {"post_step": 0, "step_fn": 0}

            def post_optimizer_step(model, aggregated_step_metrics):
                counters["post_step"] += 1

            def step_fn(model, tokenizer, prompts, answers, global_step):
                counters["step_fn"] += 1
                zeros = [torch.zeros_like(p) for p in params]
                metrics = {"loss": float(global_step)}
                return zeros, metrics

            run_training_loop(
                model=model,
                tokenizer=tokenizer,
                params=params,
                train_dataset=dataset,
                step_fn=step_fn,
                cfg=cfg,
                run_name="ema_beta_timing",
                eval_fn=None,
                post_optimizer_step=post_optimizer_step,
            )
            # 4 optimizer steps -> 4 post_step calls (NOT 12 step_fn calls).
            self.assertEqual(counters["post_step"], 4)
            self.assertEqual(counters["step_fn"], 4 * 3,
                             msg=f"expected {4*3} step_fn calls (4 optim steps * 3 accum), "
                                 f"got {counters['step_fn']}")


if __name__ == "__main__":
    unittest.main()
