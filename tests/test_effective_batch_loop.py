from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from crisp.training.batch_types import CollectedBatch
from crisp.training.common_loop import run_training_loop
from crisp.utils.config import DotDict


def test_advantages_use_all_accumulated_microbatches():
    """[0,0] and [1,1] must normalize together, not become four zeros."""
    with tempfile.TemporaryDirectory() as td:
        model = torch.nn.Linear(1, 1, bias=False)
        param = next(model.parameters())
        observed: list[torch.Tensor] = []
        collect_calls = 0

        def collect_fn(model_, tokenizer_, prompts, answers, global_step):
            nonlocal collect_calls
            reward = 0.0 if collect_calls == 0 else 1.0
            collect_calls += 1
            rewards = torch.full((len(prompts),), reward)
            return CollectedBatch(
                prompts=list(prompts),
                answers=list(answers),
                rollouts=[object() for _ in prompts],
                rewards=rewards,
            )

        def score_fn(model_, tokenizer_, collected, advantages, global_step):
            observed.append(advantages.detach().clone())
            # A correctly shaped synthetic manual gradient is sufficient to
            # exercise common_loop's accumulation and optimizer plumbing.
            grad = torch.ones_like(param) * advantages.mean().to(param.device)
            return [grad], {"loss_rl": 0.0}

        cfg = DotDict({
            "seed": 0,
            "optimizer": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "warmup_steps": 1,
                "max_grad_norm": 1.0,
            },
            "training": {
                "output_dir": td,
                "total_steps": 1,
                "micro_batch_size": 2,
                "grad_accum_steps": 2,
                "adv_eps": 1e-8,
                "log_every": 0,
                "eval_every": 0,
                "save_every": 0,
                "resume_from": None,
            },
            "checkpointing": {"save_adapter_only": False},
            "logging": {"backend": "none"},
        })
        dataset = [
            {"prompt": f"p{i}", "answer": "a"}
            for i in range(8)
        ]

        run_training_loop(
            model=model,
            tokenizer=None,
            params=[param],
            train_dataset=dataset,
            step_fn=None,
            collect_fn=collect_fn,
            score_fn=score_fn,
            cfg=cfg,
            run_name="effective-batch-test",
        )

        assert len(observed) == 2
        assert torch.allclose(observed[0], torch.tensor([-1.0, -1.0]))
        assert torch.allclose(observed[1], torch.tensor([1.0, 1.0]))
        assert (Path(td) / "_SUCCESS").exists()
        assert (Path(td) / "final" / "training_state.pt").exists()
