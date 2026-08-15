"""Shared training loop for CRISP, baselines, VACS, and CIBO.

Two execution contracts are supported:

1. Legacy ``step_fn``: one micro-batch directly returns gradients + metrics.
2. Two-stage ``collect_fn``/``score_fn``: collect all rollouts for the
   effective batch first, compute one reward mean/std over all accumulated
   micro-batches (and all distributed ranks), then score each stored
   micro-batch with those shared statistics.

The second contract is the paper-correct path for REINFORCE++-style global
advantage normalization. Collection never retains autograd graphs, so it does
not increase scoring-graph memory with ``grad_accum_steps``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import torch

from crisp.training.batch_types import AdvantageStats, CollectedBatch
from crisp.utils.distributed import (
    all_reduce_gradients,
    barrier,
    gather_1d_equal,
    is_main_process,
    rank,
    world_size,
)
from crisp.utils.logging_utils import RunLogger


class StatefulBatchStream:
    """Deterministic, resumable stream of shuffled fixed-size batches.

    In distributed mode each rank owns a disjoint strided shard. All ranks
    must use the same batch size so reward all-gather lengths match.
    """

    def __init__(self, dataset: list[dict], batch_size: int, seed: int):
        self.batch_size = int(batch_size)
        self.rng = random.Random(int(seed))
        self.indices = list(range(len(dataset)))[rank()::world_size()]
        self.dataset = dataset
        self.position = len(self.indices)  # force a shuffle before first use
        if len(self.indices) < self.batch_size:
            raise ValueError(
                f"Rank {rank()} has only {len(self.indices)} examples after distributed "
                f"sharding, smaller than micro_batch_size={self.batch_size}."
            )

    def _reshuffle(self) -> None:
        self.rng.shuffle(self.indices)
        self.position = 0

    def next_batch(self) -> list[dict]:
        if self.position + self.batch_size > len(self.indices):
            self._reshuffle()
        selected = self.indices[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        return [self.dataset[i] for i in selected]

    def state_dict(self) -> dict:
        return {
            "indices": list(self.indices),
            "position": int(self.position),
            "rng_state": self.rng.getstate(),
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise ValueError("Cannot resume with a different micro_batch_size.")
        self.indices = list(state["indices"])
        self.position = int(state["position"])
        self.rng.setstate(state["rng_state"])


def batches(dataset: list[dict], batch_size: int, seed: int):
    """Backward-compatible generator used by older external callers."""
    stream = StatefulBatchStream(dataset, batch_size, seed)
    while True:
        yield stream.next_batch()


def build_optimizer_and_scheduler(params, cfg):
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )
    warmup_steps = max(1, int(cfg.optimizer.warmup_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def _save_model(model, path: Path, cfg) -> None:
    if cfg.checkpointing.get("save_adapter_only", True) and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(path))
    else:
        torch.save(model.state_dict(), path / "model_state_dict.pt")


def save_checkpoint(
    model,
    path: Path,
    cfg,
    *,
    optimizer=None,
    scheduler=None,
    batch_stream: Optional[StatefulBatchStream] = None,
    global_step: int = 0,
    method_state: Optional[dict] = None,
) -> None:
    """Save adapter/model and complete per-rank resumable training state.

    The adapter itself is identical on every rank after gradient all-reduce, so
    only rank 0 writes model files. RNG, data-stream position, and method state
    can differ by rank and are therefore written to rank-specific state files.
    """
    path.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        _save_model(model, path, cfg)

    state = {
        "global_step": int(global_step),
        "rank": int(rank()),
        "world_size": int(world_size()),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "batch_stream": batch_stream.state_dict() if batch_stream is not None else None,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "method_state": method_state,
    }

    if world_size() > 1:
        state_path = path / f"training_state_rank_{rank()}.pt"
    else:
        state_path = path / "training_state.pt"
    torch.save(state, state_path)

    # Rank-zero aliases keep older tooling usable without sacrificing exact
    # per-rank resume data for distributed runs.
    if is_main_process() and world_size() > 1:
        torch.save(state, path / "training_state.pt")
    if is_main_process() and method_state:
        torch.save(method_state, path / "method_state.pt")


def _restore_training_state(
    resume_dir: Path,
    *,
    optimizer,
    scheduler,
    batch_stream: StatefulBatchStream,
    restore_method_state: Optional[Callable],
) -> int:
    rank_state = resume_dir / f"training_state_rank_{rank()}.pt"
    common_state = resume_dir / "training_state.pt"
    state_path = rank_state if rank_state.exists() else common_state
    if not state_path.exists():
        raise FileNotFoundError(
            f"Resume directory {resume_dir} has no training state for rank {rank()}. "
            "It may be an old adapter-only checkpoint and cannot exactly resume."
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_world = int(state.get("world_size", 1))
    if saved_world != world_size():
        raise ValueError(
            f"Cannot exactly resume a checkpoint saved with world_size={saved_world} "
            f"using world_size={world_size()}."
        )
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if state.get("batch_stream") is not None:
        batch_stream.load_state_dict(state["batch_stream"])
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if restore_method_state is not None and state.get("method_state") is not None:
        restore_method_state(state["method_state"])
    return int(state.get("global_step", 0))


def _numeric_average(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    keys = set.intersection(*[
        {k for k, v in m.items() if isinstance(v, (int, float))}
        for m in metrics
    ])
    return {k: sum(float(m[k]) for m in metrics) / len(metrics) for k in sorted(keys)}


def _write_generations(handle, records) -> None:
    if not records or handle is None:
        return
    for rec in records:
        handle.write(json.dumps(rec) + "\n")


def run_training_loop(
    model,
    tokenizer,
    params: list[torch.nn.Parameter],
    train_dataset: list[dict],
    step_fn: Optional[Callable],
    cfg,
    run_name: str,
    eval_fn: Optional[Callable] = None,
    post_optimizer_step: Optional[Callable] = None,
    checkpoint_state: Optional[Callable] = None,
    restore_method_state: Optional[Callable] = None,
    collect_fn: Optional[Callable] = None,
    score_fn: Optional[Callable] = None,
) -> None:
    if (collect_fn is None) != (score_fn is None):
        raise ValueError("collect_fn and score_fn must either both be provided or both omitted.")
    if collect_fn is None and step_fn is None:
        raise ValueError("Provide either step_fn or collect_fn+score_fn.")

    output_dir = Path(cfg.training.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    logger = RunLogger(str(output_dir), run_name, dict(cfg), backend=(cfg.logging.backend if is_main_process() else "none"))
    generations_fh = None
    if is_main_process():
        generations_fh = open(output_dir / "generations.jsonl", "a", buffering=1)

    optimizer, scheduler = build_optimizer_and_scheduler(params, cfg)
    accum_steps = int(cfg.training.grad_accum_steps)
    stream = StatefulBatchStream(
        train_dataset,
        int(cfg.training.micro_batch_size),
        seed=int(cfg.seed),
    )
    model.eval()

    resume_from = cfg.training.get("resume_from", None)
    global_step = 0
    if resume_from:
        global_step = _restore_training_state(
            Path(resume_from),
            optimizer=optimizer,
            scheduler=scheduler,
            batch_stream=stream,
            restore_method_state=restore_method_state,
        )
        if is_main_process():
            print(f"[{run_name}] resumed at optimizer step {global_step} from {resume_from}")

    completed = False
    try:
        while global_step < int(cfg.training.total_steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated: Optional[list[torch.Tensor]] = None
            accum_metrics: list[dict] = []

            if collect_fn is not None:
                collected_batches: list[CollectedBatch] = []
                for micro_idx in range(accum_steps):
                    if is_main_process():
                        print(
                            f"[{run_name}] step {global_step}: "
                            f"collecting microbatch {micro_idx + 1}/{accum_steps}"
                        )
                    batch = stream.next_batch()
                    prompts = [ex["prompt"] for ex in batch]
                    answers = [ex["answer"] for ex in batch]
                    collected = collect_fn(model, tokenizer, prompts, answers, global_step)
                    collected.validate()
                    collected_batches.append(collected)
                    _write_generations(generations_fh, collected.metadata.get("generations"))

                local_rewards = torch.cat([c.rewards.float() for c in collected_batches], dim=0)
                stats_device = params[0].device if params else model.device
                global_rewards = gather_1d_equal(local_rewards.to(stats_device))
                adv_stats = AdvantageStats(
                    mean=float(global_rewards.mean().item()),
                    std=float(global_rewards.std(unbiased=False).item()),
                    count=int(global_rewards.numel()),
                )

                for collected in collected_batches:
                    local_advantages = adv_stats.normalize(
                        collected.rewards.float(),
                        eps=float(cfg.training.adv_eps),
                    )
                    grads, metrics = score_fn(
                        model,
                        tokenizer,
                        collected,
                        local_advantages,
                        global_step,
                    )
                    metrics = dict(metrics)
                    metrics.setdefault("reward_mean", float(collected.rewards.mean().item()))
                    metrics.setdefault("reward_std", float(collected.rewards.std(unbiased=False).item()))
                    metrics["effective_reward_mean"] = adv_stats.mean
                    metrics["effective_reward_std"] = adv_stats.std
                    metrics["effective_reward_count"] = float(adv_stats.count)
                    accum_metrics.append(metrics)
                    if accumulated is None:
                        accumulated = [g.detach().clone() for g in grads]
                    else:
                        for acc, grad in zip(accumulated, grads):
                            acc.add_(grad)
            else:
                for _ in range(accum_steps):
                    batch = stream.next_batch()
                    prompts = [ex["prompt"] for ex in batch]
                    answers = [ex["answer"] for ex in batch]
                    grads, metrics = step_fn(model, tokenizer, prompts, answers, global_step)
                    metrics = dict(metrics)
                    _write_generations(generations_fh, metrics.pop("_generations", None))
                    accum_metrics.append(metrics)
                    if accumulated is None:
                        accumulated = [g.detach().clone() for g in grads]
                    else:
                        for acc, grad in zip(accumulated, grads):
                            acc.add_(grad)

            if accumulated is None:
                raise RuntimeError("No gradients were produced for this optimizer step.")
            accumulated = [g / accum_steps for g in accumulated]
            # Manual-gradient methods need explicit distributed synchronization.
            accumulated = all_reduce_gradients(accumulated)

            for parameter, gradient in zip(params, accumulated):
                parameter.grad = gradient

            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optimizer.max_grad_norm)
            optimizer.step()
            scheduler.step()

            aggregated = _numeric_average(accum_metrics)
            aggregated["grad_norm"] = float(grad_norm.item())
            aggregated["lr"] = float(scheduler.get_last_lr()[0])

            if post_optimizer_step is not None:
                post_optimizer_step(model, aggregated)

            if cfg.training.log_every and global_step % int(cfg.training.log_every) == 0 and is_main_process():
                logger.log(aggregated, global_step)
                print(
                    f"[{run_name}] step {global_step}: "
                    + ", ".join(f"{k}={v:.8g}" for k, v in aggregated.items())
                )

            if (
                eval_fn is not None
                and cfg.training.eval_every
                and global_step % int(cfg.training.eval_every) == 0
                and global_step > 0
                and is_main_process()
            ):
                eval_metrics = eval_fn(model, tokenizer, global_step)
                logger.log({f"eval/{k}": v for k, v in eval_metrics.items()}, global_step)
                model.eval()

            global_step += 1

            if cfg.training.save_every and global_step % int(cfg.training.save_every) == 0:
                method_state = checkpoint_state() if checkpoint_state is not None else None
                save_checkpoint(
                    model,
                    output_dir / f"step_{global_step}",
                    cfg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    batch_stream=stream,
                    global_step=global_step,
                    method_state=method_state,
                )
                barrier()

        completed = True

    except BaseException:
        method_state = checkpoint_state() if checkpoint_state is not None else None
        save_checkpoint(
            model,
            output_dir / "interrupted",
            cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_stream=stream,
            global_step=global_step,
            method_state=method_state,
        )
        barrier()
        raise
    finally:
        logger.close()
        if generations_fh is not None:
            generations_fh.close()

    if completed:
        method_state = checkpoint_state() if checkpoint_state is not None else None
        save_checkpoint(
            model,
            output_dir / "final",
            cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_stream=stream,
            global_step=global_step,
            method_state=method_state,
        )
        if is_main_process():
            (output_dir / "_SUCCESS").write_text("completed\n")
        barrier()
