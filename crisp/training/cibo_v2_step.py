"""CIBO-CRISP v2 single-objective scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from crisp.training.beta_controller import BetaController
from crisp.training.ema_adapter import EmaAdapterState, ema_target_forward
from crisp.training.grad_utils import compute_grads
from crisp.training.logprobs import ScoredSequence
from crisp.training.objectives import (
    cibo_token_advantages,
    normalize_advantages,
    reduce_gated_kl,
    soft_reward_gate,
)
from crisp.training.teacher_student import score_teacher_student


@dataclass
class CiboComponents:
    loss_total: torch.Tensor
    loss_rl: torch.Tensor
    loss_ib: torch.Tensor
    loss_anchor: torch.Tensor
    token_adv: torch.Tensor
    gradients: list[torch.Tensor]
    beta: float
    metrics: dict = field(default_factory=dict)


def _stack_per_row(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise ValueError("Cannot stack an empty row list.")
    max_len = max(int(row.shape[0]) for row in rows)
    out = torch.zeros(
        len(rows),
        max_len,
        device=rows[0].device,
        dtype=rows[0].dtype,
    )
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out


def build_cibo_components(
    *,
    model,
    teacher_sequences: list[ScoredSequence],
    student_sequences: list[ScoredSequence],
    pad_id: int,
    params: list[torch.nn.Parameter],
    rewards: torch.Tensor,
    ema: EmaAdapterState,
    beta_controller: BetaController,
    credit_lambda: float,
    soft_gate_slope: float,
    reward_threshold: float,
    anchor_alpha: float,
    use_anchor: bool,
    advantages: torch.Tensor | None = None,
    ib_reduction: str = "sequence_sum",
    use_soft_gate: bool = True,
) -> CiboComponents:
    """Build one scalar CIBO objective and differentiate it exactly once."""
    if use_anchor and anchor_alpha > 0.0:
        with ema_target_forward(ema, params):
            ema_scored = score_teacher_student(
                model=model,
                teacher_sequences=teacher_sequences,
                student_sequences=student_sequences,
                pad_id=pad_id,
            )
        ema_student_row_log_probs = [
            logp.detach() for logp in ema_scored.student_row_log_probs
        ]
    else:
        ema_student_row_log_probs = None

    scored = score_teacher_student(
        model=model,
        teacher_sequences=teacher_sequences,
        student_sequences=student_sequences,
        pad_id=pad_id,
    )
    teacher_token_logps = _stack_per_row(scored.teacher_token_logps)
    student_token_logps = _stack_per_row(scored.student_token_logps)

    if advantages is None:
        advantages = normalize_advantages(rewards, eps=1e-8)
    advantages = advantages.to(model.device)
    token_adv = cibo_token_advantages(
        advantages,
        teacher_token_logps,
        student_token_logps,
        credit_lambda=credit_lambda,
    )

    response_lens = [int(row.shape[0]) for row in scored.student_token_logps]
    max_len = student_token_logps.shape[1]
    valid_mask = torch.zeros(
        len(response_lens), max_len, device=student_token_logps.device
    )
    for i, length in enumerate(response_lens):
        valid_mask[i, :length] = 1.0
    if not bool(valid_mask.any().item()):
        raise RuntimeError("CIBO received no response tokens to score.")

    per_token_rl = token_adv * student_token_logps * valid_mask
    per_example_rl = -(
        per_token_rl.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
    )
    loss_rl = per_example_rl.mean()

    if use_soft_gate:
        gate = soft_reward_gate(
            rewards,
            slope=soft_gate_slope,
            threshold=reward_threshold,
        )
    else:
        gate = torch.ones_like(rewards)
    loss_ib = reduce_gated_kl(
        scored.teacher_per_token_kl,
        gate,
        reduction=ib_reduction,
    )

    # Full-distribution cross entropy H(p_EMA, p_live). The EMA side is a
    # detached target; the live student side remains differentiable.
    anchor_terms: list[torch.Tensor] = []
    if ema_student_row_log_probs is not None:
        for ema_logp, live_logp in zip(
            ema_student_row_log_probs,
            scored.student_row_log_probs,
        ):
            if ema_logp.shape != live_logp.shape:
                raise ValueError(
                    f"EMA/live response distributions differ: {ema_logp.shape} vs {live_logp.shape}"
                )
            if ema_logp.shape[0] == 0:
                continue
            ema_probs = ema_logp.exp()
            anchor_terms.append(-(ema_probs * live_logp).sum(dim=-1).mean())
    loss_anchor = (
        torch.stack(anchor_terms).mean()
        if anchor_terms
        else torch.zeros((), device=model.device)
    )

    beta = float(beta_controller.beta)
    loss_total = loss_rl + beta * loss_ib + float(anchor_alpha) * loss_anchor
    gradients = compute_grads(loss_total, params, retain_graph=False)

    nonempty_kl = [row.mean() for row in scored.teacher_per_token_kl if row.numel() > 0]
    kl_mean = (
        float(torch.stack(nonempty_kl).mean().item()) if nonempty_kl else 0.0
    )
    credit_multiplier = (
        1.0
        + credit_lambda
        * torch.tanh(
            teacher_token_logps.detach() - student_token_logps.detach()
        )
    )
    metrics = {
        "cibo/loss_total": float(loss_total.item()),
        "cibo/loss_rl": float(loss_rl.item()),
        "cibo/loss_ib": float(loss_ib.item()),
        "cibo/loss_anchor": float(loss_anchor.item()),
        "cibo/beta": beta,
        "cibo/beta_mode": beta_controller.beta_mode,
        "cibo/beta_reference_kl": float(beta_controller.reference_kl),
        "cibo/baselined_kl": (
            float(beta_controller.last_kl - beta_controller.reference_kl)
            if beta_controller.last_kl is not None
            else 0.0
        ),
        "cibo/credit_multiplier_mean": float(
            credit_multiplier[valid_mask.bool()].mean().item()
        ),
        "cibo/ib_gate_mean": float(gate.mean().item()),
        "cibo/kl_per_token_mean": kl_mean,
        "cibo/anchor_alpha": float(anchor_alpha),
        "cibo/ema_decay": float(ema.decay),
        "cibo/ib_reduction_token_mean": float(ib_reduction == "token_mean"),
    }

    return CiboComponents(
        loss_total=loss_total,
        loss_rl=loss_rl,
        loss_ib=loss_ib,
        loss_anchor=loss_anchor,
        token_adv=token_adv,
        gradients=gradients,
        beta=beta,
        metrics=metrics,
    )


def cibo_v2_step_from_arrays(
    *,
    model,
    teacher_sequences: list[ScoredSequence],
    student_sequences: list[ScoredSequence],
    pad_id: int,
    params: list[torch.nn.Parameter],
    rewards: torch.Tensor,
    ema: EmaAdapterState,
    beta_controller: BetaController,
    credit_lambda: float,
    soft_gate_slope: float,
    reward_threshold: float,
    anchor_alpha: float,
    use_anchor: bool,
    advantages: torch.Tensor | None = None,
    ib_reduction: str = "sequence_sum",
    use_soft_gate: bool = True,
):
    comp = build_cibo_components(
        model=model,
        teacher_sequences=teacher_sequences,
        student_sequences=student_sequences,
        pad_id=pad_id,
        params=params,
        rewards=rewards,
        ema=ema,
        beta_controller=beta_controller,
        credit_lambda=credit_lambda,
        soft_gate_slope=soft_gate_slope,
        reward_threshold=reward_threshold,
        anchor_alpha=anchor_alpha,
        use_anchor=use_anchor,
        advantages=advantages,
        ib_reduction=ib_reduction,
        use_soft_gate=use_soft_gate,
    )
    return comp.gradients, comp.metrics
