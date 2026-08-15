"""VACS step: token-level RL + soft-gated forward KL, asymmetrically combined.

This module implements one micro-batch of VACS-CRISP, the same way
`crisp_step.py` implements one micro-batch of CRISP. The function
`vacs_step_from_arrays` takes already-built teacher/student
`ScoredSequence`s and a fixed `params` list, runs one batched
teacher+student forward, and returns the VACS gradient + metrics.

VACS differs from CRISP in five ways that are deliberately encoded here:

  1. Token-level VACS advantages via `vacs_token_advantages` (sign(a)-
     aware exponential credit, clipped, mixed by rho), used directly in
     a plain per-token REINFORCE loss. CRISP also uses a single-pass
     policy-gradient loss; VACS differs through the token-level credit
     and gradient-coupling rules, not through PPO clipping.
  2. Soft reward gating via `soft_reward_gate`, not the hard `< 1.0`
     correctness gate. The plan calls out the soft gate as VACS's
     smooth version of the CRISP gate.
  3. Asymmetric PC-Grad via `asymmetric_pcgrad_combine`, not the
     symmetric PCGrad in `pcgrad.py`. RL is preserved bit-for-bit; SD
     is projected away from the original RL direction.
  4. Adaptive lambda via `VacsAdaptiveState`, not a fixed cosine
     schedule. The state object enforces the "observe-then-commit"
     timing invariant so a micro-batch's gradient cannot influence the
     coefficient used in its own optimizer step.
  5. Each component is independently togglable via flags
     (`use_token_credit`, `use_soft_gate`, `use_asymmetric_projection`)
     so the ablations the plan requires are first-class -- you can
     isolate exactly one factor at a time without forking the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from crisp.training.gradient_surgery import (
    asymmetric_pcgrad_combine,
    flatten_grads,
    unflatten_grads,
)
from crisp.training.grad_utils import compute_grads
from crisp.training.logprobs import ScoredSequence
from crisp.training.objectives import (
    normalize_advantages,
    reduce_gated_kl,
    soft_reward_gate,
    vacs_token_advantages,
)
from crisp.training.teacher_student import score_teacher_student
from crisp.training.vacs_state import (
    VacsAdaptiveState,
    compute_lambda_effective,
)


@dataclass
class VacsComponents:
    """Intermediate state from one VACS micro-batch.

    `loss_rl` and `loss_sd` are the scalar losses used to derive
    `g_rl` and `g_sd` (kept separate so the caller can run them through
    asymmetric surgery with full visibility into both sides).
    `auxiliary_projected` is the projected SD gradient (== `sd_grad`
    when no conflict happened or projection was disabled).
    `lambda_eff` is the adaptive mixing coefficient used to combine
    them; None when adaptive mixing was disabled.
    """

    loss_rl: torch.Tensor
    loss_sd: torch.Tensor
    rl_grad: list[torch.Tensor]
    sd_grad: list[torch.Tensor]
    auxiliary_projected: list[torch.Tensor]
    lambda_eff: Optional[float]
    metrics: dict = field(default_factory=dict)


@dataclass
class VacsStepResult:
    """Final VACS step result, ready to be applied by the outer loop."""

    gradients: list[torch.Tensor]
    components: VacsComponents
    metrics: dict
    state: VacsAdaptiveState


def build_vacs_components(
    *,
    model,
    teacher_sequences: list[ScoredSequence],
    student_sequences: list[ScoredSequence],
    pad_id: int,
    params: list[torch.nn.Parameter],
    rewards: torch.Tensor,
    base_lambda: float,
    adv_eps: float,
    clip_epsilon: float,
    sd_weight: float,
    soft_gate_slope: float,
    reward_threshold: float,
    credit_mix: float,
    use_token_credit: bool,
    use_soft_gate: bool,
    advantages: torch.Tensor | None = None,
    use_asymmetric_projection: bool = True,
    state: Optional[VacsAdaptiveState] = None,
) -> VacsComponents:
    """Compute losses, gradients, projection, and metrics for one VACS micro-batch.

    Pure function over its inputs (no optimizer, no scheduler, no .grad
    mutation on `params`).
    """
    if state is None:
        state = VacsAdaptiveState()

    # ---- One batched teacher+student forward pass over the same response tokens.
    scored = score_teacher_student(
        model=model,
        teacher_sequences=teacher_sequences,
        student_sequences=student_sequences,
        pad_id=pad_id,
    )
    # Stack per-row tensors into [N, response_len] tensors so the
    # objective primitives (vacs_token_advantages, etc.) get the shape
    # they expect. Response lengths may differ across rows, so we
    # right-pad with zeros -- the shape uniformity is what the
    # objective primitives require for the batched multiply.
    def _stack(rows):
        max_len = max(r.shape[0] for r in rows)
        out = torch.zeros(len(rows), max_len, device=rows[0].device)
        for i, r in enumerate(rows):
            out[i, :r.shape[0]] = r
        return out

    teacher_token_logps_2d = _stack(scored.teacher_token_logps)
    student_token_logps_2d = _stack(scored.student_token_logps)

    # ---- RL token-level advantages.
    if advantages is None:
        advantages = normalize_advantages(rewards, eps=adv_eps)
    advantages = advantages.to(model.device)
    # `_stack` right-pads variable-length rows to a common width with
    # zeros; a mask over the true (unpadded) response lengths is needed
    # before any per-sequence mean, or shorter rollouts get diluted by
    # zero-padding in the denominator.
    response_lens = [r.shape[0] for r in scored.student_token_logps]
    max_len = student_token_logps_2d.shape[1]
    valid_mask = torch.zeros(len(response_lens), max_len, device=student_token_logps_2d.device)
    for i, length in enumerate(response_lens):
        valid_mask[i, :length] = 1.0

    if use_token_credit:
        token_adv = vacs_token_advantages(
            advantages,
            teacher_token_logps_2d,
            student_token_logps_2d,
            credit_mix=credit_mix,
            clip_epsilon=clip_epsilon,
        )
        rho = float(credit_mix)
        token_weight_mean = (
            (1.0 - rho) + rho * torch.exp(
                torch.sign(advantages).unsqueeze(1)
                * (teacher_token_logps_2d - student_token_logps_2d).detach()
            ).clamp(min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
        )[valid_mask.bool()].mean().item()
    else:
        # No token credit: per-token advantage == global advantage.
        # Shape [N, response_len] (each token in a rollout gets the same value).
        token_adv = advantages.unsqueeze(1).expand(-1, max_len).contiguous()
        token_weight_mean = 1.0

    # doc Section 3.1's boxed RL loss is plain per-token REINFORCE --
    # no PPO ratio/clip anywhere in the formula (unlike CRISP's own RL
    # term, which uses a clip for a documented, separate reason). This
    # implements that literally: -(1/T) sum_t a_tilde_t * log pi_theta,
    # batch-averaged, with padded positions excluded from both the sum
    # and the T normalizer via valid_mask.
    per_token_rl = token_adv * student_token_logps_2d * valid_mask
    per_example_rl_loss = -(per_token_rl.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0))
    loss_rl = per_example_rl_loss.mean()

    # ---- Soft reward gate on the SD term.
    if use_soft_gate:
        gate = soft_reward_gate(rewards, slope=soft_gate_slope, threshold=reward_threshold)
        gate_mean = float(gate.mean().item())
    else:
        gate = torch.ones_like(rewards)
        gate_mean = 1.0

    loss_sd = sd_weight * reduce_gated_kl(
        scored.teacher_per_token_kl, gate, reduction="sequence_sum"
    )

    # ---- Two gradient vectors via autograd.grad (never .backward()).
    g_rl_list = compute_grads(loss_rl, params, retain_graph=True)
    if loss_sd.requires_grad:
        g_sd_list = compute_grads(loss_sd, params, retain_graph=False)
    else:
        g_sd_list = [torch.zeros_like(p) for p in params]

    g_rl_flat = flatten_grads(g_rl_list)
    g_sd_flat = flatten_grads(g_sd_list)

    # ---- Adaptive mixing coefficient (from PRIOR committed state).
    rl_norm_sq = float(g_rl_flat.detach().pow(2).sum().item())
    sd_norm_sq = float(g_sd_flat.detach().pow(2).sum().item())
    cos_value = 0.0
    rl_norm = float(g_rl_flat.detach().norm().item())
    sd_norm = float(g_sd_flat.detach().norm().item())
    if rl_norm > 0.0 and sd_norm > 0.0:
        cos_value = float(torch.dot(g_rl_flat.detach(), g_sd_flat.detach()).item()
                          / (rl_norm * sd_norm))
        cos_value = max(-1.0, min(1.0, cos_value))

    if state.disabled:
        lambda_eff: Optional[float] = float(base_lambda)
        combined_flat = g_rl_flat + lambda_eff * g_sd_flat
        auxiliary_projected = g_sd_list
        conflicted = bool((torch.dot(g_rl_flat, g_sd_flat) < 0.0).item())
    else:
        # IMPORTANT: compute lambda_eff from the COMMITTED EMA, not from
        # the just-observed norm -- a micro-batch cannot update the
        # coefficient used in its own optimizer step.
        lambda_eff = compute_lambda_effective(state, base_lambda)
        if use_asymmetric_projection:
            surgery = asymmetric_pcgrad_combine(
                g_rl_list, g_sd_list, params, lam=lambda_eff
            )
            combined_flat = flatten_grads(surgery.combined)
            auxiliary_projected = surgery.auxiliary_projected
            conflicted = surgery.conflicted
        else:
            surgery_dot = float(torch.dot(g_rl_flat, g_sd_flat).item())
            conflicted = bool(surgery_dot < 0.0)
            combined_flat = g_rl_flat + lambda_eff * g_sd_flat
            auxiliary_projected = g_sd_list

    combined_list = unflatten_grads(combined_flat, params)

    # ---- The same kind of metrics block the plan calls out, prefixed
    # "vacs/..." so they're namespaced in the run logger.
    metrics = {
        "vacs/loss_rl": float(loss_rl.item()),
        "vacs/loss_sd": float(loss_sd.item()) if loss_sd.requires_grad else 0.0,
        "vacs/token_weight_mean": float(token_weight_mean),
        "vacs/gate_mean": gate_mean,
        "vacs/cos_rl_sd": float(cos_value),
        "vacs/conflicted": float(conflicted),
        "vacs/rl_grad_norm": rl_norm,
        "vacs/sd_grad_norm": sd_norm,
        "vacs/variance_multiplier": (
            float(lambda_eff) / float(base_lambda) if base_lambda > 0 else 1.0
        ),
        "vacs/lambda_eff": float(lambda_eff) if lambda_eff is not None else 1.0,
    }

    return VacsComponents(
        loss_rl=loss_rl,
        loss_sd=loss_sd,
        rl_grad=g_rl_list,
        sd_grad=g_sd_list,
        auxiliary_projected=auxiliary_projected,
        lambda_eff=lambda_eff,
        metrics=metrics,
    )


def vacs_step_from_arrays(
    *,
    model,
    teacher_sequences: list[ScoredSequence],
    student_sequences: list[ScoredSequence],
    pad_id: int,
    params: list[torch.nn.Parameter],
    rewards: torch.Tensor,
    base_lambda: float,
    adv_eps: float,
    clip_epsilon: float,
    sd_weight: float,
    soft_gate_slope: float,
    reward_threshold: float,
    credit_mix: float,
    use_token_credit: bool,
    use_soft_gate: bool,
    advantages: torch.Tensor | None = None,
    use_asymmetric_projection: bool = True,
    state: Optional[VacsAdaptiveState] = None,
) -> VacsStepResult:
    """One VACS micro-batch end-to-end: forward, losses, gradient surgery, metrics.

    Returns the final combined gradient (aligned 1:1 with `params`) and
    a `VacsComponents` block for diagnostics. Does NOT call `optimizer.step`
    and does NOT mutate `params[i].grad` -- the caller is the common
    training loop, which accumulates gradients across micro-batches and
    steps the optimizer after `grad_accum_steps` calls.
    """
    if state is None:
        state = VacsAdaptiveState()
    components = build_vacs_components(
        model=model,
        teacher_sequences=teacher_sequences,
        student_sequences=student_sequences,
        pad_id=pad_id,
        params=params,
        rewards=rewards,
        base_lambda=base_lambda,
        advantages=advantages,
        adv_eps=adv_eps,
        clip_epsilon=clip_epsilon,
        sd_weight=sd_weight,
        soft_gate_slope=soft_gate_slope,
        reward_threshold=reward_threshold,
        credit_mix=credit_mix,
        use_token_credit=use_token_credit,
        use_soft_gate=use_soft_gate,
        use_asymmetric_projection=use_asymmetric_projection,
        state=state,
    )
    # Build the combined gradient from the components already in hand.
    rl_norm_sq = float(flatten_grads(components.rl_grad).detach().pow(2).sum().item())
    sd_norm_sq = float(flatten_grads(components.sd_grad).detach().pow(2).sum().item())
    cos_value = components.metrics["vacs/cos_rl_sd"]
    state.observe(norm_sq=rl_norm_sq, cos_value=cos_value, sd_norm_sq=sd_norm_sq)
    gradients = unflatten_grads(
        flatten_grads(components.rl_grad) + float(components.lambda_eff or 1.0)
        * flatten_grads(components.auxiliary_projected),
        params,
    )
    return VacsStepResult(
        gradients=gradients,
        components=components,
        metrics=dict(components.metrics),
        state=state,
    )


def commit_vacs_state(state: VacsAdaptiveState, *, rl_norm_sq: float, sd_norm_sq: float,
                      cos_value: float) -> None:
    """Convenience wrapper used by the training loop's post-step hook."""
    state.commit(rl_grad_norm_sq=rl_norm_sq, sd_grad_norm_sq=sd_norm_sq, cos_value=cos_value)
