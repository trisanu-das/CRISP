# Methods reference

This document records the exact formulas, reductions, state-update
timing, and metric definitions for each method in the launcher.
For usage, see `README.md`; for ablation hypotheses, see
`docs/ablations.md`.

## Notation

- `T` — number of response tokens.
- `V` — vocabulary size.
- `B` — micro-batch size (number of prompts).
- `a_t` — per-token advantage at token `t`.
- `r_i` — scalar reward for prompt `i`.
- `q_{i,t}` — teacher (reference) distribution at the response
  position for prompt `i`, token `t`. Detached.
- `p_{i,t}` — student distribution at the response position for
  prompt `i`, token `t`. Differentiated w.r.t. student parameters.

## CRISP (the existing baseline; not modified by this project)

- `loss_rl = -E_{i,t} [ min(ratio_i_t * a_t,
                            clip(ratio_i_t, 1-eps, 1+eps) * a_t) ]`
- `loss_sd = sum_t KL_stop_grad(q_{i,t} || p_{i,t})` per response,
  summed (NOT meaned) over the response span.
- `loss = loss_rl + lambda_max * loss_sd`.
- Symmetric PC-Grad: `g_combined = pcgrad_combine(g_rl, g_sd)`.

## VACS-CRISP

**Loss shape is unchanged from CRISP** (`loss_rl + lambda * loss_sd`).
VACS adds four orthogonal improvements:

1. **Token credit** (`objectives.py::vacs_token_advantages`):
   `c_t = sg[log p_T - log p_S]` (both sides detached), `w_t =
   clip(exp(sign(a_i) * c_t), 1-eps, 1+eps)`, `a_tilde_t = a_i * [(1 -
   rho) + rho * w_t]`. The `sign(a_i)` factor is load-bearing: it's what
   makes a teacher-endorsed token get *less* penalty inside an
   otherwise-wrong trajectory (a_i < 0) rather than the same boost it
   would get inside a correct one (a_i > 0). `a_tilde_t` (not the raw
   advantage) is what actually appears in the RL loss -- VACS's RL term
   is plain per-token REINFORCE (`-(1/T) sum_t a_tilde_t * log
   pi_theta(y_t|...)`), not a PPO-clipped surrogate; CRISP's RL term
   uses a clip, VACS's doesn't.

2. **Soft reward gate**: `objectives.py::soft_reward_gate` returns
   `g_i = 1 - sigmoid(slope * (r_i - threshold))` directly -- i.e. the
   *distillation weight* itself (large for low reward, small for high
   reward), not `sigmoid(...)` on its own. `g_i` is in `(0, 1)` and
   multiplies the per-prompt contribution to the SD loss. `slope = 0`
   collapses to a constant `0.5` gate.

3. **Asymmetric gradient surgery**: `g_combined = g_rl +
   lambda_eff * project_away(g_sd, g_rl)`. The RL direction is
   protected exactly; only the SD direction is projected.

4. **Bounded adaptive mixing** (`vacs_state.py::compute_lambda_effective`):
   `lambda_eff = lambda_max * clip(sqrt(rl_norm_sq_ema / (sd_norm_sq_ema
   + eps)), mult_min, mult_max) * (1 + clip(cos_ema, -1, 1)) / 2`. Both
   the norm-ratio term *and* the cosine-similarity term are combined
   into the multiplier (not norm-ratio alone -- cosine similarity is
   not just an exposed-but-unused metric here). `vacs.adaptive_mix.enabled
   = false` collapses this to `lambda_eff = lambda_max` constant.

### Reductions

- `loss_rl` is mean-over-tokens (length-normalized, per doc Section
  3.1's explicit `1/T`) per sequence, then mean over the batch -- not
  a plain sum.
- `loss_sd` is summed (NOT meaned) over response tokens per
  prompt, then averaged over prompts.
- The plan explicitly forbids silently switching reductions:
  VACS uses `sequence_sum` for SD (matches CRISP). A
  `token_mean` reduction is available as an explicit ablation.

### State-update timing

The `VacsAdaptiveState` (EMA of `grad_norm_rl`, `grad_norm_sd`,
`cos_rl_sd`, `last_variance_multiplier`) is updated in the
**post-optimizer-step hook** (`common_loop.post_optimizer_step`),
not inside the micro-batch step. This is enforced by an explicit
test: `tests/test_common_loop_hooks.py`. The reason is that the
just-applied gradient signal is the one we want EMA-tracked; if
the EMA were updated per-micro-batch, the multi-step adaptive
estimate would be inconsistent across a single optimizer step.

### Metrics

- `vacs/loss_rl` — RL term after token credit.
- `vacs/loss_sd` — SD term after soft reward gating.
- `vacs/lambda_eff` — the (possibly EMA-adapted) lambda applied
  to the SD term this step.
- `vacs/cos_rl_sd` — cosine similarity between `g_rl` and `g_sd`
  (EMA-tracked; informational).
- `vacs/conflicted` — 1.0 if PC-Grad triggered a projection, 0.0
  otherwise. Summed across all param groups.
- `vacs/rl_grad_norm`, `vacs/sd_grad_norm` — per-step norms (used
  by the adaptive mixing; also logged for inspection).
- `vacs/token_weight_mean` — mean of `(1 - rho) + rho *
  clip(exp(sign(a_i) * c_t), 1-eps, 1+eps)` across valid (unpadded)
  tokens (equals 1.0 exactly when `rho = 0`, by construction).
- `vacs/gate_mean` — mean soft-gate (= distillation weight) value;
  equals 0.5 when `slope = 0`, equals `1 - sigmoid(slope *
  (mean_reward - threshold))` in the typical case.
- `vacs/variance_multiplier` — current adaptive mixing factor.

## CIBO-CRISP v2

**Single composite loss.** NO PC-Grad, NO gradient surgery.

- `loss_rl = -mean_i [ (1/T_i) sum_t a_tilde_{i,t} * log pi_theta(y_t|x,y_{<t}) ]`
  -- plain per-token REINFORCE using the tanh-based token advantage
  below, NOT a PPO-clipped ratio (doc Section 5.1.1's boxed formula has
  no ratio/clip, unlike CRISP's own RL term).
- `loss_ib = sum_t forward_kl(q_{i,t} || p_{i,t})`
  where `q_{i,t}` is detached.
- `loss_anchor = H(p_EMA, p_live)` over the full response-token
  vocabulary distributions. The EMA target is computed under
  `ema_target_forward` and detached; gradients flow only through the
  live student distribution.
- `loss_total = loss_rl + beta * loss_ib + alpha * loss_anchor`.

One single `compute_grads(loss_total, params)` call; the optimizer
sees one gradient per parameter.

### Tanh credit

`a_tilde_t = a_t * (1 + lambda * tanh(c_t))`. The multiplier is
bounded in `[1 - lambda, 1 + lambda]`, so `a_tilde_t` has the
**same sign** as `a_t` for any `c_t`. Sign preservation is
structurally guaranteed, not empirical. Tests verify this.

`cibo_v2.credit_lambda: 0.0` is the documented "no token credit"
ablation: `a_tilde_t = a_t` (the global normalized advantage is
the per-token advantage).

### IB reductions

- `sequence_sum` (default): sum over response tokens, then mean
  over prompts.
- `token_mean`: mean over `(prompt, token)`. Same shape, different
  scale. Both are documented as explicit reductions; switching
  between them is an explicit ablation.

### Adaptive-beta controller

The controller holds a running reference KL EMA (frozen after a
warm-up window) plus fast reward/KL EMAs, and updates `beta` once
per optimizer step (not per micro-batch) per doc Section 7's
bounded sign controller comparing leakage change with reward change:

- `baselined_kl = kl_ema - reference_kl`
- `delta_reward = reward_ema - reward_ema_prev_step`
- `delta_baselined_kl = baselined_kl - baselined_kl_prev_step`
- `signal = delta_baselined_kl - delta_reward`
- `beta = clip(beta * (1 + step_size * sign(signal)), beta_min, beta_max)`

Only the sign of the baselined difference drives the update, and a small
tolerance suppresses numerical-noise updates. The multiplicative step is
bounded by `beta_min` and `beta_max`. No update is attempted until there are two
post-warm-up observations to form deltas from.

The reference EMA is built during a warm-up window
(`beta_reference_warmup_steps`, default 20) and frozen afterwards.
The controller's `step()` is called from the post-optimizer-step
hook; it never runs inside the micro-batch step. `observe(reward,
kl_value)` must be called with the batch's real mean reward each
step -- the ratio is meaningless without it.

### Metrics

- `cibo/loss_total`, `cibo/loss_rl`, `cibo/loss_ib`, `cibo/loss_anchor`
  — the four scalar losses.
- `cibo/beta` — current `beta` value.
- `cibo/beta_mode` — `"fixed"` or `"adaptive_sign"`. Logged once.
- `cibo/beta_reference_kl` — the controller's reference EMA.
- `cibo/baselined_kl` — current KL minus reference.
- `cibo/credit_multiplier_mean` — mean of `1 + lambda * tanh(c_t)`.
- `cibo/ib_gate_mean` — mean gate applied to IB (1.0 when
  `use_soft_gate: false`).
- `cibo/kl_per_token_mean` — average per-token KL.
- `cibo/anchor_alpha`, `cibo/ema_decay` — the anchor config
  values, logged once.

## Why CIBO v2 has no PC-Grad

The CRISP design's PC-Grad is a defense against destructive
interference between the RL signal and the SD signal. CIBO v2
collapses both signals into a single scalar loss, so there is
no interference to project against — `compute_grads` is called
exactly once. The EMA-anchored regularizer serves a different
role: it keeps the student close to a recent snapshot rather
than protecting the RL direction against the SD direction.

## Shared `common_loop` hooks

Both methods use the shared `run_training_loop` with two
post-optimizer-step hooks added in Task 6:

- `post_optimizer_step(model, aggregated_step_metrics)` —
  VACS commits its adaptive state; CIBO v2 commits the beta
  controller (`ctrl.step()`) and advances the EMA adapter
  (`ema.update()`). Both happen **once per optimizer step**,
  not once per micro-batch. Tests in
  `tests/test_common_loop_hooks.py` enforce this.

- `checkpoint_state()` — returns a dict containing every state
  object the method needs to resume. VACS returns the VACS
  state; CIBO v2 returns the EMA adapter + beta controller
  states. Resume path is `load_state_dict(state)` on each
  object.