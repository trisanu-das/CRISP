# Ablations reference

Every ablation in this codebase is declared in
`experiments/ablations.yaml` and run by `scripts/run_ablations.py`.
This document records: what each suite contains, what variable each
variant changes, the hypothesis each variant tests, and the
interpretation constraints.

## Suite overview

There are four suites. They are intentionally separated:

- `controls` — baselines the new methods are benchmarked against.
- `vacs-core` — VACS full method + one-factor ablations.
- `cibo-fixed` — CIBO v2 FIXED-beta full method + one-factor
  ablations.
- `cibo-adaptive` — CIBO v2 ADAPTIVE-beta full method + matched
  fixed-beta control + one-factor ablations.

CIBO fixed and CIBO adaptive are SEPARATE suites. Mixing them
under the same label is invalid because the two modes have
different update rules and different theoretical interpretation.
A comparison across modes is meaningful only when both variants
appear in the same suite.

## `controls` suite

Variants:

- `crisp`: the original CRISP method (control). Untouched by this
  project.
- `reinforce_pp`: REINFORCE++ baseline (no self-distillation).
- `opsd`: OPSD baseline (self-distillation only, no RL).
- `grpo_k1`: a single-rollout, global-normalized policy-gradient budget
  control. It is retained under the historical manifest key but is not
  described as genuine group-relative GRPO.
- `grpo_k8`: genuine GRPO with eight rollouts per prompt.
- `crisp_lambda_sweep`: CRISP with `lambda_max = 0.5` instead of
  the default 1.0. The plan calls out that the original CRISP's
  sensitivity to lambda is itself worth characterising.

No hypothesis is tested by this suite; it exists to provide
baseline numbers that VACS and CIBO results are compared against.

## `vacs-core` suite

Variants, each changing exactly one VACS-specific knob:

- `full`: VACS with all knobs at the operational defaults:
  `rho = 0.25`, `soft_gate_slope = 12`, `reward_threshold = 0.5`,
  adaptive mixing enabled (EMA decay 0.95, multiplier clamp
  `[0.1, 5.0]`), asymmetric PC-Grad on, `sd_weight = 1.0`.
- `no_token_credit`: `vacs.token_credit_mix: 0.0`. RL reduces to
  the CRISP sequence-mean advantage. **Hypothesis**: token credit
  contributes a measurable share of the gains; without it, VACS
  should look more like CRISP. **Constraint**: only one knob
  changed; other knobs still at the full-variant defaults.
- `no_soft_gate`: `vacs.use_soft_gate: false`. The gate is explicitly
  replaced by 1.0 for every rollout. **Hypothesis**: the smooth sigmoid
  gate is necessary; a hard "always on" gate should look like
  plain CRISP SD loss.
- `fixed_mixing`: `vacs.adaptive_mix.enabled: false`. `lambda_eff =
  lambda_max` constant. **Hypothesis**: the variance ratio + cosine
  scaling of the SD coefficient help; without them, VACS should
  look like CRISP at the same lambda.
- `no_asym_surgery`: `vacs.use_asymmetric_projection: false`; gradients
  are added without projection. **Hypothesis**:
  protecting the RL direction is necessary; without it, the
  expected gain disappears.
- `no_sd`: `vacs.sd_weight: 0.0`. Reduces to plain RL with token
  credit. **Hypothesis**: SD is the primary source of gains.

Every variant has metric_keys defined in the manifest. The runner
emits them; downstream analysis can verify which variant produced
which value.

## `cibo-fixed` suite

Variants, each changing exactly one CIBO v2 knob in fixed-beta mode:

- `full`: CIBO v2 with `beta = 0.10`, `credit_lambda = 0.25`,
  `anchor_alpha = 0.01`, `ib_reduction = sequence_sum`. This is
  the closest analogue to the design doc's idealized fixed-beta
  setting.
- `no_token_credit`: `cibo_v2.credit_lambda: 0.0`. RL advantage is
  the global normalized advantage. **Hypothesis**: tanh credit
  is necessary.
- `no_ib`: `cibo_v2.beta: 0.0`. The IB term contributes nothing.
  **Hypothesis**: the IB term is the primary source of gains.
- `no_anchor`: `cibo_v2.anchor_alpha: 0.0`. The EMA-anchored
  regularizer contributes nothing. **Hypothesis**: the anchor
  prevents drift that the RL+IB terms alone would cause.
- `token_mean_ib`: `cibo_v2.ib_reduction: token_mean`. **Hypothesis**:
  the reduction choice matters; switching from `sequence_sum` to
  `token_mean` is a meaningful ablation.

## `cibo-adaptive` suite

Variants:

- `full`: CIBO v2 adaptive-beta mode, with the controller's
  reference EMA built during a 20-step warm-up then frozen.
- `matched_fixed`: same config but `beta_mode = fixed`,
  `beta = 0.10`. This is the matched control for the adaptive
  variant: same seed, same model, same data; only the controller
  differs. **Hypothesis**: the controller's updates help.
- `controller_disabled`: adaptive mode but
  `cibo_v2.beta_step_size: 0.0` after warmup completes (forcing
  the controller to no-op). **Hypothesis**: the post-warmup
  adaptive updates themselves contribute.

The matched control is the most important variant in this suite:
without it, the adaptive-mode result is uninterpretable.

## Manifest discipline

`tests/test_ablation_manifest.py` enforces three structural
constraints:

1. Every variant's `method` is in `SUPPORTED_METHODS`.
2. Every variant's `base_config` is a file under `config/`.
3. Each ablation's `overrides` changes ONLY allow-listed keys
   (one-factor-at-a-time discipline). The allow-list is keyed on
   the VARIANT's method (not the suite's full variant's method),
   so e.g. a `grpo_k8` variant in the controls suite is allowed
   to override `rollout.k` even when `controls.full_variant` is
   `crisp`.

If you need to study interactions between knobs, edit the
manifest. Don't multi-override at the command line.

## Runner invariants

`scripts/run_ablations.py` guarantees:

- One process at a time (`--jobs 1`) by default.
- Refuses `--jobs > 1` without `--allow-parallel`.
- Validates the manifest + configs BEFORE any model load.
- Writes `run_metadata.json` and `resolved_config.yaml` into
  every run directory.
- Fails fast on a non-zero child command and preserves the
  child's `run.log`.

## What "ablation" means here

An ablation in this codebase is a **controlled comparison**, not
a result. The runner does not aggregate metrics across variants
or produce a leaderboard; that's an analysis step you do
downstream. The runner's job is to produce, for every
`{variant, seed}` combination, a self-contained run directory
whose contents can be re-derived from the manifest + the seed.