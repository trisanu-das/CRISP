# CRISP, VACS-CRISP and CIBO-CRISP v2

A research codebase extending an isolated reimplementation of CRISP
(CRISP stands for "Critic-free Reward-Integrated Self-distillation Policy Optimization"). This repository adds two new methods on top of
the existing baselines (CRISP, REINFORCE++, OPSD, GRPO):

- **VACS-CRISP** — variance/gradient-energy-aware causal surgery with
  token credit, asymmetric gradient projection, soft reward gating, and
  bounded adaptive mixing. See
  `docs/methods.md` for the math and `docs/ablations.md` for the
  ablation suite.
- **CIBO-CRISP v2** — single composite loss (`loss_rl + beta*loss_ib
  + alpha*loss_anchor`) with **no** gradient surgery, an EMA-anchored
  regularizer, and a bounded adaptive-beta controller as an ablation.

## Status

**Research code.** This repository contains the full implementations
of VACS-CRISP and CIBO-CRISP v2, a bounded Kaggle smoke workflow for
each, and an offline ablation manifest. **No benchmark results are
claimed here.** Before drawing scientific conclusions, run the
ablations yourself with your own compute, your own seeds, and your
own datasets. The `scripts/run_ablations.py` runner is designed to
make that straightforward.

A detailed migration/change report is in `MODIFICATION_REPORT.md`, and
extended benchmark instructions are in `docs/extended_experiments.md`.

The implementation is validated by the repository test suite and dry-run
routing checks. GPU training and benchmark results are intentionally not
claimed by this repository; run the smoke profile and paper configurations
on your target hardware before using results in a paper.

## Project layout

```text
crisp-vacs-cibo/
├── README.md                   # this file
├── PROVENANCE.md               # seed commit + provenance notes
├── train_launcher.py           # dispatcher (crisp/grpo/opsd/reinforce_pp/vacs/cibo-v2/sft)
├── predownload.py              # pre-download a model + tokenizer before launching
├── config/
│   ├── crisp_7b.yaml           # corrected full CRISP config
│   ├── crisp_pilot.yaml        # tiny smoke/pilot config
│   ├── vacs_7b.yaml            # VACS full method 7B
│   ├── vacs_kaggle_smoke.yaml  # VACS bounded Kaggle smoke (3 steps, 8 examples)
│   ├── cibo_v2_7b.yaml         # CIBO v2 FIXED-beta 7B
│   ├── cibo_v2_adaptive_beta_7b.yaml  # CIBO v2 ADAPTIVE-beta 7B
│   └── cibo_v2_kaggle_smoke.yaml  # CIBO v2 bounded Kaggle smoke
├── crisp/
│   ├── training/               # new modules (VACS, CIBO v2) + shared objectives
│   │   ├── teacher_student.py  # response-aligned scoring helper
│   │   ├── objectives.py       # token credit, soft gate, gated forward-KL, tanh credit
│   │   ├── gradient_surgery.py # asymmetric PC-Grad (legacy pcgrad.py untouched)
│   │   ├── ema_adapter.py      # EMA anchor (Task 7)
│   │   ├── beta_controller.py  # bounded adaptive beta (Task 7)
│   │   ├── vacs_state.py       # VACS adaptive mixing state
│   │   ├── vacs_step.py        # VACS per-micro-batch step
│   │   ├── vacs_train.py       # VACS training entry point
│   │   ├── cibo_v2_step.py     # CIBO v2 single composite step
│   │   ├── cibo_v2_train.py    # CIBO v2 training entry point
│   │   └── common_loop.py      # shared training loop (post_optimizer_step + checkpoint_state hooks)
│   ├── data/build_dataset.py   # load_train_dataset now accepts train_subset_size
│   └── utils/{config,validation}.py
├── experiments/ablations.yaml  # manifest of manifest-driven ablations
├── scripts/
│   ├── run_ablations.py        # serial runner over the ablation manifest
│   └── kaggle_smoke.py         # bounded Kaggle smoke runner
└── tests/                      # pytest suite (see test files)
```

## Method comparison

| Method       | RL signal             | Self-distillation term      | Surgery             | Beta        |
|--------------|-----------------------|------------------------------|---------------------|-------------|
| CRISP        | global-effective-batch policy gradient | correctness-gated forward KL | symmetric PC-Grad | n/a |
| VACS-CRISP   | sign-aware token-weighted policy gradient | soft-gated forward KL | asymmetric projection protecting RL | n/a |
| CIBO-CRISP v2| sign-preserving token policy gradient | forward-KL IB proxy + EMA anchor | none (single scalar loss) | fixed or adaptive |
| REINFORCE++  | global-effective-batch policy gradient | none | none | n/a |
| OPSD         | none (SD only) | forward KL, teacher detached | none | n/a |
| GRPO         | group-relative policy gradient (`k>=2`) | none | none | n/a |

VACS keeps the original CRISP loss shape (loss_rl + lambda * loss_sd)
and adds four orthogonal improvements: token credit, soft reward
gating, asymmetric gradient surgery, and bounded adaptive mixing.
CIBO-CRISP v2 is a *different* design: a single scalar loss with no
PC-Grad at all, an EMA-anchored regularizer to keep the student from
drifting too far from a recent snapshot, and a beta controller that
adapts the IB coefficient within strict bounds.

## Setup

This project targets Python 3.11+ and CUDA-capable GPUs. The
minimum dependencies are `torch`, `transformers`, `datasets`, `peft`,
`accelerate`, `sympy`, `pyyaml`, `pytest`, `numpy`, and (for math verification)
`math-verify`. The smoke configs target `Qwen/Qwen2.5-0.5B-Instruct`,
which fits on a single 16 GB GPU at 4-bit precision.

```bash
# Optional: pre-download the smoke model so the smoke run never
# touches the network.
python predownload.py Qwen/Qwen2.5-0.5B-Instruct
```

For 7B runs you will need roughly 24 GB of VRAM (or a 4-bit quantized
adapter setup; see `config/vacs_7b.yaml`).

## Quickstart

The launcher dispatches on a positional `method` argument:

```bash
# Original CRISP 7B (control; the legacy method, untouched)
python train_launcher.py crisp --config config/crisp_7b.yaml

# VACS 7B (full method)
python train_launcher.py vacs --config config/vacs_7b.yaml

# CIBO v2 fixed-beta 7B
python train_launcher.py cibo-v2 --config config/cibo_v2_7b.yaml

# CIBO v2 adaptive-beta 7B
python train_launcher.py cibo-v2 --config config/cibo_v2_adaptive_beta_7b.yaml

# Completion-only SFT, then initialize an RL method from the adapter
python train_launcher.py sft --config config/sft_7b.yaml
python train_launcher.py grpo --config config/paper_7b.yaml --k 8 \
  --override model.initial_adapter=runs/sft_7b/final

# Multi-GPU manual-gradient launch
NUM_GPUS=2 METHOD=crisp CONFIG=config/paper_7b.yaml \
  bash scripts/launch_distributed.sh
```

For most knobs you can override at the command line:

```bash
python train_launcher.py vacs --config config/vacs_7b.yaml \
    --override=seed=42 --override=vacs.token_credit_mix=0.5
```

## Kaggle smoke test

The Kaggle smoke runs are bounded (3 steps, 8 examples, max 128 new
tokens) and explicitly designed to verify plumbing, not learning.
Sparse or zero rewards in a 3-step run are NOT a regression.

```bash
# VACS smoke
python scripts/kaggle_smoke.py --method vacs

# CIBO v2 smoke (with predownload)
python scripts/kaggle_smoke.py --method cibo-v2 --predownload

# Dry-run: see the resolved command without touching the network
python scripts/kaggle_smoke.py --method vacs --dry-run
python scripts/kaggle_smoke.py --method cibo-v2 --dry-run
```

Successful smoke runs produce: a `final/` adapter checkpoint, a
run JSONL containing at least one record with the method's
required metric key (`vacs/loss_rl` or `cibo/loss_total`), and at
least one non-empty response in `generations.jsonl`. The runner
prints a PASS/FAIL report after the run.

A non-success indicator is **not** the reward being zero (that's
expected at 3 steps). A non-success indicator is: the required
metric key missing, a non-finite value in `metrics.jsonl`, or no
checkpoint produced. If you see `HARD FAILURE: method is in
SUPPORTED_METHODS` or `smoke config exists`, that means the method
or config name is wrong (typo, file renamed) and the run never
started.

## Ablations

Ablations are declared in `experiments/ablations.yaml` and executed
by `scripts/run_ablations.py`. The runner is serial by default
(`--jobs 1`); it refuses `--jobs > 1` without `--allow-parallel`
because on a single GPU concurrent processes will OOM.

```bash
# Dry-run a full VACS-core suite (no model download, no training)
python scripts/run_ablations.py --suite vacs-core \
    --config config/vacs_7b.yaml --seeds 0 --dry-run

# Run one variant, one seed
python scripts/run_ablations.py --suite vacs-core --variant full \
    --config config/vacs_7b.yaml --seeds 0

# Run the CIBO v2 adaptive suite across three seeds
python scripts/run_ablations.py --suite cibo-adaptive \
    --config config/cibo_v2_adaptive_beta_7b.yaml --seeds 0,1,2
```

**Ablations are comparisons, not results.** A run that fails
because of plumbing is a bug to fix, not a finding to report. A
successfully-executed variant with the same seed across an
ablation pair (e.g. `vacs-core/full` vs `vacs-core/no_token_credit`)
gives you a controlled comparison. The manifest enforces
one-factor-at-a-time discipline automatically; if you need to
study interactions between knobs, edit the manifest rather than
multi-overriding.

The four suites, in priority order, are:

- `controls`: CRISP, REINFORCE++, OPSD, a single-rollout
  global-policy-gradient control, GRPO `k=8`, and a CRISP lambda sweep.
- `vacs-core`: VACS full method + five one-factor ablations.
- `cibo-fixed`: CIBO v2 fixed-beta full method + four one-factor
  ablations.
- `cibo-adaptive`: CIBO v2 adaptive-beta full method + matched
  fixed control + controller-disabled ablation.

CIBO fixed-beta and adaptive-beta live in **separate** suites
because mixing them under the same label is invalid. See
`docs/ablations.md`.

## Theory caution

- The KL term in both methods is a *variational KL proxy*, not the
  exact information-bottleneck integral. It is computed as
  `forward_kl(teacher || student)` over the response span, with the
  teacher distribution detached, which
  introduces a known bias. This is documented in the code
  (`crisp/training/objectives.py:forward_kl_per_token`).
- The CIBO v2 design's theoretical guarantees (variational
  convergence under stated idealized assumptions) are NOT claimed
  to hold under our practical implementation, which uses
  stop-gradients and an EMA-anchored target. The fixed-beta mode
  is the closest analogue to the design's idealized setting.
- The adaptive beta controller is an **empirical** ablation only.
  There is no theoretical stationary-objective guarantee that
  adaptive updates of `beta` converge. Use the matched fixed-beta
  control (provided in the `cibo-adaptive` suite) to evaluate it.

## Metrics glossary

The names emitted by the trainer follow a `<method>/<name>` scheme
so a single dashboard can show method-specific metrics without
collisions. See `docs/methods.md` for full lists. Quick reference:

- `vacs/*`: `loss_rl`, `loss_sd`, `lambda_eff`, `cos_rl_sd`,
  `conflicted`, `rl_grad_norm`, `sd_grad_norm`, `token_weight_mean`,
  `gate_mean`, `variance_multiplier`.
- `cibo/*`: `loss_total`, `loss_rl`, `loss_ib`, `loss_anchor`,
  `beta`, `beta_mode`, `beta_reference_kl`, `baselined_kl`,
  `credit_multiplier_mean`, `ib_gate_mean`, `kl_per_token_mean`,
  `anchor_alpha`, `ema_decay`.
- `reward_mean`, `reward_std`, `response_len_mean`: emitted by
  every method.

## Checkpoint and resume

The shared `common_loop.run_training_loop` accepts a
`checkpoint_state()` hook (added in Task 6) that returns a dict
containing every per-step state the method needs to resume.
VACS-CRISP and CIBO-CRISP v2 both populate this hook (with the
EMA adapter and beta controller states respectively). On resume,
`checkpoint_state()` returns the same dict so the `load_state_dict`
path on each state object is reachable.

## Known limitations

- The CRISP control path in the launcher is the upstream
  reimplementation's existing path. We have NOT modified it. If
  the upstream reimplementation has bugs, this codebase
  inherits them.
- The EMA adapter anchor holds a copy of every trainable LoRA
  parameter, doubled for the duration of the `ema_target_forward`
  context. For rank-8 LoRA on a 7B model this is a few MB, not a
  problem. For a full-finetune, this would be a real cost.
- The smoke runner's CUDA check is informational; on a Kaggle
  notebook CUDA is always available, but on a dev box we don't
  hard-fail to keep `--dry-run` useful.
- `tests/test_logprobs.py::test_two_rows_with_different_context_lengths`
  was already failing in the seed commit (`e6228dd`) due to a torch
  2.13 issue with `vocab=3` token-id lookups. We have not touched
  this test. It's pre-existing, unrelated to VACS or CIBO v2, and
  does not affect the runtime behaviour of either method.

## Further reading

- `docs/methods.md` - formulas, reductions, state-update timing,
  and per-method metric definitions.
- `docs/ablations.md` - every suite in `experiments/ablations.yaml`,
  the changed variable, the hypothesis each variant tests, and
  interpretation constraints.
- `docs/kaggle-smoke.md` - end-to-end Kaggle sequence, predownload
  fallback, artifact inspection, and cleanup.
