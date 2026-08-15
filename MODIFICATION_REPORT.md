# Modification report

This document describes the corrections and extensions applied to the uploaded
CRISP repository. The modifications were made directly in this repository; no
upstream GitHub branch was changed.

## Validation status

- `python -m compileall -q crisp scripts train_launcher.py`: passed.
- `pytest -q`: **249 tests passed, plus 6 subtests**.
- Launcher routing: `crisp`, `grpo`, `reinforce_pp`, `opsd`, `vacs`,
  `cibo-v2`, and `sft` are recognized.
- All four ablation suites complete dry-run command generation.
- Scaling, EvalPlus, and LiveCodeBench wrappers complete dry-run generation.

No full 7B GPU training or external execution benchmark was run in the
modification environment. Run the smoke configs before beginning expensive
experiments.

## 1. Effective-batch reward normalization

The original loop normalized binary rewards independently inside each
microbatch. Consequently, `[0, 0]` and `[1, 1]` both generated zero
advantages, even when gradient accumulation was intended to create a larger
effective batch.

The new two-stage path in `crisp/training/common_loop.py`:

1. collects all rollout microbatches without retaining autograd graphs;
2. gathers their rewards across accumulation steps and distributed ranks;
3. computes one mean and standard deviation for the complete effective batch;
4. scores each stored microbatch with those shared statistics.

The new records are in `crisp/training/batch_types.py`. The corrected path is
used by CRISP, REINFORCE++, VACS, CIBO-v2, and the single-rollout policy-gradient
control. `tests/test_effective_batch_loop.py` verifies that `[0,0]` and `[1,1]`
normalize jointly to `[-1,-1]` and `[1,1]`.

## 2. Policy-gradient objective

The prior code created `old_logp = new_logp.detach()` in the same scoring
forward and then calculated a PPO ratio. Its numerical value was always one,
so clipping could never activate and the displayed scalar loss often cancelled
to zero.

CRISP, REINFORCE++, and GRPO now use their appropriate single-pass policy-
gradient losses directly. There is no claim of PPO clipping unless stored old
policy log-probabilities and multiple optimization epochs are implemented in a
future version.

## 3. CRISP corrections

- Added selectable `cosine`, `linear`, and `constant` lambda schedules.
- Added `training.use_pcgrad` for a real naive-sum ablation.
- Added `training.correctness_gate` for an always-on self-distillation
  ablation.
- The RL term now scores response tokens only and accepts effective-batch
  advantages from the shared loop.
- Empty responses are removed before response-token reductions.

## 4. REINFORCE++ and GRPO corrections

- REINFORCE++ now receives globally normalized effective-batch advantages.
- Fake PPO clipping was removed.
- GRPO `k>=2` uses within-prompt group-relative advantages.
- The historical `grpo_k1` manifest key is explicitly labelled a
  single-rollout global-policy-gradient budget control, not genuine GRPO.
- The launcher and ablation runner now forward the intended `k`; the former
  duplicate `k=1`/`k=8` command bug is removed.

## 5. VACS corrections

- Added configurable `vacs.use_soft_gate` and
  `vacs.use_asymmetric_projection` switches.
- The no-soft-gate ablation now uses a true unit gate rather than setting the
  sigmoid slope to zero, which would produce a constant gate of 0.5.
- Token credit uses the documented stop-gradient quantity
  `sg[log p_T - log p_S]` and the sign-aware clipped exponential weight.
- Adaptive statistics aggregate every accumulated microbatch instead of using
  only the final microbatch.
- Distributed ranks globally average those statistics before committing the
  next-step adaptive state.
- Documentation calls the implemented statistic gradient energy/squared norm;
  it is not misrepresented as an exact gradient-variance estimator.

## 6. CIBO-v2 corrections

- CIBO reads `cibo_v2.soft_gate_slope` and
  `cibo_v2.reward_threshold`, not the VACS section.
- `cibo_v2.ib_reduction` reaches the loss; `sequence_sum` and `token_mean` are
  now real, distinct ablations.
- Added `cibo_v2.use_soft_gate` for a true unit-gate ablation.
- The information-bottleneck term is forward KL
  `KL(detached teacher || live student)`.
- Replaced the gathered-token EMA NLL with full-distribution cross-entropy
  `H(p_EMA, p_live)`. The EMA target is detached and gradients flow only
  through the live student.
- The total gradient comes from one scalar objective; no PC-Grad is applied.
- The beta controller receives one globally averaged observation per optimizer
  update, so its dynamics do not depend on `grad_accum_steps`.
- Its bounded sign controller waits for two post-warmup observations before
  forming a delta.

## 7. Checkpoint and resume corrections

Common-loop checkpoints now contain:

- adapter/model state;
- optimizer and scheduler state;
- global optimizer step;
- resumable shuffled data-stream state;
- Python, Torch CPU, and CUDA RNG states;
- VACS/CIBO method state.

Interrupted or failed runs are written to `interrupted/`; only successfully
completed runs receive `final/` and the top-level `_SUCCESS` marker.

Distributed runs save rank-specific state files such as
`training_state_rank_1.pt`. Resuming with a different world size is rejected,
because it cannot reproduce the original data shards and RNG streams exactly.

## 8. Distributed execution

`crisp/utils/distributed.py` provides explicit reward gathering and gradient
all-reduction for the repository's manual-gradient methods. Quantized models
are pinned to each process's `LOCAL_RANK`; `device_map="auto"` is not allowed
to make every process occupy every GPU.

Launch with:

```bash
NUM_GPUS=2 METHOD=crisp CONFIG=config/paper_7b.yaml \
  bash scripts/launch_distributed.sh
```

Current CRISP/VACS semantics are **local gradient surgery followed by global
averaging of the final gradients**. This is operationally valid but is not
identical to globally averaging RL and SD task gradients before surgery. State
this choice in experimental reporting.

## 9. SFT to RL support

Added the `sft` launcher target and `config/sft_7b.yaml`.
`crisp/training/sft_train.py` performs completion-only supervised fine-tuning:
prompt tokens receive label `-100`, while assistant solution tokens contribute
to cross-entropy.

All RL/SD methods support:

```yaml
model:
  initial_adapter: runs/sft_7b/final
```

The existing adapter is loaded as trainable; a second fresh adapter is not
stacked on it.

## 10. Added experiment coverage

### HMMT February 2025

`hmmt2025` is registered in the evaluation loader and included in
`config/paper_7b.yaml`.

### HumanEval+

`scripts/merge_adapter.py` creates a merged checkpoint.
`scripts/eval_humaneval_plus.py` delegates execution to the separately
installed official EvalPlus package.

### LiveCodeBench

`scripts/eval_livecodebench.py` delegates to an installed official
LiveCodeBench environment and makes the release version explicit.

### Model scaling

`config/scaling/` contains matched Qwen2.5-Instruct configurations at 1.5B,
3B, 7B, and 14B. `scripts/run_scaling.py` runs a serial seed/method matrix.
These general Qwen2.5 checkpoints form one consistent scaling family; do not
mix their curve with Qwen2.5-Math checkpoints without labelling the family
change.

## 11. Evaluation and ablation reliability

- Bootstrap count and bootstrap seed now flow from YAML into evaluation.
- `item_correctness_std` distinguishes per-item Bernoulli variation from
  cross-seed uncertainty.
- The ablation runner validates `_SUCCESS`, metric-file existence, expected
  metric keys, and finite metric values after every run.
- The manifest's formerly inert VACS/CIBO ablations now alter actual code
  paths.
- `config/paper_7b.yaml` provides a common model/data/optimizer/rollout/eval
  basis for method comparisons.
- LoRA dropout is set to zero because rollout and manual scoring forwards use
  evaluation mode; a nonzero configured dropout would otherwise be misleading.

## 12. External dependencies and remaining caveats

- EvalPlus and LiveCodeBench are intentionally not vendored. Install and pin
  their official environments separately.
- Execution-based code benchmarks run generated code and should be isolated
  from the training runtime.
- The CIBO KL remains a variational proxy rather than exact mutual
  information.
- Fixed-beta and adaptive-beta CIBO results have different theoretical
  interpretations and remain separate experiment suites.
- Full-scale results still require multiple seeds, matched training-token
  budgets, and hardware/runtime reporting.

## 13. Kaggle storage/memory-footprint improvements

- Training subsets now use Hugging Face split slicing when `train_subset_size`
  is provided, so smoke tests do not first materialize the entire NuminaMath
  source dataset.
- Added deterministic `gsm8k_validation` loading for periodic validation.
- Added per-microbatch runtime logging to the shared effective-batch loop.
- Added GRPO runtime batch accounting and a strict `prompts × k == rewards`
  assertion.
- GRPO scoring can be chunked with `training.grpo_score_chunk_size`, preserving
  the full rollout group while reducing peak GPU memory.
- The Kaggle notebook uses the repository directly from `/kaggle/input`
  instead of copying it into `/kaggle/working`.
- Pip caching is disabled in the Kaggle notebook, and the Hugging Face dataset
  cache is separated from the model cache and cleaned after each run.
- Successful smoke-test output directories are removed automatically; training
  logs are capped to the most recent 5 MiB.

