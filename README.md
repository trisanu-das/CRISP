# CRISP

**Critic-free Reward-Integrated Self-distillation Policy Optimization**

CRISP is a critic-free RL post-training method for LLM reasoning that unifies an on-policy RLVR objective and an on-policy self-distillation objective into a single per-step loss, with PC-Grad gradient deconfliction between the two terms.

## Motivation

Self-distillation speeds up reasoning-model post-training but can degrade out-of-distribution reasoning by suppressing the epistemic hedging behavior ("wait," "let me check") a model needs to recover from wrong turns ([Kim et al., *Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?*, arXiv:2603.24472](https://arxiv.org/abs/2603.24472)). Follow-up work traces this to an irreducible information-theoretic bias: any objective that distribution-matches against a privileged-context teacher inherits a gap the student can't close, because it can't condition on that context at inference time ([Yang et al., *Self-Distilled RLVR*, arXiv:2604.03128](https://arxiv.org/abs/2604.03128)).

CRISP asks a narrower, falsifiable question. Can a single unified critic-free objective, combining RL and self-distillation and reconciled by gradient-conflict resolution rather than staging or fixed interpolation, match RL-only accuracy at a fraction of the rollout budget, without reintroducing the degradation self-distillation is known to cause?

## 1. Problem formalization

### 1.1 Setup

Let `π_θ` denote a language-model policy over completions `y` given a prompt `x`, parameterized by `θ`. Let `D = {(x_i, y_i*)}` be a dataset of prompts paired with ground-truth answers. Let `r(y, y*)` be a verifiable reward function (e.g. exact-match, unit-test pass rate).

The post-training goal is:

```
θ* = argmax_θ  E_{y ~ π_θ(·|x)} [ r(y, y*) ]
```

subject to three constraints that matter in practice:

- **(i) Token budget.** Rollout/generation cost per training step is minimal.
- **(ii) No auxiliary critic.** No separately trained value network.
- **(iii) Bounded gradient variance.** The per-step gradient estimator has controlled variance, without requiring excessive rollouts or batch sizes to stabilize.

### 1.2 The trilemma

No existing critic-free method satisfies all three simultaneously:

| Method | (i) Token budget | (ii) No critic | (iii) Bounded variance |
|---|---|---|---|
| GRPO / REINFORCE++ | Fails: needs `k >= 4` rollouts/prompt | Satisfied | Satisfied only at `k >= 4`; at `k=1` the advantage estimate is biased and training is unstable |
| OPSD / CoT-distillation | Satisfied: dense token-level supervision replaces `k` rollouts | Satisfied | Satisfied in isolation, but decoupled from reward: the student gets no reward signal during distillation, so it is typically run as a separate stage |

**Working hypothesis on root cause:** the RL-optimal parameters `θ_RL` and the distillation-optimal parameters `θ_SD` are not found in a shared optimization. No existing unified objective co-optimizes both signals from the same on-policy trajectory, on a single model, without a critic.

### 1.3 Problem statement

Find a unified per-step objective:

```
L(θ) = f( L_RL(θ), L_SD(θ) )
```

such that one forward pass per prompt yields both a reward-based advantage signal and dense token-level supervision, with mathematically bounded gradient conflict between the two terms.

This is non-trivial because `L_RL` and `L_SD` encode structurally opposed inductive biases. `L_RL` rewards sparse, outcome-level feedback and tolerates exploration. `L_SD` enforces dense, token-level imitation of a conditioned target and penalizes deviation from it. Naive gradient summation does not resolve this; it just averages two conflicting update directions rather than reconciling them.

## 2. Method

### 2.1 Unified objective

```
L(θ) = L_RL(θ) + λ(t) · L_SD(θ)
```

**RL term** (REINFORCE++-style, globally normalized advantage, clipped ratio ε = 0.2):

```
L_RL(θ) = -Â_i · log π_θ(y_i | x_i)
Â_i = (r_i - μ_batch) / σ_batch
```

**Self-distillation term** (on-policy; teacher = ground-truth-conditioned, student = unconditioned):

```
L_SD(θ) = KL[ π_θ(· | x, y*) || π_θ(· | x) ]
```

Direction matters here: the conditioned teacher `π_θ(·|x,y*)` supervises the unconditioned student `π_θ(·|x)`, i.e. `KL(teacher || student)`, which penalizes the student for missing probability mass where the teacher is confident. The reverse direction would let the student ignore teacher modes entirely and is deliberately not used.

**Coupling schedule** (cosine-annealed):

```
λ(t) = λ_max · cos(π t / (2T))
```

**Correctness gate** (mitigates FM4, below):

```
L_SD_i = 0   if r_i = 1
```

Teacher and student share weights `θ`; only the context differs (`[x, y*]` vs. `[x]`). Both are computed in one batched forward pass, not two model copies, which halves the memory overhead relative to a naive two-model implementation.

### 2.2 Gradient deconfliction (PC-Grad)

Let `g_RL = ∇_θ L_RL` and `g_SD = ∇_θ L_SD`. If `g_RL · g_SD < 0` (conflicting directions), project each onto the other's normal plane before combining:

```
if g_RL · g_SD < 0:
    g_RL ← g_RL - (g_RL · g_SD / ‖g_SD‖²) · g_SD
    g_SD ← g_SD - (g_SD · g_RL / ‖g_RL‖²) · g_RL

θ ← θ - α · (g_RL + λ(t) · g_SD)
```

**Scope of what this does and doesn't fix.** PC-Grad only intervenes when the two gradients conflict. When they're aligned, plausibly common early in training when both objectives push toward locally sound reasoning, both gradients pass through unmodified, including whatever privileged-information bias `L_SD` inherits from conditioning on `y*`. So PC-Grad resolves *directional conflict*; by construction it does not resolve the *leakage bias* that motivated this project (Kim et al., 2603.24472; Yang et al., 2604.03128). Whether the correctness gate is sufficient to control this on its own is the open question that Section 4 (FM1) and the experiments in Section 5 are designed to answer, not an assumption the method relies on.

### 2.3 Per-step algorithm

For a batch of `N` prompts `{x_1, ..., x_N}`:

```
1. Sample on-policy rollout:      y_i ~ π_θ(· | x_i)                    [student context]
2. Compute verifiable reward:     r_i = r(y_i, y*_i)
3. Forward pass (teacher):        logits_T = π_θ(· | x_i, y*_i)         [teacher context]
4. Compute L_SD per prompt:       KL[ logits_T ‖ π_θ(y_i | x_i) ] on generated tokens
                                   gate: L_SD_i = 0 if r_i = 1           [FM4 mitigation]
5. Compute global advantage:      Â_i = (r_i − μ_batch) / σ_batch
6. Compute L_RL:                  −Â_i · log π_θ(y_i | x_i), clipped ratio ε = 0.2
7. Compute gradients:             g_RL = ∇L_RL,  g_SD = ∇L_SD
8. PC-Grad deconfliction:         if g_RL · g_SD < 0:
                                     g_RL ← g_RL − (g_RL·g_SD / ‖g_SD‖²) · g_SD
                                     g_SD ← g_SD − (g_SD·g_RL / ‖g_RL‖²) · g_RL
9. Combined update:               θ ← θ − α · (g_RL + λ(t) · g_SD)
10. Log:                          r_i, L_SD_i, Â_i, σ²_A, cos(g_RL, g_SD), H(π)
```

Teacher and student forward passes are computed by concatenating `[x, y*]` and `[x]` in the same batched forward pass and splitting the resulting logits, never by instantiating two model copies.

## 3. Hypotheses

Stated as falsifiable predictions, not results. No experiments have been run yet (see Status).

**H1 (main).** The unified objective with PC-Grad deconfliction achieves Pass@1 within 2 pp of GRPO (k=8) on AIME 2024/2025, using no more than 25% of GRPO (k=8)'s rollout token budget, with reward variance no higher than REINFORCE++'s across training steps.

**H1a.** CRISP at k=1 matches GRPO (k=8) within 2 pp on math benchmarks, because `L_SD` supplies a correction signal dense enough to offset the variance of a single-sample advantage estimate. Effective compute is approximately 2 forward passes per prompt (student + teacher) versus 8 for GRPO.

**H1b.** PC-Grad deconfliction reduces the frequency of conflicting updates (`cos(g_RL, g_SD) < 0`) relative to naive summation, reduces gradient-norm variance by at least 30%, and prevents *advantage collapse*: the failure mode where `λ` dominates and per-step advantage variance `σ²_A → 0`.

**H2 (exploratory, not required for H1; kept separate so a null result here doesn't undermine the core claim).** CRISP-trained "slice" students, each trained on a disjoint data partition with the unified objective, produce more diverse and complementary weight vectors than SFT-trained students on the same partitions. A CNN aggregator trained to synthesize these slices (backprop through the CNN only, forward pass through the synthesized model) outperforms a single large model SFT-trained on the union of all partitions. This is a distinct, more speculative claim from H1 and is treated as a stretch goal, not a dependency.

## 4. Known failure modes

**FM1: Distillation reward hacking.** The model minimizes `L_SD` by matching the teacher's token distribution regardless of whether `r(y,y*)` is satisfied, producing fluent but incorrect reasoning chains (`L_SD` decreases while `r` stagnates). *Detection:* track `r` and `L_SD` independently; divergence sustained for at least 300 steps is a hacking signal. No correction mechanism is implemented yet; this is currently a monitor, not a fix.

**FM2: Aggregator instability (H2 only).** The CNN aggregator may fail to find stable weight combinations when student slices are trained on non-IID partitions, producing catastrophic interference between learned feature representations. *Mitigation:* deterministic partitioning by concept category (arithmetic, algebra, geometry), cosine-similarity diversity regularization in the aggregation loss, and monitoring gradient interference between aggregated student weights.

**FM3: Self-distillation signal degradation at small scale.** Predicted to appear at 1.7B parameters (used only as a negative-ablation scale, not a primary model). *Detection:* track `L_SD` independently at each scale in the model-scale ablation (Section 5); a plateau indicates the self-distillation signal has stopped being informative.

**FM4: Correctness-gate gaming.** The gate (Section 2.1) applies `L_SD` only on incorrect rollouts, on the intuition that correct rollouts need the correction signal less urgently. Edge case: the model could learn to produce confident-but-wrong outputs specifically to trigger the distillation gate, compounding rather than fixing the hacking in FM1. *Detection:* compare rollout entropy on gate-triggering versus non-triggering examples as a secondary monitor.

## 5. Experimental design

### 5.1 Models

**Primary:** Qwen2.5-Math-7B-Instruct, chosen for direct comparability with published OPSD (Qwen3 family) and GRPO/DeepSeek-R1 baselines without re-running them from scratch.

**Scale ablation:** Qwen3-4B, Qwen3-8B, Qwen3-14B. Tests H1a's token-efficiency claim across scale and FM3's predicted degradation point. Qwen3-1.7B is used only as a negative-ablation scale (FM3 predicts failure), not as a primary model.

### 5.2 Compute

| | |
|---|---|
| VRAM (bf16 + LoRA r=64) | ~32-40 GB |
| Minimum hardware | 2x A40 48GB |
| Preferred hardware | 2x A100 80GB |
| Training steps | 1,000-2,000 |
| Est. wall-clock (2x A100) | ~6-10 hours |
| Rollout cost vs. GRPO (k=8) | ~25% (2/8 forward passes per prompt) |

LoRA config: rank 64, alpha 128, target modules `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`. Global batch size 512-1024 is required for REINFORCE++'s global advantage normalization to be statistically sound; below 256 the estimate degrades toward biased GRPO(k=1) without the grouping benefit.

### 5.3 Baselines: what each isolates

| Baseline | Isolates |
|---|---|
| **CRISP** (proposed) | Full claim: joint `L_RL + λL_SD` + PC-Grad |
| GRPO (k=8) | Accuracy ceiling: does CRISP match it? |
| **GRPO (k=1)** | Same rollout budget as CRISP, no distillation. **The critical comparison.** |
| REINFORCE++ | Does joint training add anything beyond global advantage normalization alone? |
| OPSD (k=1) | Is the RL branch necessary, or does self-distillation alone suffice? |
| SFT to GRPO | Is joint training better than the standard sequential pipeline? |

**Falsification criterion:** if CRISP does not outperform GRPO(k=1) at matched rollout budget, `L_SD` is not contributing and H1 is falsified cleanly. There is no need to beat GRPO(k=8) on raw accuracy for a publishable negative result.

### 5.4 Benchmarks and metrics

- **Math:** AIME 2024, AIME 2025, HMMT 2025, MATH-500
- **Code:** HumanEval+, LiveCodeBench
- **Primary metric:** Pass@1, all baselines, all benchmarks
- **Efficiency metric:** tokens-per-correct-solution, defined as total rollout tokens during training divided by correct solutions at eval
- **Stability metrics:** reward curve, `σ²_A` per step, `cos(g_RL, g_SD)`, policy entropy `H(π)`
- **Wall-clock training time**, measured on identical hardware across baselines (FLOP estimates alone are insufficient)

### 5.5 Ablations

- **λ sweep:** `{0, 0.1, 0.5, 1.0, 2.0, ∞}`. `λ=0` is pure REINFORCE++, `λ=∞` is pure OPSD.
- **PC-Grad vs. naive summation.** Identical hyperparameters, toggling only deconfliction. Expected: naive summation shows higher reward variance and lower final accuracy (tests FM1's contribution).
- **Correctness-gated vs. always-on `L_SD`.** Tests FM4; hacking, if present, should concentrate in the always-on condition.
- **λ schedule:** constant vs. cosine-annealed vs. linear warmdown. Tests advantage collapse; constant high `λ` should collapse, annealing should not.
- **Model scale:** 4B / 7B / 14B, with `L_SD` reported independently at each scale (FM3).

### 5.6 Statistical requirements

Minimum 3 random seeds per reported number, mean plus standard deviation. AIME benchmarks (~30 problems each) additionally require bootstrapped 95% confidence intervals given their small size and high variance.

## Repository layout

```
CRISP/
├── train_launcher.py       # CLI entry point (repo root, not under scripts/)
├── sweep.sh                 # ablation sweeps (repo root)
├── requirements.txt
├── config/
│   ├── crisp_7b.yaml         # full-scale target config (2x A100 80GB)
│   └── crisp_kaggle.yaml     # Kaggle-scale pilot config (single seed, small model)
├── data/                    # dataset building + reward computation
├── eval/                    # evaluation runner + metrics
├── model/                   # model/tokenizer loading, LoRA setup, quantization
├── training/                 # CRISP training loop and step implementation
└── baselines/
    ├── grpo/                 # training_step.py (loss) + training_loop.py (loop)
    ├── opsd/                  # same split
    └── reinforce_pp/           # same split
```

## Installation

```bash
pip install -r requirements.txt
```

Also install `transformers`, `accelerate`, `peft`, `bitsandbytes` (for 4-bit/QLoRA loading). Key deps: `torch`, `datasets`, `pyyaml`, `math_verify` (reward verification); `wandb` is optional. The training loop is plain `transformers` + `peft`, no Ray/vLLM/veRL dependency despite earlier drafts of this README implying otherwise.

## Training

```bash
python train_launcher.py crisp        --config config/crisp_7b.yaml
python train_launcher.py grpo         --config config/crisp_7b.yaml --k 8
python train_launcher.py reinforce_pp --config config/crisp_7b.yaml
python train_launcher.py opsd         --config config/crisp_7b.yaml
```

For a Kaggle pilot (single T4x2 or P100 notebook), swap the config:

```bash
python train_launcher.py crisp        --config config/crisp_kaggle.yaml
python train_launcher.py grpo         --config config/crisp_kaggle.yaml --k 1
python train_launcher.py reinforce_pp --config config/crisp_kaggle.yaml
python train_launcher.py opsd         --config config/crisp_kaggle.yaml
```

**Smoke test:** `total_steps: 50-100`, `micro_batch_size: 1`, `max_new_tokens: 64-128`, `eval_every: 50`.
**Limited VRAM:** LoRA + 4-bit quantization (`model.load_in_4bit: true`); reduce `micro_batch_size`, `max_new_tokens`, `max_prompt_length`, `max_answer_length`. `config/crisp_kaggle.yaml` already applies all of these.

## Evaluation

```bash
python eval/run_eval.py
```

Reports `pass@1_mean`, `pass@1_std`, token usage, tokens-per-correct, and bootstrap confidence intervals across AIME-style, MATH-500, and HumanEval suites. Generation is now batched (`evaluation.eval_batch_size`, default 8) rather than one example at a time, and `evaluation.eval_subset_size` caps how many problems per benchmark get evaluated, useful for keeping a Kaggle-session eval pass from taking hours on the full MATH-500 test split.

## Status

Hypothesis-driven research codebase, pre-registration stage. The training loop, baselines, and eval pipeline are implemented and runnable end to end, but no benchmark results exist yet. Section 3's numbers are falsifiable predictions, not findings. The immediate next milestone is the CRISP-vs-GRPO(k=1) comparison (Section 5.3), since it's the cleanest test of whether `L_SD` contributes anything at all.

A round of bug-fixing (import paths not matching actual filenames, an invalid `torch.autograd.grad` kwarg, AIME eval sources pointing at a placeholder with no loader, unbounded sequence lengths, flash-attention defaulting on for hardware/installs that can't use it) has since been applied on top of the version these predictions were written against; none of it changes the hypotheses in Section 3, all of it was required to actually run them.

(P.S. I'm very low on compute)

## Related work

- Shao et al., [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Yu et al., [PCGrad: Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
- Hübotter et al., [Reinforcement Learning via Self-Distillation (SDPO)](https://arxiv.org/abs/2601.20802)
- Zhao et al., [Self-Distilled Reasoner: On-Policy Self-Distillation (OPSD)](https://arxiv.org/abs/2601.18734)
- [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?](https://arxiv.org/abs/2603.24472)
- Yang et al., [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128)
- Kim et al., [Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR](https://arxiv.org/abs/2605.10781)
- Cheng et al., [Invariant Gradient Alignment for Robust Reasoning Distillation](https://arxiv.org/abs/2606.05025)
- Li et al., [Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing](https://arxiv.org/abs/2604.02288)

