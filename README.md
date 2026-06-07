# CRISP

Critic-free Reward-Integrated Self-distillation Policy Optimization

CRISP is a training framework for finetuning language models with:
- on-policy rollouts
- verifiable reward signals
- self-distillation
- gradient conflict deconfliction

It is designed to compare against:
- GRPO
- REINFORCE++
- OPSD

and to study sample efficiency, convergence stability, and token efficiency for math-reasoning and related tasks.

## What this repository contains

- `training/` — CRISP training loop and step implementation
- `baselines/` — GRPO, REINFORCE++, and OPSD baselines
- `data/` — dataset building and reward computation
- `model/` — model/tokenizer loading and LoRA setup
- `eval/` — evaluation runner and metrics
- `scripts/` — sweep and CLI launcher scripts
- `config/` — main experiment configuration

## Core idea

CRISP combines:
1. an RL objective from on-policy rollouts,
2. a self-distillation objective from a privileged teacher context,
3. PC-Grad style gradient conflict resolution,
4. correctness gating to avoid distilling successful rollouts unnecessarily.

The goal is to improve:
- convergence speed
- token efficiency
- stability
- final benchmark performance

## Supported methods

### CRISP
Joint RL + self-distillation optimization.

### GRPO
Critic-free group-relative policy optimization baseline.

### REINFORCE++
Critic-free policy gradient baseline with global reward normalization.

### OPSD
On-policy self-distillation baseline without the RL branch.

## Dependencies

Typical dependencies:
- `torch`
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `pyyaml`
- `wandb` (optional)
- `vllm` (optional)
- `math_verify` or equivalent verifier utilities

Install with:

```bash
pip install -r requirements.txt
```

If you are using a fresh environment, also install the Hugging Face stack you need for your setup.

## Configuration

The main config lives in:

```bash
config/crisp_7b.yaml
```

It defines:
- model name
- LoRA settings
- optimizer settings
- training schedule
- rollout settings
- reward settings
- train/eval datasets
- logging options
- checkpointing options

## Training

Use the launcher from the repo root.

### CRISP

```bash
python scripts/train_launcher.py crisp --config config/crisp_7b.yaml
```

### GRPO

```bash
python scripts/train_launcher.py grpo --config config/crisp_7b.yaml --k 1
python scripts/train_launcher.py grpo --config config/crisp_7b.yaml --k 8
```

### REINFORCE++

```bash
python scripts/train_launcher.py reinforce_pp --config config/crisp_7b.yaml
```

### OPSD

```bash
python scripts/train_launcher.py opsd --config config/crisp_7b.yaml
```

## Evaluation

Standalone evaluation is handled through:

```bash
eval/run_eval.py
```

It supports:
- AIME-style evaluation
- MATH-500 evaluation
- HumanEval evaluation

The evaluation reports:
- `pass@1_mean`
- `pass@1_std`
- token usage
- tokens-per-correct
- bootstrap confidence intervals

## Data

The dataset pipeline normalizes prompts and answers into a common schema:

```json
{
  "prompt": "...",
  "answer": "..."
}
```

Supported sources include:
- NuminaMath-CoT
- GSM8K
- AIME-style benchmark datasets
- MATH-500-style evaluation datasets
- HumanEval

## Logging

WandB logging is optional.

To disable it:

```yaml
logging:
  backend: none
```

or set:

```bash
export WANDB_DISABLED=true
```

## Hardware notes

This project was designed with large GPU setups in mind, but it can be adapted for smaller hardware using:
- LoRA
- 4-bit quantization
- shorter rollout lengths
- smaller batch sizes
- fewer training steps

For limited VRAM, reduce:
- `micro_batch_size`
- `max_new_tokens`
- `max_prompt_length`
- `max_answer_length`
- LoRA rank

## Suggested first run

For a smoke test, keep the run short:

- `total_steps`: 50 to 100
- `micro_batch_size`: 1
- `max_new_tokens`: 64 to 128
- `eval_every`: 50

This verifies:
- model loading
- dataset loading
- rollout generation
- reward calculation
- optimizer updates
- checkpoint saving

## File layout

```text
CRISP-main/
├── config/
│   └── crisp_7b.yaml
├── data/
│   ├── build_dataset.py
│   └── reward.py
├── eval/
│   ├── metrics.py
│   └── run_eval.py
├── model/
│   └── load.py
├── training/
│   ├── crisp_step.py
│   └── train.py
├── baselines/
│   ├── grpo.py
│   ├── reinforce_pp.py
│   ├── opsd.py
│   ├── train_grpo.py
│   ├── train_reinforce_pp.py
│   └── train_opsd.py
└── scripts/
    ├── sweep.sh
    └── train_launcher.py
```

## Research goals

This repository is aimed at studying:
- joint RL and self-distillation objectives
- gradient conflict mitigation
- token-efficient reasoning finetuning
- stable critic-free post-training methods

## Notes

This is an experimental research codebase. Expect to tune:
- sequence lengths
- rollout lengths
- LoRA rank
- batch sizes
- evaluation cadence
- reward parsing behavior

before doing large runs.

