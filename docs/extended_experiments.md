# Extended experiment support

## Matched paper configuration

`config/paper_7b.yaml` is the canonical shared configuration for all RL/SD
methods. It keeps model, data, optimizer, rollout budget, evaluation suite,
and effective batch settings fixed. Only method-specific sections should be
overridden.

## HMMT February 2025

Add `hmmt2025` to `data.eval_datasets`. The loader uses
`MathArena/hmmt_feb_2025` and normalizes the `problem` and `answer` fields.

## SFT -> GRPO / CRISP / VACS / CIBO

```bash
python train_launcher.py sft --config config/sft_7b.yaml
python train_launcher.py grpo --config config/paper_7b.yaml --k 8 \
  --override model.initial_adapter=runs/sft_7b/final
```

The same `model.initial_adapter` override works for CRISP, VACS, CIBO, OPSD,
and REINFORCE++. It loads the existing adapter as trainable instead of
stacking a fresh adapter.

## HumanEval+

Merge an adapter, install the official EvalPlus package, then run the wrapper:

```bash
python scripts/merge_adapter.py \
  --base-model Qwen/Qwen2.5-Math-7B-Instruct \
  --adapter runs/crisp/final \
  --output merged/crisp
pip install -U evalplus
python scripts/eval_humaneval_plus.py --model merged/crisp --backend hf
```

## LiveCodeBench

Use the official LiveCodeBench environment and a pinned release. After merging
the adapter:

```bash
python scripts/eval_livecodebench.py \
  --local-model-path merged/crisp \
  --model-style qwen \
  --release-version release_v6
```

## Model scaling

The `config/scaling/` directory contains matched Qwen2.5-Instruct configs at
1.5B, 3B, 7B, and 14B. Run the reduced matrix first:

```bash
python scripts/run_scaling.py --methods reinforce_pp,crisp,cibo-v2 --seeds 0,1,2
```

## Distributed training

The manual-gradient loop supports `torchrun`. Each rank collects a disjoint
data shard, reward statistics are gathered globally, and final gradients are
averaged explicitly. CRISP/VACS currently perform gradient surgery locally
and then average the final gradients; report this choice in the paper.

```bash
NUM_GPUS=2 METHOD=crisp CONFIG=config/paper_7b.yaml \
  bash scripts/launch_distributed.sh
```
