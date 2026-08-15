# Kaggle smoke workflow

The smoke runs in this codebase are bounded, observable, and
explicitly designed to verify plumbing — not learning. A successful
smoke run completes the full pipeline (model load → dataset load →
rollout → reward → step → checkpoint) in three training steps with
eight examples. Sparse or zero rewards are NOT a regression.

## End-to-end sequence

```bash
# 1. (Recommended) Pre-download the smoke model so the run never
#    touches the network. This is a single ~1 GB download for
#    Qwen/Qwen2.5-0.5B-Instruct at 4-bit precision.
python predownload.py Qwen/Qwen2.5-0.5B-Instruct

# 2. Dry-run to see the resolved command and preflight results
#    without launching training. This is the right command to run
#    first on a fresh Kaggle notebook.
python scripts/kaggle_smoke.py --method vacs --dry-run
python scripts/kaggle_smoke.py --method cibo-v2 --dry-run

# 3. Actually run. Both smoke runs should complete in minutes on a
#    Kaggle T4 or P100.
python scripts/kaggle_smoke.py --method vacs
python scripts/kaggle_smoke.py --method cibo-v2

# 4. (Optional) Combine the two steps: --predownload runs step 1
#    for you.
python scripts/kaggle_smoke.py --method cibo-v2 --predownload
```

## Output directory

The smoke runner writes to `runs/kaggle_smoke/<method>_smoke/`
by default. You can override with `--output-root`. The contents
of that directory are:

```text
runs/kaggle_smoke/<method>_smoke/
├── checkpoint.pt            # final adapter checkpoint
├── metrics.jsonl            # one JSON record per logged step
├── generations.jsonl        # one JSON record per rollout
└── smoke_run.log            # full stdout/stderr from training
```

## Success indicators

A successful smoke run produces:

- a `checkpoint.pt` file in the output directory,
- a `metrics.jsonl` containing at least one record,
- at least one record in `metrics.jsonl` with the method's
  required metric key (`vacs/loss_rl` for VACS,
  `cibo/loss_total` for CIBO v2),
- every value in `metrics.jsonl` is finite (no NaN / Inf),
- at least one non-empty response in `generations.jsonl`,
- a `smoke_run.log` with no traceback.

The runner's post-run verification prints a PASS / FAIL line for
each check. The final summary reports how many of the required
checks passed.

## Non-success indicators

A non-success indicator is **not** the reward being zero — that's
expected at 3 steps. A non-success indicator is:

- **Missing required metric key.** The method's training loop
  should emit it on every step. If it's missing, the training
  loop ran but the metric path is broken (a regression in
  `crisp/training/vacs_step.py` or `cibo_v2_step.py`).
- **Non-finite values** (`NaN`, `Inf`) in `metrics.jsonl`. The
  RL signal or the IB/anchor terms produced an unstable gradient.
  Investigate the post-optimizer-step hooks (`ctrl.step()` for
  CIBO v2; `vacs_state.commit()` for VACS) — most non-finite
  cases trace back to a beta or lambda that grew unboundedly.
- **No `checkpoint.pt`.** The save path ran but no checkpoint was
  emitted. This usually means `training.save_every = 0` AND the
  trainer's final-step save didn't fire.
- **HARD FAILURE in preflight**: the runner exits with code 2
  before launching. The preflight output names which check
  failed (`method is in SUPPORTED_METHODS`, `smoke config
  exists`, etc.).

## Predownload fallback

`predownload.py` is a thin wrapper around the Hugging Face Hub
client. It accepts a model id (e.g. `Qwen/Qwen2.5-0.5B-Instruct`)
and downloads both the model weights and the tokenizer. On a
Kaggle notebook, this download should take well under a minute.

If `predownload.py` fails (network error, disk full, hub rate
limit), the smoke runner will fail at the model-load step. The
fix is:

1. Verify the model id exists on the Hub (typos are the most
   common cause).
2. Check disk space — `predownload.py` and the smoke run together
   need ~3 GB free for the 0.5B model.
3. Re-run `predownload.py` standalone; if it fails repeatedly,
   check the Kaggle notebook's network egress policy.

The smoke configs set `model.local_files_only: false` so the
trainer can fall back to the network if predownload was skipped.
For reproducible Kaggle runs, always predownload first.

## train_subset_size

Both smoke configs set `data.train_subset_size: 8` so the loader
returns only the first 8 examples of GSM8K. This is what makes the
smoke run bounded: 3 optimizer steps × 1 micro-batch per step × 8
examples = 24 rollouts total, which completes in minutes on a Kaggle
GPU. `load_train_dataset(name, split, train_subset_size=N)` applies
the truncation BEFORE the trainer ever sees a batch, so the
iteration count is correctly bounded — the loop terminates after the
subset is exhausted, not after `len(dataset)` is exhausted.

## Artifact inspection

After a smoke run, the artifacts can be inspected without
re-running anything:

```bash
# Quick view of every metrics record
cat runs/kaggle_smoke/vacs_smoke/metrics.jsonl

# Filter to the vacs-required key
grep -o '"vacs/loss_rl":[^,}]*' runs/kaggle_smoke/vacs_smoke/metrics.jsonl

# Look at one generation
head -1 runs/kaggle_smoke/vacs_smoke/generations.jsonl | python -m json.tool

# Confirm checkpoint non-empty
ls -lh runs/kaggle_smoke/vacs_smoke/checkpoint.pt
```

The `metrics.jsonl` records contain every metric the trainer
logged. The `generations.jsonl` records contain the prompt, the
model's response text, and the response token ids.

## Cleanup

```bash
# Remove a single smoke run
rm -rf runs/kaggle_smoke/vacs_smoke

# Remove all smoke runs
rm -rf runs/kaggle_smoke
```

The smoke runs are not on the model's checkpoint path: deleting
them does not affect any other run, and predownload.py's cache
directory is in `~/.cache/huggingface/` (untouched by the smoke
runner).