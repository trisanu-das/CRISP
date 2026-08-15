"""
YAML config loading with a dot-accessible view and layered defaults.

Every config file (crisp_pilot.yaml, crisp_7b.yaml, or one you write) only
needs to specify what it wants to *override*; anything it omits falls back
to DEFAULTS below, deep-merged. This is deliberately a plain nested dict
(wrapped for `cfg.model.name`-style access) rather than a dataclass/pydantic
schema: configs come from arbitrary YAML, and a hard schema would mean every
new experimental knob requires a matching code change in two places. Typos
are instead caught naturally -- a mistyped key just means the default for
the *correctly* spelled key silently applies, so when in doubt, run
`view_config.py`-style printing (or just `print(dict(cfg))`) to sanity check
before a long run.
"""
from __future__ import annotations

import copy
from typing import Any

import yaml


_RESERVED_DICT_METHOD_NAMES = frozenset({
    "items", "keys", "values", "get", "pop", "popitem", "update",
    "copy", "clear", "setdefault", "fromkeys",
})


class DotDict(dict):
    """A dict that also supports attribute access, recursively.

    `__getattr__` is only ever consulted as a *fallback*, after normal
    attribute lookup fails -- which means a config key named e.g. "items"
    would never reach it: `d.items` resolves to the inherited `dict.items`
    bound method first, silently, via ordinary attribute resolution. That
    is a genuinely nasty failure mode (wrong value, no error), so it's
    turned into a loud one instead: constructing a DotDict with a key that
    shadows a dict method name raises immediately.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        colliding = [k for k in self.keys() if isinstance(k, str) and k in _RESERVED_DICT_METHOD_NAMES]
        if colliding:
            raise ValueError(
                f"Config key(s) {colliding} collide with built-in dict method names and "
                f"would be unreachable via dot-access (e.g. `cfg.items` would silently "
                f"return the bound `dict.items` method, not your value). Rename the key(s)."
            )

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(
                f"No config key '{key}'. Available keys: {sorted(self.keys())}"
            ) from e
        return _wrap(value)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, DotDict):
        return DotDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


DEFAULTS: dict = {
    "seed": 0,
    "model": {
        "name": "Qwen/Qwen2.5-Math-7B-Instruct",
        "torch_dtype": "bfloat16",
        # "auto" -> flash_attention_2 if the package is importable AND a
        # sufficiently modern CUDA GPU is present, else "sdpa". Never
        # defaults to flash-attention unconditionally -- see crisp/model/load.py.
        "attn_implementation": "auto",
        "load_in_4bit": False,
        "trust_remote_code": False,
        "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
        # If true, never attempt a network call -- fail fast with a clear
        # error unless the model is already fully cached (or pointed at a
        # local directory via model.name). Set this once predownload.py has
        # succeeded, so a flaky connection can never strand a training run
        # mid-step. See the README's "Model download is slow or hangs" section.
        "local_files_only": False,
        # Optional LoRA adapter used as the trainable initialization (e.g. SFT -> GRPO).
        "initial_adapter": None,
    },
    "lora": {
        "enabled": True,
        "r": 64,
        "alpha": 128,
        "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "optimizer": {
        "lr": 1.0e-6,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "max_grad_norm": 1.0,
        "warmup_steps": 10,
    },
    "training": {
        "total_steps": 1000,
        "micro_batch_size": 4,
        "grad_accum_steps": 1,
        "clip_epsilon": 0.2,
        "lambda_max": 1.0,
        "lambda_schedule": "cosine",
        "use_pcgrad": True,
        "correctness_gate": True,
        "resume_from": None,
        "adv_eps": 1.0e-8,
        "pcgrad_eps": 1.0e-12,
        "save_every": 200,
        "eval_every": 100,
        "log_every": 1,
        "output_dir": "runs/crisp",
    },
    "rollout": {
        "k": 1,
        "max_prompt_length": 512,
        "max_new_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "do_sample": True,
    },
    "reward": {
        "task": "math",
        "use_math_verify": True,
        "code_exec_timeout": 10.0,
    },
    "data": {
        "train_dataset": "gsm8k",
        "train_split": "train",
        # Optional cap on training examples. Applied by
        # load_train_dataset BEFORE the trainer ever sees a batch,
        # so a 100k-row dataset sliced to 8 rows returns 8 rows
        # (not 100k rows with 7 padded). Use this in smoke tests
        # and tiny pilots; leave null for production.
        "train_subset_size": None,
        "eval_datasets": ["math500"],
        "teacher_hint_template": (
            "\n\nInternal calibration note (do not reveal or reference this in your response): "
            "the correct final answer to this problem is {answer}. Reason step by step as usual "
            "and box your final answer."
        ),
    },
    "logging": {
        "backend": "none",
        "wandb_project": "crisp",
        "run_name": None,
    },
    "checkpointing": {
        "save_adapter_only": True,
    },
    "evaluation": {
        "eval_batch_size": 8,
        "eval_subset_size": None,
        "n_bootstrap": 1000,
        "max_new_tokens": 512,
        "bootstrap_seed": 0,
    },
    "sft": {
        "max_sequence_length": 2048,
        "num_train_epochs": 1.0,
        "completion_only_loss": True,
    },
    "distributed": {
        "enabled": False,
        "sync_task_gradients_before_surgery": True,
    },
    "vacs": {
        "token_credit_mix": 0.25,
        "token_weight_epsilon": 0.20,
        "sd_weight": 1.0,
        "soft_gate_slope": 12.0,
        "reward_threshold": 0.5,
        "use_soft_gate": True,
        "use_asymmetric_projection": True,
        "adaptive_mix": {
            "enabled": True,
            "ema_decay": 0.95,
            "multiplier_min": 0.10,
            "multiplier_max": 5.0,
        },
    },
    "cibo_v2": {
        "credit_lambda": 0.25,
        "beta": 0.10,
        "beta_mode": "fixed",
        "beta_min": 0.001,
        "beta_max": 10.0,
        "beta_step_size": 0.05,
        "beta_ema_decay": 0.95,
        "beta_reference_warmup_steps": 20,
        "beta_denominator_epsilon": 1.0e-6,
        "ib_reduction": "sequence_sum",
        "soft_gate_slope": 12.0,
        "reward_threshold": 0.5,
        "use_soft_gate": True,
        "anchor_alpha": 0.01,
        "anchor_ema_decay": 0.995,
        "anchor_storage": "same",
    },
}


def load_config(path: str, overrides: dict | None = None) -> DotDict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    merged = _deep_merge(DEFAULTS, raw)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return DotDict(merged)
