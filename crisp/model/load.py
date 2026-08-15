"""
Model + tokenizer loading: hardware-aware attention implementation choice,
optional 4-bit quantization, and LoRA attachment.

Known failure mode this specifically guards against (see the reference
project's own changelog): flash-attention defaulting on for hardware/installs
that can't use it. We only ever request `flash_attention_2` if the package is
importable AND a CUDA GPU with a compatible compute capability is present;
otherwise we fall back to `sdpa` (ships with any recent PyTorch, no extra
install needed), and fall back again to `eager` if construction with `sdpa`
itself fails for a given model/version combination.

A second failure mode worth naming explicitly: HuggingFace Hub downloads can
hang or stall indefinitely on some networks (a known issue with the Hub's
Xet-based transfer backend under certain proxies/firewalls -- see the
"Model download is slow or hangs" section of the top-level README). Nothing
here can fix that from inside Python once a download is already stuck, but
`model.local_files_only` lets you opt out of ever attempting a network call
at all once you've predownloaded (see predownload.py), and failures are
reported with an actionable message instead of a bare stack trace.

A third: `from_pretrained`'s dtype kwarg was renamed from `torch_dtype` to
`dtype` partway through the transformers 4.x series. `_from_pretrained_with_dtype`
tries the new name first and falls back to the old one, so this doesn't pin
to one exact transformers version on either side of that rename.
"""
from __future__ import annotations

import logging

import torch
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from crisp.utils.device import resolve_device

logger = logging.getLogger(__name__)

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_DOWNLOAD_HELP = (
    "Failed to load '{model_name}' ({reason}). If this looked like it was "
    "hanging rather than erroring, see the 'Model download is slow or hangs' "
    "section of the README -- in short:\n"
    "  1. Try `python predownload.py {model_name}` on its own first, so you "
    "can see real progress/errors decoupled from training.\n"
    "  2. If that hangs too, try `python predownload.py {model_name} "
    "--disable-xet` (a known class of hang in HF Hub's newer transfer "
    "backend on some networks).\n"
    "  3. Once predownload.py succeeds, set `model.local_files_only: true` "
    "in your config so training never touches the network again."
)


def _pick_attn_implementation(requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere or newer; flash-attn 2 needs this in practice
            return "flash_attention_2"
        return "sdpa"
    except ImportError:
        return "sdpa"


def load_tokenizer(model_name: str, trust_remote_code: bool = False, local_files_only: bool = False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as e:
        raise RuntimeError(_DOWNLOAD_HELP.format(model_name=model_name, reason=e)) from e
    if tokenizer.pad_token is None:
        # Extremely common gotcha for causal-LM-only checkpoints (Qwen, Llama,
        # ...): there is no dedicated pad token. Reusing eos as pad is
        # standard practice; it means eos-as-pad tokens must never silently
        # enter a loss region, which is handled structurally in
        # training/logprobs.py via explicit response spans rather than by
        # relying on pad-token filtering alone.
        tokenizer.pad_token = tokenizer.eos_token
    # NOTE: we deliberately never set a persistent tokenizer.padding_side
    # here. Generation needs left-padding; scoring/training forward passes
    # need right-padding (see training/rollout.py and training/logprobs.py
    # for why). Relying on a single mutable global for both, and remembering
    # to flip it at the right times, is exactly the kind of shared-state leak
    # that produces "sometimes gibberish" bugs -- so padding is instead built
    # by hand per use case, and this tokenizer's `padding_side` is never read.
    return tokenizer


def _from_pretrained_with_dtype(model_name: str, dtype: torch.dtype, **kwargs):
    """`from_pretrained`'s dtype kwarg was renamed from `torch_dtype` to
    `dtype` partway through the transformers 4.x series (the old name still
    works but now prints a deprecation warning: "torch_dtype is deprecated!
    Use dtype instead!"). Try the new name first; fall back to the old one
    only if it's flatly not recognized (a transformers version old enough to
    predate the rename), so this works on either side of it without pinning
    to one exact version.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError as e:
        if "dtype" not in str(e):
            raise
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def load_model_and_tokenizer(model_cfg):
    """Returns (model, tokenizer). Does not attach LoRA -- see `attach_lora`."""
    local_files_only = bool(model_cfg.get("local_files_only", False))
    tokenizer = load_tokenizer(
        model_cfg.name, trust_remote_code=model_cfg.get("trust_remote_code", False),
        local_files_only=local_files_only,
    )

    resolved_dtype = _DTYPE_MAP[model_cfg.get("torch_dtype", "bfloat16")]
    attn_impl = _pick_attn_implementation(model_cfg.get("attn_implementation", "auto"))

    quant_config = None
    if model_cfg.get("load_in_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=resolved_dtype,
        )

    load_kwargs = dict(
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        local_files_only=local_files_only,
    )
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        # In one-process-per-GPU distributed launches, device_map="auto"
        # would let every process see and potentially occupy every GPU. Pin
        # each quantized replica to its LOCAL_RANK instead.
        import os
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            load_kwargs["device_map"] = {"": local_rank}
        else:
            load_kwargs["device_map"] = "auto"
    # else: placed explicitly on a single device below, so every tensor this
    # codebase builds by hand has one unambiguous target device (see
    # crisp/utils/device.py). We deliberately avoid device_map="auto" for the
    # non-quantized path: naive pipeline-parallel sharding across GPUs works
    # for forward/backward, but it multiplies the ways a hand-built tensor can
    # land on the wrong shard, which is precisely the bug class this project
    # is trying to eliminate. For a model that only fits across multiple GPUs
    # unquantized, load_in_4bit + LoRA is the recommended path instead of
    # multi-GPU sharding of a full-precision model.

    try:
        try:
            model = _from_pretrained_with_dtype(model_cfg.name, resolved_dtype, attn_implementation=attn_impl, **load_kwargs)
        except (ImportError, ValueError) as e:
            if attn_impl == "flash_attention_2":
                logger.warning("flash_attention_2 requested but failed to load (%s); retrying with sdpa.", e)
                model = _from_pretrained_with_dtype(model_cfg.name, resolved_dtype, attn_implementation="sdpa", **load_kwargs)
            else:
                raise
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as e:
        raise RuntimeError(_DOWNLOAD_HELP.format(model_name=model_cfg.name, reason=e)) from e

    if quant_config is None:
        device = resolve_device()
        model = model.to(device)

    return model, tokenizer


def attach_lora(model, lora_cfg):
    """Wrap `model` with a LoRA adapter per `lora_cfg`.

    Returns `(model, is_peft: bool)`. If `lora_cfg.enabled` is False, returns
    the model unchanged with `is_peft=False` (e.g. for a full-finetune run --
    in that case every base-model parameter has `requires_grad=True` already,
    so nothing else needs to change downstream).
    """
    if not lora_cfg.get("enabled", True):
        return model, False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    is_quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    if is_quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # prepare_model_for_kbit_training does this (and a bit more, like
        # casting norm layers to fp32) for the quantized path; the
        # non-quantized path needs the input-grad hook explicitly, or
        # gradient checkpointing through a frozen embedding layer produces a
        # graph with no grad_fn at the checkpoint boundary and LoRA silently
        # gets zero gradient.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, True


def attach_or_load_lora(model, lora_cfg, initial_adapter: str | None = None):
    """Attach a fresh LoRA adapter or resume/initialize from an existing one.

    ``initial_adapter`` may point to an SFT adapter or a CRISP checkpoint.
    The adapter is loaded as trainable, avoiding accidental stacking of a
    second fresh adapter on top of the first.
    """
    if initial_adapter:
        from peft import PeftModel, prepare_model_for_kbit_training

        is_quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
        if is_quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)
        return model, True
    return attach_lora(model, lora_cfg)
