"""Model and tokenizer loading utilities for CRISP.

This module centralizes:
- HuggingFace causal LM loading
- tokenizer setup
- LoRA attachment via PEFT
- device / dtype normalization

The goal is to keep training and baseline scripts thin and to ensure that the
same model-loading contract is used everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


@dataclass(frozen=True)
class ModelLoadConfig:
    name: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    use_flash_attention_2: bool = True
    gradient_checkpointing: bool = True
    max_position_embeddings: Optional[int] = None
    load_in_4bit: bool = False


def auto_dtype() -> torch.dtype:
    """Pick a safe compute dtype for whatever GPU is actually present.

    T4 (compute capability 7.5) and P100 (6.0) do not have bf16 tensor-core
    support; Ampere+ (8.0+) does. Falls back to fp16 on anything older, and to
    fp32 if no CUDA device is visible at all.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _supports_flash_attention_2() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _load_with_attn_fallback(model_name: str, *, torch_dtype: torch.dtype, trust_remote_code: bool,
                              use_flash_attention_2: bool, quantization_config: Optional["BitsAndBytesConfig"] = None):
    """Load a causal LM, trying flash-attention-2 only when it can actually work,
    and falling back to sdpa (then eager) rather than crashing.
    """
    attn_candidates = []
    if use_flash_attention_2 and _supports_flash_attention_2():
        attn_candidates.append("flash_attention_2")
    attn_candidates.extend(["sdpa", "eager"])

    last_error: Optional[Exception] = None
    for attn_impl in attn_candidates:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_impl,
                quantization_config=quantization_config,
            )
        except Exception as e:  # pragma: no cover - depends on local env/hardware
            last_error = e
            continue
    raise RuntimeError(f"Failed to load {model_name} with any attention implementation") from last_error


@dataclass(frozen=True)
class LoraLoadConfig:
    enabled: bool = True
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    modules_to_save: tuple[str, ...] = ()


def _to_dtype(dtype_value: str | torch.dtype | None) -> torch.dtype:
    if dtype_value is None:
        return torch.bfloat16
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    value = str(dtype_value).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_value}")


def _config_get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_tokenizer(config: Mapping[str, Any] | str):
    """Load and normalize a tokenizer.

    Ensures:
    - padding token exists
    - left padding for causal LM generation
    - remote code is allowed when requested by the config
    """
    if isinstance(config, str):
        model_name = config
        trust_remote_code = True
    else:
        model_name = _config_get(config, "model.name")
        if not model_name:
            raise ValueError("Missing required config field: model.name")
        trust_remote_code = bool(_config_get(config, "model.trust_remote_code", True))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        # For decoder-only LMs, reusing EOS as PAD is the standard safe choice.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _apply_max_position_embeddings(model: torch.nn.Module, max_position_embeddings: Optional[int]) -> None:
    if max_position_embeddings is None:
        return
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = int(max_position_embeddings)


def load_base_model(config: Mapping[str, Any] | str):
    """Load the base causal language model before any PEFT wrapping."""
    if isinstance(config, str):
        model_name = config
        model_cfg = ModelLoadConfig(name=model_name)
    else:
        model_name = _config_get(config, "model.name")
        if not model_name:
            raise ValueError("Missing required config field: model.name")
        dtype_cfg = _config_get(config, "model.dtype", "auto")
        model_cfg = ModelLoadConfig(
            name=model_name,
            dtype=str(dtype_cfg),
            trust_remote_code=bool(_config_get(config, "model.trust_remote_code", True)),
            use_flash_attention_2=bool(_config_get(config, "model.use_flash_attention_2", True)),
            gradient_checkpointing=bool(_config_get(config, "model.gradient_checkpointing", True)),
            max_position_embeddings=_config_get(config, "model.max_position_embeddings", None),
            load_in_4bit=bool(_config_get(config, "model.load_in_4bit", False)),
        )

    dtype = auto_dtype() if model_cfg.dtype == "auto" else _to_dtype(model_cfg.dtype)

    quantization_config = None
    if model_cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = _load_with_attn_fallback(
        model_cfg.name,
        torch_dtype=dtype,
        trust_remote_code=model_cfg.trust_remote_code,
        use_flash_attention_2=model_cfg.use_flash_attention_2,
        quantization_config=quantization_config,
    )

    _apply_max_position_embeddings(model, model_cfg.max_position_embeddings)

    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    return model


def build_lora_config(config: Mapping[str, Any] | None = None) -> "LoraConfig":
    """Create a PEFT LoRA config from a mapping or defaults."""
    if LoraConfig is None or TaskType is None:
        raise RuntimeError("peft is not installed, cannot build LoRA config")

    if config is None:
        lora_cfg = LoraLoadConfig()
    else:
        lora_cfg = LoraLoadConfig(
            enabled=bool(_config_get(config, "lora.enabled", True)),
            r=int(_config_get(config, "lora.r", 64)),
            lora_alpha=int(_config_get(config, "lora.lora_alpha", 128)),
            lora_dropout=float(_config_get(config, "lora.lora_dropout", 0.05)),
            bias=str(_config_get(config, "lora.bias", "none")),
            target_modules=tuple(_config_get(config, "lora.target_modules", ())),
            modules_to_save=tuple(_config_get(config, "lora.modules_to_save", ())),
        )

    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lora_cfg.target_modules),
        modules_to_save=list(lora_cfg.modules_to_save),
    )


def apply_lora(model: torch.nn.Module, config: Mapping[str, Any] | None = None) -> torch.nn.Module:
    """Wrap a model with LoRA adapters if enabled in config.

    Returns the original model unchanged when LoRA is disabled.
    """
    if config is None:
        enabled = True
    else:
        enabled = bool(_config_get(config, "lora.enabled", True))

    if not enabled:
        return model

    if get_peft_model is None:
        raise RuntimeError("peft is not installed, cannot apply LoRA")

    # 4-bit (bitsandbytes) base models need this before LoRA is attached:
    # it upcasts norms to fp32, makes the input embedding require grad, and
    # enables gradient checkpointing in a way that's compatible with
    # quantized layers. Skipping it is a common source of silently-zero
    # gradients when training QLoRA on a low-VRAM GPU.
    is_quantized = bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False))
    if is_quantized:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("peft.prepare_model_for_kbit_training is unavailable but the base model is quantized")
        use_gc = bool(_config_get(config, "model.gradient_checkpointing", True)) if config is not None else True
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)

    lora_config = build_lora_config(config)
    return get_peft_model(model, lora_config)


def load_model(config: Mapping[str, Any] | str):
    """Load the base model and apply LoRA if configured."""
    model = load_base_model(config)
    if isinstance(config, str):
        return model
    return apply_lora(model, config)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the first parameter device, or CPU if the model is empty."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    return model


def unfreeze_parameters_by_name(model: torch.nn.Module, names: Sequence[str]) -> torch.nn.Module:
    name_set = set(names)
    for n, p in model.named_parameters():
        if any(key in n for key in name_set):
            p.requires_grad = True
    return model


def shared_weight_forward(model: torch.nn.Module, *args, **kwargs):
    """Thin wrapper kept for semantic clarity in teacher/student shared-weight runs.

    It simply forwards through the same model instance. The function exists so
    downstream code can read clearly when a teacher or student pass is being
    performed with shared weights.
    """
    return model(*args, **kwargs)
