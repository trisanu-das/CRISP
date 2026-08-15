"""Configuration validation shared by every training launcher method.

The loader intentionally remains permissive: it deep-merges arbitrary YAML so
experiments can add metadata without changing a schema. This module validates
the values that affect the safety and semantics of VACS/CIBO training before a
launcher imports any training/model code or starts a dataset download.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when a method configuration is invalid or unsafe to run."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        section = config[name]
    except KeyError as exc:
        raise ConfigValidationError(f"Missing required configuration section '{name}'.") from exc
    if not isinstance(section, Mapping):
        raise ConfigValidationError(f"Configuration section '{name}' must be a mapping.")
    return section


def _number(section: Mapping[str, Any], key: str, path: str) -> float:
    try:
        value = section[key]
    except KeyError as exc:
        raise ConfigValidationError(f"Missing required configuration value '{path}'.") from exc
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ConfigValidationError(f"Configuration value '{path}' must be a finite number.")
    return float(value)


def _unit_interval(section: Mapping[str, Any], key: str, path: str) -> float:
    value = _number(section, key, path)
    if not 0.0 <= value <= 1.0:
        raise ConfigValidationError(f"Configuration value '{path}' must be in [0, 1].")
    return value


def _ema_decay(section: Mapping[str, Any], key: str, path: str) -> float:
    value = _number(section, key, path)
    if not 0.0 <= value < 1.0:
        raise ConfigValidationError(f"Configuration value '{path}' must be in [0, 1).")
    return value


def _positive(section: Mapping[str, Any], key: str, path: str, *, allow_zero: bool = False) -> float:
    value = _number(section, key, path)
    if value < 0.0 or (value == 0.0 and not allow_zero):
        comparison = "non-negative" if allow_zero else "positive"
        raise ConfigValidationError(f"Configuration value '{path}' must be {comparison}.")
    return value


def _validate_vacs(vacs: Mapping[str, Any]) -> None:
    _unit_interval(vacs, "token_credit_mix", "vacs.token_credit_mix")
    epsilon = _positive(vacs, "token_weight_epsilon", "vacs.token_weight_epsilon", allow_zero=True)
    if epsilon >= 1.0:
        raise ConfigValidationError(
            "Configuration value 'vacs.token_weight_epsilon' must be below 1 so clipped token weights remain positive."
        )
    _positive(vacs, "sd_weight", "vacs.sd_weight", allow_zero=True)
    _positive(vacs, "soft_gate_slope", "vacs.soft_gate_slope")
    _unit_interval(vacs, "reward_threshold", "vacs.reward_threshold")

    adaptive_mix = _section(vacs, "adaptive_mix")
    enabled = adaptive_mix.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigValidationError("Configuration value 'vacs.adaptive_mix.enabled' must be boolean.")
    _ema_decay(adaptive_mix, "ema_decay", "vacs.adaptive_mix.ema_decay")
    multiplier_min = _positive(adaptive_mix, "multiplier_min", "vacs.adaptive_mix.multiplier_min")
    multiplier_max = _positive(adaptive_mix, "multiplier_max", "vacs.adaptive_mix.multiplier_max")
    if multiplier_min > multiplier_max:
        raise ConfigValidationError(
            "Configuration values 'vacs.adaptive_mix.multiplier_min' and "
            "'vacs.adaptive_mix.multiplier_max' must be ordered (min <= max)."
        )


def _validate_cibo_v2(cibo: Mapping[str, Any]) -> None:
    _unit_interval(cibo, "credit_lambda", "cibo_v2.credit_lambda")

    beta = _positive(cibo, "beta", "cibo_v2.beta", allow_zero=True)
    beta_min = _positive(cibo, "beta_min", "cibo_v2.beta_min", allow_zero=True)
    beta_max = _positive(cibo, "beta_max", "cibo_v2.beta_max", allow_zero=True)
    if beta_min > beta_max:
        raise ConfigValidationError(
            "Configuration values 'cibo_v2.beta_min' and 'cibo_v2.beta_max' must be ordered (min <= max)."
        )
    if not beta_min <= beta <= beta_max:
        raise ConfigValidationError(
            "Configuration value 'cibo_v2.beta' must lie within [cibo_v2.beta_min, cibo_v2.beta_max]."
        )

    beta_mode = cibo.get("beta_mode")
    if beta_mode not in {"fixed", "adaptive_sign"}:
        raise ConfigValidationError("Configuration value 'cibo_v2.beta_mode' must be 'fixed' or 'adaptive_sign'.")
    _positive(cibo, "beta_step_size", "cibo_v2.beta_step_size", allow_zero=True)
    _ema_decay(cibo, "beta_ema_decay", "cibo_v2.beta_ema_decay")

    warmup_steps = cibo.get("beta_reference_warmup_steps")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ConfigValidationError(
            "Configuration value 'cibo_v2.beta_reference_warmup_steps' must be a non-negative integer."
        )
    _positive(cibo, "beta_denominator_epsilon", "cibo_v2.beta_denominator_epsilon")

    if cibo.get("ib_reduction") not in {"sequence_sum", "token_mean"}:
        raise ConfigValidationError(
            "Configuration value 'cibo_v2.ib_reduction' must be 'sequence_sum' or 'token_mean'."
        )
    _positive(cibo, "soft_gate_slope", "cibo_v2.soft_gate_slope")
    _unit_interval(cibo, "reward_threshold", "cibo_v2.reward_threshold")
    _positive(cibo, "anchor_alpha", "cibo_v2.anchor_alpha", allow_zero=True)
    _ema_decay(cibo, "anchor_ema_decay", "cibo_v2.anchor_ema_decay")
    if cibo.get("anchor_storage") not in {"same", "cpu"}:
        raise ConfigValidationError("Configuration value 'cibo_v2.anchor_storage' must be 'same' or 'cpu'.")


def validate_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate VACS/CIBO safety constraints and return ``config`` unchanged.

    Calling this is deliberately explicit so library callers can inspect a
    partially merged experimental config. Every CLI launcher calls it before
    importing a method runner, model library, or dataset loader.
    """
    if not isinstance(config, Mapping):
        raise ConfigValidationError("The root configuration must be a mapping.")
    _validate_vacs(_section(config, "vacs"))
    _validate_cibo_v2(_section(config, "cibo_v2"))
    return config
