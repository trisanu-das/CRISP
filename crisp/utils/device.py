"""
Device-handling utilities.

The most common source of "data-device mismatch" crashes in a hand-rolled
RL/training loop is *not* the model itself -- `from_pretrained(...,
device_map=...)` gets that right on its own. It's every *other* tensor the
loop creates by hand: reward tensors, advantage tensors, attention masks
assembled manually, gather indices, and so on.

The rule enforced throughout this codebase: never cache a single global
`device` variable and assume every tensor should live there. Instead, derive
the device from the tensor you are about to combine something with (usually
a model output tensor for that particular forward pass), and route every
hand-built tensor through `.to(that_device)` right before it touches a model
tensor. This matters especially when `device_map="auto"` shards a model
across multiple GPUs: `model.device` (where you should send *inputs*) and
the device that `model(...).logits` actually comes out on are not
guaranteed to be the same object, so code that assumes one global device for
everything can silently break the moment a model is sharded.
"""
from __future__ import annotations

import os

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick a device once, for things like *inputs* headed into the model.

    This is intentionally not memoized in a module-level global: call it
    again if the environment could have changed. Once a model is loaded,
    prefer `model.device` over calling this a second time -- see the module
    docstring for why those two are not always interchangeable.
    """
    if preferred is not None and preferred != "auto":
        return torch.device(preferred)
    if os.environ.get("CRISP_FORCE_CPU", "") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj, device: torch.device):
    """Recursively move tensors in a (nested) dict/list/tuple onto `device`.

    Leaves non-tensor values untouched instead of raising, since batches
    routinely carry plain-Python bookkeeping (ints, strings, None) alongside
    tensors, and crashing on those is a common self-inflicted wound.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cast = list if isinstance(obj, list) else tuple
        return cast(to_device(v, device) for v in obj)
    return obj


def assert_same_device(*tensors: torch.Tensor, context: str = "") -> None:
    """Cheap runtime guard: call this right before an op that combines
    tensors constructed in different places -- the classic source of
    `RuntimeError: Expected all tensors to be on the same device`.

    Deliberately a no-op-fast-path for the common case (all on one device)
    so it's cheap enough to sprinkle liberally rather than reserve for
    "suspicious" spots only.
    """
    devices = {t.device for t in tensors if torch.is_tensor(t)}
    if len(devices) > 1:
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"Device mismatch{where}: found tensors on {sorted(str(d) for d in devices)}. "
            "Move everything onto the same device before this op -- see crisp/utils/device.py."
        )
