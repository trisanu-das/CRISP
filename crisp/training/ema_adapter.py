"""EMA adapter anchor for CIBO v2.

CIBO v2's `loss_total = loss_rl + beta * loss_ib + alpha * loss_anchor`
needs a frozen "previous-step student" to compute the anchor
cross-entropy term against. Rather than keep a second copy of a 7B
base model, this module tracks an EMA of only the trainable
parameters (LoRA adapters, in practice) and provides a context
manager that swaps the EMA values into the live parameters for the
duration of an EMA-anchored forward, then restores the live values
in `finally` -- so an exception in the caller's forward path still
restores the optimizer model to its pre-swap state.

The context manager is a *pure operation*: it does NOT call
`optimizer.step` or touch `param.grad`. The caller is the common
training loop, which only invokes this once per optimizer step (after
the optimizer step has already happened), via the post_optimizer_step
hook added in Task 6.

Two extra safety properties the plan calls out:

  - The EMA must always be on the live device/dtype at swap time.
    We convert at swap-out and convert back at restore so the live
    parameter's storage is preserved exactly (same device, same
    dtype, same shape).
  - The anchor target computed inside the context manager MUST NOT
    carry autograd through the swapped (EMA) parameters -- otherwise
    the EMA target would itself become a training signal. The
    `torch.no_grad()` block plus the fact that the EMA values are
    detached from the autograd graph guarantee this.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch


class EmaAdapterState:
    """EMA of a fixed list of trainable parameters (typically LoRA adapters).

    The EMA is initialised to the current parameter values; subsequent
    `update()` calls move it forward by the EMA decay formula.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter], decay: float):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1]; got {decay}")
        self.decay = float(decay)
        # Detached snapshot of the params at init time.
        self.ema_values: list[torch.Tensor] = [
            p.detach().clone() for p in params
        ]
        # Track the live parameters this state is bound to. The state
        # is positional -- the EMA tuple aligns 1:1 with this list.
        self._params: list[torch.nn.Parameter] = list(params)

    def update(self) -> None:
        """Advance the EMA by one step from the current live parameters."""
        new_ema: list[torch.Tensor] = []
        for live, stored in zip(self._params, self.ema_values):
            # The live tensor MAY carry grad; the EMA does not.
            # detach() the new observation so the EMA itself never
            # ends up on an autograd path.
            obs = live.detach()
            new_val = self.decay * stored + (1.0 - self.decay) * obs
            new_ema.append(new_val)
        self.ema_values = new_ema

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_values": [v.detach().cpu().clone() for v in self.ema_values],
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        # Materialise on the live parameter's device/dtype so swaps
        # don't trigger device transfers at use time.
        self.ema_values = [
            v.to(device=p.device, dtype=p.dtype)
            for v, p in zip(sd["ema_values"], self._params)
        ]


@contextmanager
def ema_target_forward(ema: EmaAdapterState, params: Sequence[torch.nn.Parameter]):
    """Swap EMA values into the live `params` for the duration of a `with` block.

    On exit (including exception), `params[i].data` is restored to its
    pre-swap value, with the same device and dtype as before. The
    caller is responsible for actually computing the EMA forward
    inside the block.

    The whole block (swap + yield + restore) runs under
    `torch.no_grad()` so the EMA-anchored target forward never
    contributes to the optimizer's autograd graph. The caller
    computes the live-student forward OUTSIDE this context manager.
    """
    if len(ema.ema_values) != len(params):
        raise ValueError(
            f"ema/params length mismatch: {len(ema.ema_values)} vs {len(params)}"
        )

    # 1. Snapshot the live values (must restore these exactly).
    saved = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            # 2. Swap in the EMA values, matching the live device/dtype.
            for live, ema_val in zip(params, ema.ema_values):
                live.data.copy_(ema_val.to(device=live.device, dtype=live.dtype))
            yield
    finally:
        with torch.no_grad():
            # 3. Always restore, even if the body raised.
            for live, snap in zip(params, saved):
                live.data.copy_(snap.to(device=live.device, dtype=live.dtype))