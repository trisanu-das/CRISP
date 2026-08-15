"""
EOS-token resolution and scanning helpers for generation.

Deliberately torch-free (plain ints/lists/sets only) so this logic is
directly unit-testable without a model or tokenizer object -- see
tests/test_eos_utils.py. Used by training/rollout.py.
"""
from __future__ import annotations

from typing import Optional


def resolve_eos_token_ids(generation_config_eos, tokenizer_eos_token_id: Optional[int]) -> list[int]:
    """Every token id that should count as "the model chose to stop here".

    `generation_config_eos` is whatever `model.generation_config.eos_token_id`
    holds -- `None`, a single int, or a list of ints (HF supports all three;
    Qwen checkpoints in particular often ship a list, e.g. both `<|im_end|>`
    and `<|endoftext|>`). Relying on only the tokenizer's single
    `eos_token_id` can mean `generate()` fails to recognize a stop token the
    model actually emits, so generation silently runs all the way to
    `max_new_tokens` on every rollout instead of stopping naturally --
    which also starves responses of ever reaching a concluding answer.
    `response_len` pinned exactly at `max_new_tokens` with zero variance
    across many rollouts is the signature of this happening.
    """
    ids: set[int] = set()
    if generation_config_eos is not None:
        if isinstance(generation_config_eos, int):
            ids.add(generation_config_eos)
        else:
            ids.update(int(x) for x in generation_config_eos)
    if tokenizer_eos_token_id is not None:
        ids.add(int(tokenizer_eos_token_id))
    if not ids:
        raise ValueError(
            "Could not resolve any eos_token_id from either the model's generation_config "
            "or the tokenizer -- generation would never stop early. Check that the model/"
            "tokenizer were loaded correctly."
        )
    return sorted(ids)


def first_eos_cutoff(token_ids: list[int], eos_ids: set[int]) -> Optional[int]:
    """Index one-past the first occurrence of any id in `eos_ids` within
    `token_ids`, or None if none appear.
    """
    for i, t in enumerate(token_ids):
        if t in eos_ids:
            return i + 1
    return None
