"""
Rollout sampling: turns a batch of prompts into on-policy generations.

Shared by CRISP and every baseline (GRPO, REINFORCE++, OPSD): they all start
from the same "sample y_i ~ pi_theta(.|x_i)" step (README Section 2.3, step
1), so this is the one place that logic lives.

Three deliberate correctness choices, all aimed at bug classes the reference
project's own changelog and the user's prior "gibberish output" experience
both point at:

1. Generation uses hand-built LEFT padding (never the tokenizer's own
   `padding_side` state -- see model/load.py for why). Scoring passes
   elsewhere in this codebase use RIGHT padding instead; the two are never
   allowed to interact through shared mutable state.
2. `generate()` pads finished sequences in a batch out to the same length as
   the longest one, using eos/pad tokens as filler. Those filler tokens are
   trimmed off (at the first real eos) before a response is used anywhere
   else, so they can never leak into a loss/KL computation as if they were
   real generated content.
3. "The" eos token is resolved from every source that can define one, not
   just `tokenizer.eos_token_id`. Qwen models are a well-documented case
   where this matters: `generation_config.json` can list *multiple* valid
   stop tokens (e.g. both `<|im_end|>` and `<|endoftext|>`), and checking
   only the tokenizer's single eos means `generate()` can fail to recognize
   a stop token the model actually emits -- generation then silently runs
   all the way to `max_new_tokens` on every rollout instead of stopping
   naturally, which also starves the response of ever reaching a concluding
   answer. `response_len` pinned exactly at `max_new_tokens` with zero
   variance across many rollouts is the signature of this happening.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from crisp.training.eos_utils import first_eos_cutoff, resolve_eos_token_ids


@dataclass
class Rollout:
    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str
    prompt_text: str


def _left_pad(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, max_len - len(seq):] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, max_len - len(seq):] = 1
    return input_ids, attention_mask


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    num_return_sequences: int = 1,
) -> list[Rollout]:
    """Sample `num_return_sequences` on-policy continuations per prompt.

    `prompts` are raw problem statements; the chat template + system prompt
    is applied here so every caller constructs the student context
    identically. Assumes the caller has already put `model` in `.eval()`
    (see training/common_loop.py's module docstring for why dropout is kept
    off for the whole RL step, not just generation).
    """
    prompt_id_lists, prompt_texts = [], []
    for p in prompts:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_prompt_length:
            ids = ids[-max_prompt_length:]  # keep the tail: closest to the generation prompt
        prompt_id_lists.append(ids)
        prompt_texts.append(text)

    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = _left_pad(prompt_id_lists, pad_id)

    # Inputs go to `model.device`: correct even for a `device_map="auto"`-
    # sharded model, since that's the device HF's dispatch hooks expect
    # activations to originate from (see utils/device.py for why that is NOT
    # the same claim as "the whole model lives on one device").
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    generation_config_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_token_ids = resolve_eos_token_ids(generation_config_eos, tokenizer.eos_token_id)
    eos_id_set = set(eos_token_ids)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_id,
        eos_token_id=eos_token_ids,  # generate() accepts int or list[int]; a list covers every valid stop token
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    prompt_len = input_ids.shape[1]  # fixed: every row was left-padded to this same length

    rollouts: list[Rollout] = []
    for row in range(output.shape[0]):
        # HF `generate` with num_return_sequences=k repeats each input k
        # times via repeat_interleave (prompt0 x k, prompt1 x k, ...), so
        # integer-dividing the output row index by k recovers which prompt
        # it came from.
        prompt_idx = row // num_return_sequences
        new_tokens = output[row, prompt_len:].tolist()
        # `generate` keeps the batch rectangular by padding sequences that
        # finish early with eos/pad; truncate at the FIRST eos (any valid
        # one, see resolve_eos_token_ids) so those trailing filler tokens
        # never enter the response span used for loss/KL elsewhere. Padding
        # tokens leaking into a loss region is a classic way to quietly
        # corrupt training into gibberish.
        cut = first_eos_cutoff(new_tokens, eos_id_set)
        if cut is not None:
            new_tokens = new_tokens[:cut]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        rollouts.append(Rollout(
            prompt_ids=prompt_id_lists[prompt_idx],
            response_ids=new_tokens,
            response_text=response_text,
            prompt_text=prompt_texts[prompt_idx],
        ))
    return rollouts
