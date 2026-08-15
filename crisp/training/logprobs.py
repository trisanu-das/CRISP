"""
Batched forward passes for scoring rollouts, and the log-prob / KL math that
CRISP (and every baseline) is built on.

This is the single most bug-sensitive module in the whole codebase: any
off-by-one in the causal-LM shift, or any padding token that leaks into a
"response" span, silently corrupts the training signal instead of crashing
-- which is exactly the shape of a "gibberish output after training" bug.
Padding, shifting, and span extraction are therefore centralized here, kept
deliberately simple (explicit per-example spans + a Python-level loop over
the batch, rather than a clever vectorized ragged-tensor scheme), and unit
tested in isolation (tests/test_logprobs.py) against tiny hand-computed
logits, independent of any real model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ScoredSequence:
    """One row's worth of book-keeping for a scoring forward pass."""

    ids: list[int]        # full token sequence: context (prompt [+ teacher hint]) + response
    response_start: int    # index into `ids` where the response begins
    response_len: int      # number of response tokens


def build_scoring_batch(sequences: list[ScoredSequence], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch of variable-length sequences for a scoring forward pass.

    Right-padding (not left) is deliberate: with right-padding, position 0 is
    token 0 for every row regardless of that row's length, so a row's
    `response_start` means the same absolute index whether or not the row
    was padded. Left-padding would require re-deriving every offset relative
    to the (per-row-varying) padding amount, which is exactly the kind of
    indirection this module exists to avoid. HF causal LMs handle a
    right-padded `attention_mask` correctly for a plain forward pass (as
    opposed to `generate`, which specifically needs left-padding -- see
    training/rollout.py).
    """
    max_len = max(len(s.ids) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, s in enumerate(sequences):
        input_ids[i, :len(s.ids)] = torch.tensor(s.ids, dtype=torch.long)
        attention_mask[i, :len(s.ids)] = 1
    return input_ids, attention_mask


def response_log_probs(
    logits: torch.Tensor, input_ids: torch.Tensor, sequences: list[ScoredSequence]
) -> list[torch.Tensor]:
    """Per-row token-level log-probs (log_softmax already applied) over just
    the response span, shape `[response_len_i, vocab]`.

    `logits` and `input_ids` must be the batch tensors returned by a single
    forward pass, with `logits[i]` / `input_ids[i]` corresponding to
    `sequences[i]`. Padding positions are never referenced: every slice below
    is bounded by `sequences[i].response_start/len`, which by construction
    only ever point at real tokens.
    """
    # Standard causal-LM shift: logits[:, t, :] is the model's distribution
    # over the token at position t+1. So `shift_logits[:, t, :]` is what
    # predicts `input_ids[:, t+1]`; the token at absolute position `p` is
    # therefore predicted by shift index `p - 1`.
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]

    out = []
    for i, s in enumerate(sequences):
        lo = s.response_start - 1
        hi = lo + s.response_len
        if lo < 0:
            raise ValueError(
                f"Row {i}: response_start={s.response_start} leaves no context token to predict "
                "the first response token from. Every sequence needs at least one context token."
            )
        if hi > shift_logits.shape[1]:
            raise ValueError(
                f"Row {i}: response span [{lo}:{hi}] exceeds available length {shift_logits.shape[1]}. "
                "A sequence was likely truncated after its response span was computed -- check "
                "max_prompt_length/max_new_tokens against how `ids` was built."
            )
        # Slice BEFORE softmax, not after: computing log_softmax over the
        # full (padded, prompt-including) sequence length and slicing
        # afterwards gives the identical numeric result (log_softmax is
        # applied independently per position) but wastes compute and memory
        # on the vocab-sized softmax for every prompt/padding position we're
        # about to throw away anyway.
        row_logits = shift_logits[i, lo:hi, :].float()
        row_log_probs = F.log_softmax(row_logits, dim=-1)

        row_ids = shift_ids[i, lo:hi]
        expected = torch.tensor(
            s.ids[s.response_start: s.response_start + s.response_len],
            device=row_ids.device, dtype=row_ids.dtype,
        )
        if not torch.equal(row_ids, expected):
            raise RuntimeError(
                f"Row {i}: response tokens read back from the padded batch don't match the "
                "tokens recorded when `sequences` was built. This means the batch was built or "
                "padded inconsistently -- check build_scoring_batch / how response spans were computed."
            )
        out.append(row_log_probs)
    return out


def gather_token_log_probs(row_log_probs: list[torch.Tensor], sequences: list[ScoredSequence]) -> list[torch.Tensor]:
    """log p(actual generated token) at each response position, shape `[response_len_i]`."""
    result = []
    for lp, s in zip(row_log_probs, sequences):
        ids = torch.tensor(
            s.ids[s.response_start: s.response_start + s.response_len], device=lp.device
        )
        result.append(lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1))
    return result


def sequence_log_prob(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, sequences: list[ScoredSequence]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One forward pass -> (per-row token log-probs `[resp_len_i]`, per-row
    full log-prob distributions `[resp_len_i, vocab]`) over each response span.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # `input_ids` is read back on `logits`' device, not assumed to already be
    # there: under `device_map="auto"` the two are not guaranteed to be the
    # same object even though they started out matching, since the lm_head
    # may live on a different shard than the embedding layer that first
    # consumed `input_ids`. See utils/device.py.
    input_ids_for_gather = input_ids.to(logits.device)
    row_log_probs = response_log_probs(logits, input_ids_for_gather, sequences)
    token_log_probs = gather_token_log_probs(row_log_probs, sequences)
    return token_log_probs, row_log_probs


def forward_kl_teacher_student(
    teacher_row_log_probs: list[torch.Tensor], student_row_log_probs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """KL[ teacher || student ], averaged over the response span, per example.

    Direction is load-bearing (README Section 2.1): the teacher is the
    ground-truth-conditioned pass, the student is the unconditioned one, and
    KL(teacher || student) -- not the reverse -- is what penalizes the
    student for putting too little mass where the teacher is confident.
    Swapping the argument order here would silently change the method into
    the reverse (mode-seeking) KL the README explicitly says is *not* used.

    The teacher distribution is detached: it is a target for this step, not
    something this loss should also be shaping. That's standard practice for
    a distillation loss generally, and here specifically it closes off a
    degenerate shortcut that's only available because teacher and student
    are the same weights -- minimizing the loss by making the *teacher*
    easier to match, instead of improving the student.
    """
    out = []
    for i, (t_lp, s_lp) in enumerate(zip(teacher_row_log_probs, student_row_log_probs)):
        if t_lp.shape[0] != s_lp.shape[0]:
            raise ValueError(
                f"Row {i}: teacher/student response lengths differ ({t_lp.shape[0]} vs "
                f"{s_lp.shape[0]}). They must score the exact same sampled response tokens, "
                "just under different left-context -- check how teacher/student ScoredSequences "
                "were built (both should append the identical response_ids)."
            )
        if t_lp.shape[0] == 0:
            out.append(torch.zeros((), dtype=s_lp.dtype, device=s_lp.device))
            continue
        t_lp = t_lp.detach()
        t_p = t_lp.exp()
        per_token_kl = (t_p * (t_lp - s_lp)).sum(dim=-1)  # [response_len]
        out.append(per_token_kl.mean())
    return out
