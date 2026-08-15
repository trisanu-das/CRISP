"""Shared response-aligned teacher/student scoring.

Single batched forward pass per micro-batch over both teacher and student
contexts, with the exact same sampled response tokens. This is the
extraction point for a guarantee the plan holds non-negotiable: teacher
and student score IDENTICAL response ids, just under different left
context -- the privileged answer hint changes ONLY the teacher prefix,
never the response tokens.

Centralizing this lets every method (CRISP, OPSD, future VACS, future
CIBO) share:

  - `build_teacher_student_sequences(...)` -- identical response_ids, two
    `ScoredSequence`s per rollout, response_len carried explicitly.
  - `score_teacher_student(...)` -- one batched forward, returning
    gathered token log-probs AND per-row log-prob distributions for both
    contexts.
  - `forward_kl_per_token(...)` -- per-token (NOT averaged) forward
    KL(teacher || student) on response tokens, with teacher detached
    from the autograd graph.

The original `crisp.training.logprobs.forward_kl_teacher_student` is
preserved as a thin compatibility wrapper so legacy CRISP/OPSD do not
need to change their loss/gate semantics. New methods should prefer the
per-token variant, which is what VACS/CIBO objective primitives need.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from crisp.training.logprobs import (
    ScoredSequence,
    build_scoring_batch,
    gather_token_log_probs,
    response_log_probs,
)


@dataclass
class TeacherStudentScores:
    """Result of one batched teacher+student scoring pass.

    `teacher_*` and `student_*` lists share the same length, in the same
    order as the input prompts/rollouts. Per-row tensors are independent
    (no view-sharing) and shaped by `response_len`.
    """

    teacher_sequences: list[ScoredSequence]
    student_sequences: list[ScoredSequence]
    teacher_token_logps: list[torch.Tensor]
    student_token_logps: list[torch.Tensor]
    teacher_row_log_probs: list[torch.Tensor]
    student_row_log_probs: list[torch.Tensor]
    teacher_per_token_kl: list[torch.Tensor]


def build_teacher_student_sequences(
    prompts: list[str],
    answers: list[str],
    rollouts,
    *,
    tokenizer,
    sys_prompt: str,
    hint_template: str,
) -> tuple[list[ScoredSequence], list[ScoredSequence]]:
    """Build paired teacher/student `ScoredSequence`s sharing response tokens.

    - Student sequences reuse the rollout's prompt_ids as the prefix.
    - Teacher sequences prepend a chat-template prefix that injects the
      privileged answer hint into the SYSTEM message (no fabricated role).
    - In BOTH cases, the response_ids appended at the end are bit-for-bit
      identical to the rollout's response tokens.
    - An empty response is preserved as `response_len=0`; the caller is
      responsible for deciding whether to skip or fail loudly. Silently
      scoring a zero-length response would corrupt downstream reductions.
    """
    teacher_seqs: list[ScoredSequence] = []
    student_seqs: list[ScoredSequence] = []
    for prompt, answer, rollout in zip(prompts, answers, rollouts):
        response_ids = rollout.response_ids

        student_seqs.append(ScoredSequence(
            ids=rollout.prompt_ids + response_ids,
            response_start=len(rollout.prompt_ids),
            response_len=len(response_ids),
        ))

        teacher_messages = [
            {"role": "system", "content": sys_prompt + hint_template.format(answer=answer)},
            {"role": "user", "content": prompt},
        ]
        teacher_prefix_text = tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True
        )
        teacher_prefix_ids = tokenizer(teacher_prefix_text, add_special_tokens=False)["input_ids"]
        teacher_seqs.append(ScoredSequence(
            ids=teacher_prefix_ids + response_ids,
            response_start=len(teacher_prefix_ids),
            response_len=len(response_ids),
        ))
    return teacher_seqs, student_seqs


def score_teacher_student(
    *,
    model,
    teacher_sequences: list[ScoredSequence],
    student_sequences: list[ScoredSequence],
    pad_id: int,
) -> TeacherStudentScores:
    """Run one batched teacher+student forward pass and package the results.

    Sequence order is `[...teacher, ...student]`, fixed for the lifetime of
    the micro-batch. Padding tokens never enter any response span because
    every slice below is bounded by `response_start`/`response_len` from
    the `ScoredSequence` records.
    """
    n = len(teacher_sequences)
    if len(student_sequences) != n:
        raise ValueError(
            f"teacher/student sequence counts disagree ({n} vs {len(student_sequences)}); "
            "they must share the same prompts and rollouts."
        )

    all_seqs = teacher_sequences + student_sequences
    input_ids, attention_mask = build_scoring_batch(all_seqs, pad_id)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    input_ids_for_gather = input_ids.to(logits.device)

    row_log_probs = response_log_probs(logits, input_ids_for_gather, all_seqs)
    token_log_probs = gather_token_log_probs(row_log_probs, all_seqs)

    teacher_row_log_probs = row_log_probs[:n]
    student_row_log_probs = row_log_probs[n:]
    teacher_token_logps = token_log_probs[:n]
    student_token_logps = token_log_probs[n:]

    per_token_kl = forward_kl_per_token(teacher_row_log_probs, student_row_log_probs)

    return TeacherStudentScores(
        teacher_sequences=teacher_sequences,
        student_sequences=student_sequences,
        teacher_token_logps=teacher_token_logps,
        student_token_logps=student_token_logps,
        teacher_row_log_probs=teacher_row_log_probs,
        student_row_log_probs=student_row_log_probs,
        teacher_per_token_kl=per_token_kl,
    )


def forward_kl_per_token(
    teacher_row_log_probs: list[torch.Tensor],
    student_row_log_probs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Per-token KL(teacher || student), shape `[response_len_i]` per row.

    Critical contracts (each protects a documented silent-corruption bug):

    - **Per-token, NOT per-example average.** VACS/CIBO objective
      primitives need per-token credit; a per-example mean hides the
      signal that drives token-weighted RL.
    - **Teacher is detached** from the autograd graph: it is a target
      distribution for this micro-batch, never a co-trainable signal.
      Otherwise the same loss could be minimized by "loosening" the
      teacher instead of improving the student.
    - **Lengths MUST match**: teacher and student score the exact same
      sampled response ids under different left context. A length
      mismatch indicates one of them was built against the wrong
      response tokens and MUST crash, not be averaged away.
    """
    out: list[torch.Tensor] = []
    for i, (t_lp, s_lp) in enumerate(zip(teacher_row_log_probs, student_row_log_probs)):
        if t_lp.shape[0] != s_lp.shape[0]:
            raise ValueError(
                f"Row {i}: teacher/student response lengths differ ({t_lp.shape[0]} vs "
                f"{s_lp.shape[0]}). They must score the exact same sampled response tokens, "
                "just under different left-context -- check how teacher/student ScoredSequences "
                "were built (both should append the identical response_ids)."
            )
        if t_lp.shape[0] == 0:
            # Caller responsibility: build_teacher_student_sequences leaves
            # empty response rows untouched, and score_teacher_student
            # already runs through them. Returning an empty 1-D tensor
            # here keeps downstream reductions (sum/mean/clamp) well-defined.
            out.append(torch.zeros(0, dtype=t_lp.dtype, device=t_lp.device))
            continue
        t_lp = t_lp.detach()
        t_p = t_lp.exp()
        # per_token_kl[t] = sum_v p_T(v|t) * (log p_T(v|t) - log p_S(v|t))
        per_token = (t_p * (t_lp - s_lp)).sum(dim=-1)  # [response_len]
        out.append(per_token)
    return out
