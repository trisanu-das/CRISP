"""
OPSD baseline: on-policy self-distillation only, no RL branch (no reward-
based loss term, no PC-Grad -- there's only one loss). Isolates whether the
self-distillation term alone is sufficient (README Section 5.3: "Is the RL
branch necessary, or does self-distillation alone suffice?").

Everything else -- the correctness gate, the teacher-hint construction, the
single batched teacher+student forward pass -- is kept identical to
crisp_step.py so this ablation changes exactly one thing (removing the RL
branch) rather than incidentally changing something else too.
"""
from __future__ import annotations

import torch

from crisp.training.grad_utils import compute_grads
from crisp.training.logprobs import build_scoring_batch, forward_kl_teacher_student, sequence_log_prob
from crisp.training.rollout import generate_rollouts
from crisp.training.teacher_student import build_teacher_student_sequences


def opsd_step(
    model, tokenizer, prompts: list[str], answers: list[str], reward_fn,
    params: list[torch.nn.Parameter], cfg,
) -> tuple[list[torch.Tensor], dict]:
    model.eval()
    sys_prompt = cfg.model.system_prompt
    n = len(prompts)

    rollouts = generate_rollouts(
        model, tokenizer, prompts, sys_prompt,
        max_prompt_length=cfg.rollout.max_prompt_length,
        max_new_tokens=cfg.rollout.max_new_tokens,
        temperature=cfg.rollout.temperature, top_p=cfg.rollout.top_p,
        do_sample=cfg.rollout.do_sample, num_return_sequences=1,
    )
    rewards = torch.tensor([reward_fn(r.response_text, a) for r, a in zip(rollouts, answers)], dtype=torch.float32)

    hint_template = cfg.data.teacher_hint_template
    # Same sequence construction as CRISP -- one source of truth for the
    # "teacher and student score the same response tokens" invariant.
    teacher_seqs, student_seqs = build_teacher_student_sequences(
        prompts, answers, rollouts, tokenizer=tokenizer,
        sys_prompt=sys_prompt, hint_template=hint_template,
    )

    all_seqs = teacher_seqs + student_seqs
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = build_scoring_batch(all_seqs, pad_id)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    _, row_log_probs = sequence_log_prob(model, input_ids, attention_mask, all_seqs)
    teacher_row_log_probs, student_row_log_probs = row_log_probs[:n], row_log_probs[n:]

    kl_per_example = forward_kl_teacher_student(teacher_row_log_probs, student_row_log_probs)
    gate = (rewards.to(model.device) < 1.0).float()
    denom = gate.sum().clamp(min=1.0)
    gated_kl = torch.stack([k * g for k, g in zip(kl_per_example, gate)])
    loss = gated_kl.sum() / denom

    grads = compute_grads(loss, params)
    metrics = {
        "reward_mean": rewards.mean().item(),
        "loss_sd": loss.item(),
        "pct_gated_correct": 1.0 - gate.mean().item(),
        "response_len_mean": sum(len(r.response_ids) for r in rollouts) / n,
    }
    return grads, metrics
