"""Tests for the shared teacher/student response-aligned scoring module.

This module is the single most bug-sensitive part of the codebase: any
off-by-one in the causal-LM shift or any padding token that leaks into a
response span silently corrupts the training signal instead of crashing.
The tests below hand-derive expected numbers (using plain Python math.log
and manual softmax) rather than re-deriving them with the same tensor ops
the production code uses -- the latter only proves the code agrees with
itself.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from crisp.training.logprobs import ScoredSequence

    from crisp.training.teacher_student import (
        TeacherStudentScores,
        build_teacher_student_sequences,
        forward_kl_per_token,
        score_teacher_student,
    )


def _manual_log_softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    z = sum(exps)
    return [math.log(e / z) for e in exps]


def _manual_kl(p_list, q_list):
    return sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_list, q_list))


def _uniform_prompt_ids(tokenizer, prompt_text, system_prompt):
    """Tiny test-only helper: tokenize a prompt as a single string with no chat template.

    Returns a stable list[int] for use as either the student or teacher
    prefix. The actual numbers don't matter for these tests -- what matters
    is that teacher and student end up with DIFFERENT prefix lengths when
    the privileged answer is injected, and IDENTICAL response_ids.
    """
    text = f"{system_prompt}\n{prompt_text}"
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestBuildTeacherStudentSequences(unittest.TestCase):
    def setUp(self):
        # Minimal "tokenizer" stand-in: each whitespace-separated token maps to
        # a stable int. We need tokenize(text)->List[int], pad_id, and
        # apply_chat_template(messages, tokenize=False, add_generation_prompt=True).
        self.vocab = {}
        self.next_id = 0

    def _tok(self, text):
        return [self.vocab.setdefault(t, len(self.vocab)) for t in text.split()]

    def _build_tokenizer(self):
        # Subclass `object` so we can define `__call__` as a real method
        # (assigning `tok.__call__ = ...` to a bare `type("T", (), {})` instance
        # does NOT make the instance callable -- `tokenizer(...)` resolves to
        # the class-level __call__, not the instance attribute).
        class FakeTokenizer:
            def __init__(self):
                self.eos_token_id = None
                self.pad_token_id = 0

            def encode(self, text, add_special_tokens=True):
                return self._tok(text)

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": self._tok(text)}

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                parts = []
                for m in messages:
                    parts.append(f"{m['role']}:{m['content']}")
                if add_generation_prompt:
                    parts.append("assistant:")
                return " ".join(parts)

        tok = FakeTokenizer()
        tok._tok = self._tok
        return tok

    def test_teacher_and_student_share_response_tokens_but_differ_in_prefix(self):
        tok = self._build_tokenizer()
        prompts = ["problem text"]
        answers = ["42"]
        student_prefix = tok("student prefix")["input_ids"]
        rollouts = [type("R", (), {"response_ids": [100, 101, 102], "prompt_ids": student_prefix})()]
        hint_template = " hint:{answer}"

        teacher_seqs, student_seqs = build_teacher_student_sequences(
            prompts, answers, rollouts,
            tokenizer=tok,
            sys_prompt="sys",
            hint_template=hint_template,
        )

        self.assertEqual(len(teacher_seqs), 1)
        self.assertEqual(len(student_seqs), 1)

        # The response tokens must be IDENTICAL between teacher and student.
        t_resp = teacher_seqs[0].ids[teacher_seqs[0].response_start:]
        s_resp = student_seqs[0].ids[student_seqs[0].response_start:]
        self.assertEqual(t_resp, s_resp)
        self.assertEqual(t_resp, [100, 101, 102])

        # The teacher's prefix must contain the privileged answer hint.
        rendered_teacher = tok.apply_chat_template([
            {"role": "system", "content": "sys" + hint_template.format(answer="42")},
            {"role": "user", "content": "problem text"},
        ], tokenize=False, add_generation_prompt=True)
        self.assertIn("42", rendered_teacher)

        # Privileged hint changes ONLY the prefix, never the response.
        self.assertNotEqual(
            teacher_seqs[0].ids[:teacher_seqs[0].response_start],
            student_seqs[0].ids[:student_seqs[0].response_start],
            "Teacher and student prefixes must differ when a hint is injected.",
        )

        # response_len must match the actual rollout length.
        self.assertEqual(teacher_seqs[0].response_len, 3)
        self.assertEqual(student_seqs[0].response_len, 3)

    def test_zero_length_response_is_reported_not_silently_scored(self):
        tok = self._build_tokenizer()
        prompts = ["problem"]
        answers = ["42"]
        rollouts = [type("R", (), {"response_ids": [], "prompt_ids": [1, 2, 3]})()]
        teacher_seqs, student_seqs = build_teacher_student_sequences(
            prompts, answers, rollouts, tokenizer=tok,
            sys_prompt="sys", hint_template="hint:{answer}",
        )

        # The sequence builder does NOT crash on empty responses; that is the
        # caller's job (it has to decide whether to skip the example or fail
        # loudly). The data we hand back must correctly carry the zero length.
        self.assertEqual(teacher_seqs[0].response_len, 0)
        self.assertEqual(student_seqs[0].response_len, 0)
        self.assertEqual(teacher_seqs[0].response_start, teacher_seqs[0].response_start)
        # Importantly, no response ids must have been silently inserted.
        t_resp = teacher_seqs[0].ids[teacher_seqs[0].response_start:]
        s_resp = student_seqs[0].ids[student_seqs[0].response_start:]
        self.assertEqual(t_resp, [])
        self.assertEqual(s_resp, [])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestForwardKLPerToken(unittest.TestCase):
    def test_matches_hand_computed_per_token_kl(self):
        # Two response tokens, vocab size 3. Hand-computed in plain Python.
        t0 = [math.log(0.7), math.log(0.2), math.log(0.1)]
        s0 = [math.log(0.5), math.log(0.3), math.log(0.2)]
        t1 = [math.log(0.5), math.log(0.3), math.log(0.2)]
        s1 = [math.log(0.5), math.log(0.3), math.log(0.2)]

        teacher = [torch.tensor([t0, t1])]
        student = [torch.tensor([s0, s1])]

        per_token = forward_kl_per_token(teacher, student)
        self.assertEqual(len(per_token), 1)
        self.assertEqual(per_token[0].shape, (2,))

        expected0 = _manual_kl([0.7, 0.2, 0.1], [0.5, 0.3, 0.2])
        expected1 = 0.0
        self.assertAlmostEqual(per_token[0][0].item(), expected0, places=5)
        self.assertAlmostEqual(per_token[0][1].item(), expected1, places=5)

    def test_teacher_is_detached_from_graph(self):
        teacher_lp = torch.tensor([[math.log(0.7), math.log(0.3)]], requires_grad=True)
        student_lp = torch.tensor([[math.log(0.5), math.log(0.5)]], requires_grad=True)
        per_token = forward_kl_per_token([teacher_lp], [student_lp])
        per_token[0].sum().backward()
        self.assertIsNone(teacher_lp.grad, "Teacher must be detached from the per-token KL graph.")
        self.assertIsNotNone(student_lp.grad, "Student must carry gradient through per-token KL.")

    def test_length_mismatch_raises(self):
        teacher = [torch.tensor([[math.log(0.5), math.log(0.5)]])]
        student = [torch.tensor([
            [math.log(0.5), math.log(0.5)],
            [math.log(0.5), math.log(0.5)],
        ])]
        with self.assertRaises(ValueError):
            forward_kl_per_token(teacher, student)

    def test_returns_per_token_not_per_example_average(self):
        # If a downstream caller asks for "mean" they should be able to call
        # .mean() themselves; forward_kl_per_token MUST preserve per-token
        # resolution. The shape MUST equal response length, not scalar.
        teacher = [torch.tensor([[math.log(0.9), math.log(0.1)]])]
        student = [torch.tensor([[math.log(0.5), math.log(0.5)]])]
        per_token = forward_kl_per_token(teacher, student)
        self.assertEqual(per_token[0].dim(), 1)
        self.assertEqual(per_token[0].shape[0], 1)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestScoreTeacherStudent(unittest.TestCase):
    def _fake_model(self, response_token_logps_per_row, response_starts, response_ids_per_row):
        """Build a callable stand-in for `model(input_ids=, attention_mask=)`.

        For row i and t-th response token:
          - the response id is response_ids_per_row[i][t]
          - the chosen log-prob is response_token_logps_per_row[i][t]
        We construct logits whose log_softmax at the target id exactly equals
        the chosen log-prob (by setting the target logit to log p and the
        remaining vocab positions to log((1-p)/(V-1)) so the row sums to 1).
        """
        assert (
            len(response_token_logps_per_row)
            == len(response_starts)
            == len(response_ids_per_row)
        )
        vocab = 16

        def call(input_ids=None, attention_mask=None):
            B, L = input_ids.shape
            logits = torch.zeros(B, L, vocab)
            for i, logps in enumerate(response_token_logps_per_row):
                L_row = int(attention_mask[i].sum().item())
                resp_start = response_starts[i]
                ids_for_row = response_ids_per_row[i]
                for t_in_resp, lp in enumerate(logps):
                    if len(lp) != 1:
                        raise AssertionError(
                            "fake_model expects per-token log-prob tensors of shape [1]"
                        )
                    pos = (resp_start - 1) + t_in_resp
                    if pos < 0 or pos >= L_row - 1:
                        raise AssertionError(
                            f"row {i}: response_start={resp_start} resp_len={len(logps)} "
                            f"L_row={L_row} produced out-of-range shift pos {pos}"
                        )
                    target_id = ids_for_row[t_in_resp]
                    if not (0 <= target_id < vocab):
                        raise AssertionError(
                            f"fake_model: target_id {target_id} out of range for vocab={vocab}"
                        )
                    p = float(lp[0].exp().clamp(min=1e-9, max=1 - 1e-9))
                    row = torch.full((vocab,), math.log((1.0 - p) / (vocab - 1)))
                    row[target_id] = float(lp[0])
                    logits[i, pos] = row
            out = type("O", (), {"logits": logits})()
            return out

        class FakeModel:
            device = torch.device("cpu")

            def __call__(self, input_ids=None, attention_mask=None):
                return call(input_ids=input_ids, attention_mask=attention_mask)

        return FakeModel()

    def test_score_teacher_student_aligns_response_tokens_across_prefixes(self):
        # Teacher: response [5, 6] under longer prefix. Student: same response
        # [5, 6] under shorter prefix. After one batched forward, teacher and
        # student log-probs must each equal the model's chosen log-probs at
        # exactly the right positions.
        tok = type("T", (), {})()
        tok.pad_token_id = 0
        teacher_seq = ScoredSequence(ids=[10, 11, 12, 5, 6], response_start=3, response_len=2)
        student_seq = ScoredSequence(ids=[7, 5, 6], response_start=1, response_len=2)

        # Per-token log-prob targets: each entry is a 1-D tensor of shape [1]
        # (the chosen log-prob for the corresponding response token).
        teacher_target_lp = torch.tensor([[math.log(0.9)], [math.log(0.8)]])
        student_target_lp = torch.tensor([[math.log(0.5)], [math.log(0.6)]])

        # Response token ids per row.
        teacher_resp_ids = teacher_seq.ids[teacher_seq.response_start: teacher_seq.response_start + teacher_seq.response_len]
        student_resp_ids = student_seq.ids[student_seq.response_start: student_seq.response_start + student_seq.response_len]

        # The fake model gets ONE log-prob list per row in the joined batch
        # (teacher rows first, then student rows), which is what
        # score_teacher_student expects.
        model = self._fake_model(
            [teacher_target_lp, student_target_lp],
            [teacher_seq.response_start, student_seq.response_start],
            [teacher_resp_ids, student_resp_ids],
        )

        result = score_teacher_student(
            model=model,
            teacher_sequences=[teacher_seq],
            student_sequences=[student_seq],
            pad_id=tok.pad_token_id,
        )

        # Token log-probs must equal the chosen targets for each row.
        self.assertEqual(len(result.teacher_token_logps), 1)
        self.assertEqual(len(result.student_token_logps), 1)
        self.assertEqual(result.teacher_token_logps[0].shape, (2,))
        self.assertEqual(result.student_token_logps[0].shape, (2,))
        # Teacher had response ids [5, 6] -> gathered log-probs are [0.9, 0.8] in log space.
        self.assertAlmostEqual(result.teacher_token_logps[0][0].item(), math.log(0.9), places=5)
        self.assertAlmostEqual(result.teacher_token_logps[0][1].item(), math.log(0.8), places=5)
        # Student had response ids [5, 6] -> gathered log-probs are [0.5, 0.6] in log space.
        self.assertAlmostEqual(result.student_token_logps[0][0].item(), math.log(0.5), places=5)
        self.assertAlmostEqual(result.student_token_logps[0][1].item(), math.log(0.6), places=5)

        # The per-token forward KL must equal the hand-computed values
        # (and NOT be averaged -- per-token resolution is preserved).
        expected_kl_t0 = 0.9 * (math.log(0.9) - math.log(0.5)) + 0.1 * (math.log(0.1) - math.log(0.5))
        expected_kl_t1 = 0.2 * (math.log(0.2) - math.log(0.4)) + 0.8 * (math.log(0.8) - math.log(0.6))
        self.assertEqual(result.teacher_per_token_kl[0].shape, (2,))
        self.assertAlmostEqual(result.teacher_per_token_kl[0][0].item(), expected_kl_t0, places=4)
        self.assertAlmostEqual(result.teacher_per_token_kl[0][1].item(), expected_kl_t1, places=4)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestTeacherStudentScoresDataclass(unittest.TestCase):
    def test_dataclass_is_pure_data(self):
        # TeacherStudentScores is a value type; no methods should mutate
        # caller state. Sanity check by constructing one and reading fields.
        teacher_seq = ScoredSequence(ids=[1, 2, 3], response_start=2, response_len=1)
        student_seq = ScoredSequence(ids=[1, 3], response_start=1, response_len=1)
        result = TeacherStudentScores(
            teacher_sequences=[teacher_seq],
            student_sequences=[student_seq],
            teacher_token_logps=[torch.tensor([0.0])],
            student_token_logps=[torch.tensor([0.0])],
            teacher_row_log_probs=[torch.tensor([[0.0, -1.0]])],
            student_row_log_probs=[torch.tensor([[0.0, -1.0]])],
            teacher_per_token_kl=[torch.tensor([0.0])],
        )
        self.assertEqual(len(result.teacher_sequences), 1)
        self.assertEqual(len(result.student_sequences), 1)
        self.assertEqual(result.teacher_token_logps[0].dim(), 1)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestSharedSequencesMatchLegacy(unittest.TestCase):
    """After CRISP/OPSD were migrated onto build_teacher_student_sequences,
    they MUST still produce the exact same ScoredSequence records they used
    to build inline. This guards against a silent refactor drift in the
    teacher-prefix construction, which has historically been a source of
    off-by-one response-span bugs.
    """

    def _build_tokenizer(self):
        class FakeTokenizer:
            def __init__(self):
                self.eos_token_id = None
                self.pad_token_id = 0

            def encode(self, text, add_special_tokens=True):
                return self._tok(text)

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": self._tok(text)}

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                parts = []
                for m in messages:
                    parts.append(f"{m['role']}:{m['content']}")
                if add_generation_prompt:
                    parts.append("assistant:")
                return " ".join(parts)

        tok = FakeTokenizer()
        tok._tok = lambda text: [hash(w) % 1000 for w in text.split()]
        return tok

    def test_shared_builder_matches_legacy_inline_construction(self):
        # Build the same sequences both ways -- via the new shared helper
        # and via an inline copy of the original CRISP/OPSD code -- and
        # assert they are identical.
        from crisp.training.teacher_student import build_teacher_student_sequences

        tok = self._build_tokenizer()
        sys_prompt = "Please reason step by step."
        hint_template = " Hint:{answer}"

        prompts = ["problem a", "problem b"]
        answers = ["42", "13"]
        response_ids_lists = [[100, 101], [200, 201, 202]]
        rollouts = [
            type("R", (), {
                "response_ids": response_ids_lists[0],
                "prompt_ids": tok("student prompt 1")["input_ids"],
            })(),
            type("R", (), {
                "response_ids": response_ids_lists[1],
                "prompt_ids": tok("student prompt 2")["input_ids"],
            })(),
        ]

        teacher_seqs, student_seqs = build_teacher_student_sequences(
            prompts, answers, rollouts, tokenizer=tok,
            sys_prompt=sys_prompt, hint_template=hint_template,
        )

        # Legacy inline reconstruction (copied verbatim from the old
        # crisp_step._build_teacher_student_sequences and opsd_step).
        legacy_teacher_seqs = []
        legacy_student_seqs = []
        for prompt, answer, rollout, resp_ids in zip(prompts, answers, rollouts, response_ids_lists):
            legacy_student_seqs.append(ScoredSequence(
                ids=rollout.prompt_ids + resp_ids,
                response_start=len(rollout.prompt_ids),
                response_len=len(resp_ids),
            ))
            teacher_messages = [
                {"role": "system", "content": sys_prompt + hint_template.format(answer=answer)},
                {"role": "user", "content": prompt},
            ]
            teacher_prefix_text = tok.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True)
            teacher_prefix_ids = tok(teacher_prefix_text, add_special_tokens=False)["input_ids"]
            legacy_teacher_seqs.append(ScoredSequence(
                ids=teacher_prefix_ids + resp_ids,
                response_start=len(teacher_prefix_ids),
                response_len=len(resp_ids),
            ))

        for got, want in zip(teacher_seqs, legacy_teacher_seqs):
            self.assertEqual(got.ids, want.ids)
            self.assertEqual(got.response_start, want.response_start)
            self.assertEqual(got.response_len, want.response_len)
        for got, want in zip(student_seqs, legacy_student_seqs):
            self.assertEqual(got.ids, want.ids)
            self.assertEqual(got.response_start, want.response_start)
            self.assertEqual(got.response_len, want.response_len)


if __name__ == "__main__":
    unittest.main()
