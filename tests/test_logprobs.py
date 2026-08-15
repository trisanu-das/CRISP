"""
Unit tests for crisp/training/logprobs.py.

Requires torch (skipped automatically if it isn't installed). This module is
the single most bug-sensitive piece of the codebase -- an off-by-one here
silently corrupts the training signal rather than crashing -- so every test
below hand-derives its expected numbers independently (via plain Python
math.log / manual softmax) rather than re-deriving them with the same tensor
ops the production code uses, which would just check the code agrees with
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
    from crisp.training.logprobs import (
        ScoredSequence,
        build_scoring_batch,
        forward_kl_teacher_student,
        gather_token_log_probs,
        response_log_probs,
    )


def _manual_log_softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    z = sum(exps)
    return [math.log(e / z) for e in exps]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestResponseLogProbs(unittest.TestCase):
    def test_single_row_matches_hand_computed_log_softmax(self):
        # vocab size 4. context = [10, 11] (2 tokens), response = [2, 3] (2 tokens).
        # full ids: [10, 11, 2, 3] ; response_start=2, response_len=2.
        seq = ScoredSequence(ids=[10, 11, 2, 3], response_start=2, response_len=2)
        # logits[t] is the distribution used to predict input_ids[t+1].
        # We only need logits at shifted indices 1 and 2 (predicting
        # positions 2 and 3) to be meaningful; positions 0 is irrelevant.
        logits_row = [
            [0.0, 0.0, 0.0, 0.0],   # predicts position 1 (irrelevant to the response span)
            [1.0, 2.0, 0.5, -1.0],  # predicts position 2 (the first response token, id=2)
            [0.2, 0.1, 3.0, 0.4],   # predicts position 3 (the second response token, id=3)
            [0.0, 0.0, 0.0, 0.0],   # unused (last position has no "next token" to predict)
        ]
        logits = torch.tensor([logits_row])
        input_ids = torch.tensor([[10, 11, 2, 3]])

        [row_log_probs] = response_log_probs(logits, input_ids, [seq])

        self.assertEqual(row_log_probs.shape, (2, 4))
        expected_pos2 = _manual_log_softmax(logits_row[1])
        expected_pos3 = _manual_log_softmax(logits_row[2])
        for i in range(4):
            self.assertAlmostEqual(row_log_probs[0, i].item(), expected_pos2[i], places=5)
            self.assertAlmostEqual(row_log_probs[1, i].item(), expected_pos3[i], places=5)

    def test_two_rows_with_different_context_lengths(self):
        # Row 0: context length 1 (student, no hint). Row 1: context length 3
        # (teacher, with a hint) -- same 2-token response [5, 6] in both,
        # right-padded to a common length of 5.
        seq0 = ScoredSequence(ids=[9, 5, 6], response_start=1, response_len=2)
        seq1 = ScoredSequence(ids=[9, 100, 101, 5, 6], response_start=3, response_len=2)
        pad_id = 0
        input_ids, attention_mask = build_scoring_batch([seq0, seq1], pad_id)

        self.assertEqual(input_ids.shape, (2, 5))
        self.assertEqual(input_ids[0].tolist(), [9, 5, 6, 0, 0])
        self.assertEqual(attention_mask[0].tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(input_ids[1].tolist(), [9, 100, 101, 5, 6])
        self.assertEqual(attention_mask[1].tolist(), [1, 1, 1, 1, 1])

        # `vocab` must exceed every token id actually used as a *value*
        # anywhere in this test (context ids go up to 101; response ids are
        # 5/6) -- token ids and vocab size are independent quantities, and
        # conflating "keep the vocab small for a simple test" with "the
        # token id values can be anything" is exactly the bug this test
        # previously had: with vocab=3, `gather_token_log_probs` tried to
        # gather index 5/6 out of a size-3 last dimension and raised
        # `RuntimeError: index out of bounds` the moment this test was
        # actually executed against real torch (it wasn't, when first
        # written, in an environment without torch installed -- this is
        # exactly the class of bug that only running the test catches).
        vocab = 150
        # Arbitrary but distinct logits per position/row so a mix-up between
        # rows or positions would change the numbers. The "hot" index (0/1/2)
        # is unrelated to the response token id values (5/6) -- this is
        # deliberately exercising that those are two different axes.
        logits = torch.zeros(2, 5, vocab)
        logits[0, 0, 0] = 1.0   # predicts row0 pos1 (token id 5)
        logits[0, 1, 1] = 1.0   # predicts row0 pos2 (token id 6)
        logits[1, 2, 2] = 1.0   # predicts row1 pos3 (token id 5)
        logits[1, 3, 0] = 1.0   # predicts row1 pos4 (token id 6)
        logits[1, 3, 1] = 1.0

        row_log_probs = response_log_probs(logits, input_ids, [seq0, seq1])
        token_log_probs = gather_token_log_probs(row_log_probs, [seq0, seq1])

        self.assertEqual(row_log_probs[0].shape, (2, vocab))
        self.assertEqual(row_log_probs[1].shape, (2, vocab))

        # The point of this test: row 0 and row 1 have different
        # `response_start` (1 vs 3), so they read from different absolute
        # shift-index offsets. Verify each row's extracted slice matches
        # log_softmax of the *specific* raw logits row/position intended for
        # it, not e.g. the other row's offset or an off-by-one neighbor.
        def _one_hot(hot_indices: list[int]) -> list[float]:
            return [1.0 if i in hot_indices else 0.0 for i in range(vocab)]

        expected_row0_pos1 = _manual_log_softmax(_one_hot([0]))
        expected_row0_pos2 = _manual_log_softmax(_one_hot([1]))
        expected_row1_pos3 = _manual_log_softmax(_one_hot([2]))
        expected_row1_pos4 = _manual_log_softmax(_one_hot([0, 1]))
        for i in range(vocab):
            self.assertAlmostEqual(row_log_probs[0][0, i].item(), expected_row0_pos1[i], places=5)
            self.assertAlmostEqual(row_log_probs[0][1, i].item(), expected_row0_pos2[i], places=5)
            self.assertAlmostEqual(row_log_probs[1][0, i].item(), expected_row1_pos3[i], places=5)
            self.assertAlmostEqual(row_log_probs[1][1, i].item(), expected_row1_pos4[i], places=5)

        # response_ids are [5, 6] in both rows; gather_token_log_probs must
        # pick out the log-prob of THOSE ids (now safely < vocab), not e.g.
        # silently wrap/clip -- assert the actual values, not just shape,
        # so a future regression back to "gather ignores the real id" would
        # be caught here too.
        self.assertEqual(token_log_probs[0].shape, (2,))
        self.assertEqual(token_log_probs[1].shape, (2,))
        self.assertAlmostEqual(token_log_probs[0][0].item(), expected_row0_pos1[5], places=5)
        self.assertAlmostEqual(token_log_probs[0][1].item(), expected_row0_pos2[6], places=5)
        self.assertAlmostEqual(token_log_probs[1][0].item(), expected_row1_pos3[5], places=5)
        self.assertAlmostEqual(token_log_probs[1][1].item(), expected_row1_pos4[6], places=5)

    def test_response_start_zero_raises(self):
        seq = ScoredSequence(ids=[1, 2], response_start=0, response_len=2)
        logits = torch.zeros(1, 2, 4)
        input_ids = torch.tensor([[1, 2]])
        with self.assertRaises(ValueError):
            response_log_probs(logits, input_ids, [seq])

    def test_response_span_exceeding_length_raises(self):
        seq = ScoredSequence(ids=[1, 2, 3], response_start=2, response_len=5)
        logits = torch.zeros(1, 3, 4)
        input_ids = torch.tensor([[1, 2, 3]])
        with self.assertRaises(ValueError):
            response_log_probs(logits, input_ids, [seq])

    def test_mismatched_ids_raises_runtime_error(self):
        # `sequences[0].ids` claims the response is [2, 3], but `input_ids`
        # (as if the batch had been built inconsistently) actually has
        # different tokens there -- this must be caught, not silently scored
        # against the wrong tokens.
        seq = ScoredSequence(ids=[10, 11, 2, 3], response_start=2, response_len=2)
        logits = torch.zeros(1, 4, 4)
        corrupted_input_ids = torch.tensor([[10, 11, 999, 998]])
        with self.assertRaises(RuntimeError):
            response_log_probs(logits, corrupted_input_ids, [seq])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestForwardKLTeacherStudent(unittest.TestCase):
    def test_matches_hand_computed_kl(self):
        # One response token, vocab size 3.
        teacher_probs = [0.7, 0.2, 0.1]
        student_probs = [0.5, 0.3, 0.2]
        teacher_log_probs = torch.tensor([[math.log(p) for p in teacher_probs]])
        student_log_probs = torch.tensor([[math.log(p) for p in student_probs]])

        [kl] = forward_kl_teacher_student([teacher_log_probs], [student_log_probs])

        expected = sum(p * (math.log(p) - math.log(q)) for p, q in zip(teacher_probs, student_probs))
        self.assertAlmostEqual(kl.item(), expected, places=5)

    def test_identical_distributions_give_zero_kl(self):
        log_probs = torch.tensor([[math.log(0.5), math.log(0.5)]])
        [kl] = forward_kl_teacher_student([log_probs], [log_probs.clone()])
        self.assertAlmostEqual(kl.item(), 0.0, places=6)

    def test_averages_over_multiple_response_tokens(self):
        # Two response tokens with different per-token KL; result should be
        # their mean, not their sum (length-normalized, so it doesn't
        # penalize longer responses just for being longer).
        t = torch.tensor([
            [math.log(0.9), math.log(0.1)],
            [math.log(0.5), math.log(0.5)],
        ])
        s = torch.tensor([
            [math.log(0.5), math.log(0.5)],
            [math.log(0.5), math.log(0.5)],
        ])
        [kl] = forward_kl_teacher_student([t], [s])
        kl_tok0 = 0.9 * (math.log(0.9) - math.log(0.5)) + 0.1 * (math.log(0.1) - math.log(0.5))
        kl_tok1 = 0.0  # identical distributions
        self.assertAlmostEqual(kl.item(), (kl_tok0 + kl_tok1) / 2, places=5)

    def test_teacher_is_detached_from_graph(self):
        t = torch.tensor([[math.log(0.7), math.log(0.3)]], requires_grad=True)
        s = torch.tensor([[math.log(0.5), math.log(0.5)]], requires_grad=True)
        [kl] = forward_kl_teacher_student([t], [s])
        kl.backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(s.grad)

    def test_length_mismatch_raises(self):
        t = torch.tensor([[math.log(0.5), math.log(0.5)]])
        s = torch.tensor([
            [math.log(0.5), math.log(0.5)],
            [math.log(0.5), math.log(0.5)],
        ])
        with self.assertRaises(ValueError):
            forward_kl_teacher_student([t], [s])


if __name__ == "__main__":
    unittest.main()
