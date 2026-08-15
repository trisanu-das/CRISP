"""
Unit tests for the pure-Python parts of crisp/data/build_dataset.py:
schema normalization and answer extraction, exercised against synthetic
in-memory rows so no network access or `datasets` install is required.
The actual `load_dataset(...)` calls are exercised only implicitly (by
inspection / at real run time), not here.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.data.build_dataset import (  # noqa: E402
    extract_gsm8k_answer,
    extract_last_boxed,
    normalize_rows,
)


class TestExtractGsm8kAnswer(unittest.TestCase):
    def test_basic(self):
        raw = "Natalia sold 48 clips in April.\n#### 48"
        self.assertEqual(extract_gsm8k_answer(raw), "48")

    def test_strips_thousands_comma(self):
        raw = "Some reasoning.\n#### 1,234"
        self.assertEqual(extract_gsm8k_answer(raw), "1234")

    def test_no_marker_returns_none(self):
        self.assertIsNone(extract_gsm8k_answer("no marker in this string"))

    def test_uses_final_marker_if_multiple(self):
        raw = "step #### 1 more step #### 2"
        self.assertEqual(extract_gsm8k_answer(raw), "2")


class TestExtractLastBoxed(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(extract_last_boxed("solution... \\boxed{204}"), "204")

    def test_none_when_absent(self):
        self.assertIsNone(extract_last_boxed("no boxed answer"))


class TestNormalizeRows(unittest.TestCase):
    def test_basic_normalization(self):
        rows = [{"problem": "1+1=?", "answer": "2"}, {"problem": "2+2=?", "answer": "4"}]
        out = normalize_rows(rows, ["problem"], ["answer"], "fake_source")
        self.assertEqual(out, [{"prompt": "1+1=?", "answer": "2"}, {"prompt": "2+2=?", "answer": "4"}])

    def test_tries_candidates_in_order(self):
        rows = [{"Problem": "x", "Answer": "y"}]
        out = normalize_rows(rows, ["problem", "Problem"], ["answer", "Answer"], "fake_source")
        self.assertEqual(out, [{"prompt": "x", "answer": "y"}])

    def test_raises_with_actual_columns_when_no_match(self):
        rows = [{"totally_different_col": "x"}]
        with self.assertRaises(ValueError) as ctx:
            normalize_rows(rows, ["problem"], ["answer"], "fake_source")
        # The whole point of failing loud is that the message is actionable:
        # it must name the columns that *were* found.
        self.assertIn("totally_different_col", str(ctx.exception))

    def test_empty_rows_raises(self):
        with self.assertRaises(ValueError):
            normalize_rows([], ["problem"], ["answer"], "fake_source")

    def test_answer_transform_applied(self):
        rows = [{"problem": "x", "answer": "reasoning #### 42"}]
        out = normalize_rows(rows, ["problem"], ["answer"], "fake_source",
                              answer_transform=extract_gsm8k_answer)
        self.assertEqual(out, [{"prompt": "x", "answer": "42"}])

    def test_rows_where_transform_returns_none_are_skipped_not_crashed(self):
        rows = [
            {"problem": "has answer", "answer": "reasoning #### 1"},
            {"problem": "missing marker", "answer": "no marker here"},
        ]
        out = normalize_rows(rows, ["problem"], ["answer"], "fake_source",
                              answer_transform=extract_gsm8k_answer)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["prompt"], "has answer")

    def test_all_transforms_failing_raises_rather_than_returning_empty(self):
        rows = [{"problem": "x", "answer": "no marker at all"}]
        with self.assertRaises(ValueError):
            normalize_rows(rows, ["problem"], ["answer"], "fake_source",
                            answer_transform=extract_gsm8k_answer)


if __name__ == "__main__":
    unittest.main()
