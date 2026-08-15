from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.training.eos_utils import first_eos_cutoff, resolve_eos_token_ids  # noqa: E402


class TestResolveEosTokenIds(unittest.TestCase):
    def test_single_int_generation_config_eos(self):
        self.assertEqual(resolve_eos_token_ids(151645, None), [151645])

    def test_list_generation_config_eos(self):
        # The exact Qwen2.5 case: generation_config.json lists both im_end
        # and endoftext as valid stops.
        self.assertEqual(resolve_eos_token_ids([151645, 151643], None), [151643, 151645])

    def test_tokenizer_eos_included_even_if_not_in_generation_config(self):
        result = resolve_eos_token_ids([151643], 151645)
        self.assertEqual(set(result), {151643, 151645})

    def test_deduplicates_overlapping_ids(self):
        result = resolve_eos_token_ids([151645], 151645)
        self.assertEqual(result, [151645])

    def test_generation_config_none_falls_back_to_tokenizer_only(self):
        self.assertEqual(resolve_eos_token_ids(None, 151645), [151645])

    def test_both_none_raises_rather_than_silently_never_stopping(self):
        with self.assertRaises(ValueError):
            resolve_eos_token_ids(None, None)

    def test_result_is_sorted(self):
        result = resolve_eos_token_ids([151645, 100, 151643], 50)
        self.assertEqual(result, sorted(result))

    def test_qwen2_5_0_5b_instruct_real_values(self):
        # Confirmed values for this exact model/tokenizer: tokenizer.eos_token_id
        # is 151645 ("<|im_end|>"); this is the pilot config's default model.
        self.assertEqual(resolve_eos_token_ids(None, 151645), [151645])


class TestFirstEosCutoff(unittest.TestCase):
    def test_finds_single_eos(self):
        self.assertEqual(first_eos_cutoff([1, 2, 3, 151645, 5, 6], {151645}), 4)

    def test_finds_whichever_eos_appears_first_among_multiple_valid_ids(self):
        # 151643 appears before 151645 here -- must cut at the FIRST valid
        # stop token encountered, not prefer one id over another.
        self.assertEqual(first_eos_cutoff([1, 151643, 3, 151645], {151643, 151645}), 2)

    def test_no_eos_present_returns_none(self):
        self.assertIsNone(first_eos_cutoff([1, 2, 3, 4], {151645, 151643}))

    def test_eos_at_very_first_position(self):
        self.assertEqual(first_eos_cutoff([151645, 2, 3], {151645}), 1)

    def test_empty_token_list_returns_none(self):
        self.assertIsNone(first_eos_cutoff([], {151645}))

    def test_regression_max_new_tokens_worth_of_tokens_with_no_eos(self):
        # Simulates exactly what the bug report showed: 128 tokens, none of
        # which are a recognized eos id -> must return None (truncate to the
        # full budget), not silently invent a cutoff.
        tokens = list(range(1000, 1128))  # 128 tokens, none matching eos ids
        self.assertIsNone(first_eos_cutoff(tokens, {151645, 151643}))


if __name__ == "__main__":
    unittest.main()
