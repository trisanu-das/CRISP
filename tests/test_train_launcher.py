from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_launcher import SUPPORTED_METHODS, build_parser, parse_overrides  # noqa: E402


class TestParseOverrides(unittest.TestCase):
    def test_empty_list_gives_empty_dict(self):
        self.assertEqual(parse_overrides([]), {})

    def test_simple_scalar(self):
        self.assertEqual(parse_overrides(["seed=5"]), {"seed": 5})

    def test_nested_key_path(self):
        self.assertEqual(parse_overrides(["training.lambda_max=0.5"]), {"training": {"lambda_max": 0.5}})

    def test_float_parsed_as_float_not_string(self):
        result = parse_overrides(["optimizer.lr=1.0e-6"])
        self.assertIsInstance(result["optimizer"]["lr"], float)

    def test_bool_parsed_as_bool(self):
        result = parse_overrides(["lora.enabled=false"])
        self.assertIs(result["lora"]["enabled"], False)

    def test_string_value_stays_string(self):
        result = parse_overrides(["logging.run_name=lambda_0.5"])
        self.assertEqual(result["logging"]["run_name"], "lambda_0.5")

    def test_list_value(self):
        result = parse_overrides(["lora.target_modules=[q_proj, v_proj]"])
        self.assertEqual(result["lora"]["target_modules"], ["q_proj", "v_proj"])

    def test_multiple_overrides_merge_into_one_dict(self):
        result = parse_overrides(["training.lambda_max=0.5", "training.total_steps=10", "seed=1"])
        self.assertEqual(result, {"training": {"lambda_max": 0.5, "total_steps": 10}, "seed": 1})

    def test_missing_equals_sign_raises(self):
        with self.assertRaises(ValueError):
            parse_overrides(["training.lambda_max"])

    def test_deeply_nested_path(self):
        result = parse_overrides(["a.b.c.d=1"])
        self.assertEqual(result, {"a": {"b": {"c": {"d": 1}}}})


class TestMethodChoices(unittest.TestCase):
    def test_vacs_and_cibo_v2_are_advertised_without_importing_method_modules(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["vacs", "--config", "config/vacs_7b.yaml"]).method, "vacs")
        self.assertEqual(
            parser.parse_args(["cibo-v2", "--config", "config/cibo_v2_7b.yaml"]).method,
            "cibo-v2",
        )
        self.assertIn("vacs", SUPPORTED_METHODS)
        self.assertIn("cibo-v2", SUPPORTED_METHODS)


if __name__ == "__main__":
    unittest.main()
