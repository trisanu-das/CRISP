from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.utils.config import load_config  # noqa: E402
from crisp.utils.validation import ConfigValidationError, validate_config  # noqa: E402
from train_launcher import validate_method  # noqa: E402


class TestMethodConfigurationValidation(unittest.TestCase):
    def _config_with(self, yaml_fragment: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text(yaml_fragment)
            cfg = load_config(str(path))
        return cfg

    def test_cibo_credit_lambda_rejects_values_outside_zero_one(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                cfg = self._config_with(f"cibo_v2:\n  credit_lambda: {value}\n")
                with self.assertRaisesRegex(ConfigValidationError, "credit_lambda"):
                    validate_config(cfg)

    def test_cibo_credit_lambda_accepts_closed_unit_interval(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                cfg = self._config_with(f"cibo_v2:\n  credit_lambda: {value}\n")
                validate_config(cfg)

    def test_beta_bounds_must_be_ordered_and_contain_beta(self):
        cfg = self._config_with(
            "cibo_v2:\n"
            "  beta: 0.10\n"
            "  beta_min: 1.0\n"
            "  beta_max: 0.5\n"
        )
        with self.assertRaisesRegex(ConfigValidationError, "beta_min"):
            validate_config(cfg)

        cfg = self._config_with(
            "cibo_v2:\n"
            "  beta: 11.0\n"
            "  beta_min: 0.001\n"
            "  beta_max: 10.0\n"
        )
        with self.assertRaisesRegex(ConfigValidationError, "beta"):
            validate_config(cfg)

    def test_vacs_multiplier_bounds_must_be_positive_and_ordered(self):
        invalid_configs = (
            "vacs:\n  adaptive_mix:\n    multiplier_min: 0.0\n",
            "vacs:\n  adaptive_mix:\n    multiplier_min: 2.0\n    multiplier_max: 1.0\n",
        )
        for fragment in invalid_configs:
            with self.subTest(fragment=fragment):
                cfg = self._config_with(fragment)
                with self.assertRaisesRegex(ConfigValidationError, "multiplier"):
                    validate_config(cfg)

    def test_valid_defaults_pass_validation(self):
        cfg = self._config_with("")
        validate_config(cfg)


class TestLauncherMethodValidation(unittest.TestCase):
    def test_unknown_method_raises_before_runner_or_model_loading(self):
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            validate_method("not-a-method")


if __name__ == "__main__":
    unittest.main()
