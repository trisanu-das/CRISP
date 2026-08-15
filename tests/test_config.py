from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.utils.config import DEFAULTS, DotDict, _deep_merge, load_config  # noqa: E402


class TestDotDict(unittest.TestCase):
    def test_attribute_access(self):
        d = DotDict({"a": 1, "b": {"c": 2}})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b.c, 2)

    def test_missing_key_raises_attribute_error_not_key_error(self):
        d = DotDict({"a": 1})
        with self.assertRaises(AttributeError):
            _ = d.nonexistent

    def test_list_of_dicts_is_wrapped_too(self):
        d = DotDict({"eval_datasets": [{"x": 1}, {"x": 2}]})
        self.assertEqual(d.eval_datasets[0].x, 1)
        self.assertEqual(d.eval_datasets[1].x, 2)

    def test_still_behaves_as_a_plain_dict(self):
        d = DotDict({"a": 1})
        self.assertEqual(d["a"], 1)
        self.assertIn("a", d)
        self.assertEqual(dict(d), {"a": 1})

    def test_key_colliding_with_dict_method_name_raises_loudly(self):
        # `d.items` would otherwise silently resolve to the *bound method*
        # dict.items instead of a config value -- see DotDict's docstring.
        # This must fail fast at construction time, not fail confusingly
        # somewhere downstream when someone calls `cfg.items` expecting data.
        with self.assertRaises(ValueError):
            DotDict({"items": "some_value"})
        with self.assertRaises(ValueError):
            DotDict({"keys": "some_value"})

    def test_nested_dict_with_colliding_key_raises_when_wrapped(self):
        d = DotDict({"outer": {"get": "some_value"}})
        with self.assertRaises(ValueError):
            _ = d.outer


class TestDeepMerge(unittest.TestCase):
    def test_override_replaces_scalar(self):
        base = {"a": 1, "b": 2}
        out = _deep_merge(base, {"a": 99})
        self.assertEqual(out, {"a": 99, "b": 2})

    def test_nested_dicts_merge_recursively(self):
        base = {"model": {"name": "x", "dtype": "bf16"}}
        out = _deep_merge(base, {"model": {"name": "y"}})
        self.assertEqual(out, {"model": {"name": "y", "dtype": "bf16"}})

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        self.assertEqual(base, {"a": {"b": 1}})


class TestLoadConfig(unittest.TestCase):
    def test_partial_yaml_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("training:\n  total_steps: 5\n")
            cfg = load_config(str(path))
            self.assertEqual(cfg.training.total_steps, 5)
            # Untouched defaults should still be present and reachable.
            self.assertEqual(cfg.training.micro_batch_size, DEFAULTS["training"]["micro_batch_size"])
            self.assertEqual(cfg.model.name, DEFAULTS["model"]["name"])

    def test_overrides_param_applied_on_top_of_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("training:\n  total_steps: 5\n")
            cfg = load_config(str(path), overrides={"training": {"total_steps": 999}})
            self.assertEqual(cfg.training.total_steps, 999)

    def test_empty_yaml_file_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("")
            cfg = load_config(str(path))
            self.assertEqual(cfg.model.name, DEFAULTS["model"]["name"])

    def test_lora_target_modules_list_survives_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("lora:\n  target_modules: ['q_proj', 'v_proj']\n")
            cfg = load_config(str(path))
            self.assertEqual(list(cfg.lora.target_modules), ["q_proj", "v_proj"])

    def test_vacs_and_cibo_defaults_are_available_without_yaml_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text("")
            cfg = load_config(str(path))

        self.assertEqual(cfg.vacs.token_credit_mix, 0.25)
        self.assertTrue(cfg.vacs.adaptive_mix.enabled)
        self.assertEqual(cfg.vacs.adaptive_mix.multiplier_min, 0.10)
        self.assertEqual(cfg.cibo_v2.beta_mode, "fixed")
        self.assertEqual(cfg.cibo_v2.ib_reduction, "sequence_sum")
        self.assertEqual(cfg.cibo_v2.anchor_storage, "same")

    def test_method_specific_defaults_are_deep_merged_with_partial_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text(
                "vacs:\n"
                "  adaptive_mix:\n"
                "    enabled: false\n"
                "cibo_v2:\n"
                "  beta: 0.5\n"
            )
            cfg = load_config(str(path))

        self.assertFalse(cfg.vacs.adaptive_mix.enabled)
        self.assertEqual(cfg.vacs.adaptive_mix.ema_decay, 0.95)
        self.assertEqual(cfg.vacs.adaptive_mix.multiplier_max, 5.0)
        self.assertEqual(cfg.cibo_v2.beta, 0.5)
        self.assertEqual(cfg.cibo_v2.beta_mode, "fixed")
        self.assertEqual(cfg.cibo_v2.anchor_ema_decay, 0.995)


class TestShippedConfigFiles(unittest.TestCase):
    """Sanity-checks the actual config/*.yaml files shipped in this repo,
    not just synthetic ones -- catches YAML syntax errors or typo'd keys in
    the real artifacts a user will actually run.
    """

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def test_pilot_config_loads_and_has_expected_shape(self):
        cfg = load_config(str(self._repo_root() / "config" / "crisp_pilot.yaml"))
        self.assertEqual(cfg.model.name, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIs(cfg.model.local_files_only, False)
        self.assertTrue(cfg.lora.enabled)
        self.assertGreater(cfg.training.total_steps, 0)

    def test_7b_config_loads_and_has_expected_shape(self):
        cfg = load_config(str(self._repo_root() / "config" / "crisp_7b.yaml"))
        self.assertEqual(cfg.model.name, "Qwen/Qwen2.5-Math-7B-Instruct")
        self.assertIs(cfg.model.local_files_only, False)
        self.assertIn("aime2024", cfg.data.eval_datasets)


if __name__ == "__main__":
    unittest.main()
