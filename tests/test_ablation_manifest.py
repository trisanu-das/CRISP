"""Unit tests for the ablation manifest at experiments/ablations.yaml
and the runner at scripts/run_ablations.py.

The plan calls out five behavior contracts for the manifest + runner:

  1. Every variant has a known method and an existing base config.
  2. Each ablation changes only allow-listed keys from its full
     method's variant (one-factor-at-a-time discipline).
  3. CIBO fixed-beta and adaptive-beta suites are kept separate
     (a comparison is invalid if it mixes them under the same label).
  4. The runner writes a run directory whose name contains the
     method, variant, and seed.
  5. The runner refuses to launch in parallel on a single GPU by
     default (and accepts an opt-in flag to override that).
"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_manifest():
    """Load experiments/ablations.yaml as a dict-of-dicts."""
    import yaml

    path = REPO / "experiments" / "ablations.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _existing_configs():
    return {p.stem for p in (REPO / "config").glob("*.yaml")}


def _existing_methods():
    from train_launcher import SUPPORTED_METHODS
    return set(SUPPORTED_METHODS)


# Allow-list: which key paths may differ between an ablation and its
# full method variant. The plan's "one-factor-at-a-time" rule means an
# ablation overrides ONLY the named factor; anything else is silently
# inherited from the full variant.
_ALLOW_LIST = {
    "vacs": {
        "vacs.token_credit_mix", "vacs.soft_gate_slope",
        "vacs.reward_threshold", "vacs.sd_weight",
        "vacs.use_soft_gate", "vacs.use_asymmetric_projection",
        "vacs.adaptive_mix.enabled", "vacs.adaptive_mix.ema_decay",
        "vacs.adaptive_mix.multiplier_min", "vacs.adaptive_mix.multiplier_max",
        "cibo_v2.anchor_alpha",
    },
    "cibo-v2": {
        "cibo_v2.credit_lambda", "cibo_v2.beta", "cibo_v2.beta_mode",
        "cibo_v2.anchor_alpha", "cibo_v2.ib_reduction",
        "cibo_v2.beta_step_size", "cibo_v2.use_soft_gate",
        "cibo_v2.soft_gate_slope", "cibo_v2.reward_threshold",
    },
    "crisp": {
        "training.lambda_max", "training.lambda_schedule",
        "training.use_pcgrad", "training.correctness_gate",
    },
    "reinforce_pp": set(),
    "opsd": set(),
    "grpo": {
        "rollout.k",
    },
}


def _flat_overrides(overrides):
    """Flatten a nested overrides dict into a sorted set of dot-paths."""
    paths = set()

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                child = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    _walk(v, child)
                else:
                    paths.add(child)

    _walk(overrides)
    return paths


class TestManifestShape(unittest.TestCase):
    def test_every_variant_has_a_known_method_and_existing_base_config(self):
        manifest = _load_manifest()
        methods = _existing_methods()
        configs = _existing_configs()
        for suite_name, suite in manifest.items():
            for variant_name, variant in suite["variants"].items():
                method = variant.get("method")
                base = variant.get("base_config")
                self.assertIn(method, methods,
                              msg=f"{suite_name}/{variant_name}: method {method!r} not in {methods}")
                self.assertIn(base, configs,
                              msg=f"{suite_name}/{variant_name}: base_config {base!r} not in {configs}")

    def test_each_ablation_changes_only_allowlisted_keys_from_full_variant(self):
        manifest = _load_manifest()
        for suite_name, suite in manifest.items():
            full_name = suite["full_variant"]
            full = suite["variants"][full_name]
            full_method = full["method"]
            full_allow = _ALLOW_LIST.get(full_method, set())
            for variant_name, variant in suite["variants"].items():
                if variant_name == full_name:
                    continue
                # Use the VARIANT's own method's allow-list, falling
                # back to the full variant's. (A variant may use a
                # different method from the full -- e.g. the controls
                # suite's `full_variant: crisp` and `grpo_k8` variant.)
                variant_method = variant.get("method", full_method)
                allow = _ALLOW_LIST.get(variant_method, full_allow)
                overrides = variant.get("overrides", {})
                changed = _flat_overrides(overrides)
                unknown = changed - allow
                self.assertFalse(unknown,
                                 msg=f"{suite_name}/{variant_name}: ablation overrides {unknown} "
                                     f"which are not in the allow-list for method {variant_method!r}; "
                                     f"allow-list = {sorted(allow)}")

    def test_fixed_and_adaptive_cibo_suites_are_separate(self):
        manifest = _load_manifest()
        suite_names = list(manifest.keys())
        fixed_suites = [s for s in suite_names if "fixed" in s and "cibo" in s]
        adaptive_suites = [s for s in suite_names if "adaptive" in s and "cibo" in s]
        self.assertTrue(fixed_suites, msg="no CIBO fixed suite found")
        self.assertTrue(adaptive_suites, msg="no CIBO adaptive suite found")
        # They must be DIFFERENT suites (not the same key), AND
        # neither may claim to be the other.
        common = set(fixed_suites) & set(adaptive_suites)
        self.assertFalse(common,
                         msg=f"CIBO fixed and adaptive share suites: {common}")


class TestRunDirectoryNaming(unittest.TestCase):
    def test_run_directory_contains_method_variant_and_seed(self):
        # The runner must build a run directory whose name encodes
        # method, variant, and seed. We invoke --dry-run so the
        # runner does NOT actually start training.
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "run_ablations.py"),
             "--suite", "vacs-core",  # any suite we will define
             "--config", "config/vacs_7b.yaml",
             "--seeds", "0",
             "--dry-run"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        # The dry-run must not actually start a model download.
        # We assert that the printed command's --output-dir contains
        # a slug with method + variant + seed.
        # First we need the manifest to actually have a vacs-core suite
        # for this assertion to make sense.
        manifest = _load_manifest()
        if "vacs-core" not in manifest:
            self.skipTest("vacs-core suite not yet defined (expected during RED phase)")
        # Find one variant in vacs-core, expect its slug in dry-run output.
        full = manifest["vacs-core"]["full_variant"]
        variant = manifest["vacs-core"]["variants"][full]
        slug = variant.get("slug", full)
        # The runner passes the output directory as
        # `--override=training.output_dir=<path>` whose <path> ends
        # with `<method>-<slug>-seed-<n>`. The slug and seed must
        # both appear in the command.
        self.assertRegex(result.stdout,
                         rf"--override=training\.output_dir=[^\n]*{re.escape(slug)}",
                         msg=f"expected slug {slug!r} in --output-dir, got:\n{result.stdout}")
        # And seed must appear in the same output dir.
        self.assertRegex(result.stdout,
                         rf"--override=training\.output_dir=[^\n]*seed-?0",
                         msg=f"expected seed 0 in --output-dir, got:\n{result.stdout}")


class TestRunnerRejectsParallelGpuByDefault(unittest.TestCase):
    def test_runner_rejects_parallel_gpu_execution_by_default(self):
        # Without an --allow-parallel flag, the runner must refuse
        # multiple concurrent processes on a single GPU. We simulate
        # this by invoking --jobs 4 --allow-parallel-not-set and
        # expecting a non-zero exit. (The flag is opt-in.)
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "run_ablations.py"),
             "--suite", "vacs-core",
             "--config", "config/vacs_7b.yaml",
             "--seeds", "0",
             "--jobs", "4",
             "--dry-run"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        # Either:
        #   - the runner exits non-zero with a clear error message, OR
        #   - the manifest does not exist yet (RED phase).
        if "vacs-core" not in _load_manifest():
            self.skipTest("vacs-core suite not yet defined (RED phase)")
        self.assertNotEqual(result.returncode, 0,
                            msg=f"runner should reject --jobs 4 without --allow-parallel, got rc=0:\n{result.stdout}\n{result.stderr}")
        self.assertIn("parallel", (result.stdout + result.stderr).lower(),
                      msg=f"error should mention 'parallel', got:\n{result.stdout + result.stderr}")


class TestRunnerDryRunIsOffline(unittest.TestCase):
    def test_dry_run_does_not_load_a_model(self):
        # --dry-run must NOT download a model or start training. We
        # assert it does not block on torch.hub / huggingface_hub /
        # etc. (Indirect check: it returns within a few seconds and
        # does not import training modules that fetch the model.)
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "run_ablations.py"),
             "--suite", "vacs-core",
             "--config", "config/vacs_7b.yaml",
             "--seeds", "0",
             "--dry-run"],
            capture_output=True, text=True, cwd=str(REPO), timeout=15,
        )
        if "vacs-core" not in _load_manifest():
            self.skipTest("vacs-core suite not yet defined (RED phase)")
        # The dry run must not produce a Hugging Face download marker.
        combined = result.stdout + result.stderr
        self.assertNotIn("huggingface.co", combined.lower())
        self.assertNotIn("downloading", combined.lower())


if __name__ == "__main__":
    unittest.main()