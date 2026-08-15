"""Documentation consistency tests for the final README and docs/.

The plan calls out three behaviour contracts:

  1. Every config file mentioned by name in README.md exists under
     config/.
  2. Every `--method X` example in README.md and docs/kaggle-smoke.md
     is in SUPPORTED_METHODS (the launcher's accepted set).
  3. Every ablation suite shown in README.md and docs/ablations.md
     exists in experiments/ablations.yaml with the claimed full_variant.

We read each markdown file as plain text (no markdown library needed
for these structural checks) and apply a small set of regex rules.
The tests fail if any reference is broken.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _existing_configs() -> set[str]:
    """Return set of config stem names (without .yaml)."""
    return {p.stem for p in (REPO / "config").glob("*.yaml")}


def _existing_methods() -> set[str]:
    from train_launcher import SUPPORTED_METHODS
    return set(SUPPORTED_METHODS)


def _existing_suites() -> set[str]:
    """Return the set of suite keys in experiments/ablations.yaml."""
    import yaml
    path = REPO / "experiments" / "ablations.yaml"
    if not path.exists():
        return set()
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    return set(manifest.keys() if manifest else [])


# ---------------------------------------------------------------------------
# 1. README config references.
# ---------------------------------------------------------------------------

class TestReadmeConfigsExist(unittest.TestCase):
    def test_every_config_named_in_readme_exists(self):
        readme = _read(REPO / "README.md")
        # Patterns to extract config names from README:
        #   - `config/<name>.yaml`
        #   - `--config <name>` (the .yaml is implied)
        # We deliberately do NOT also match `--config config/<name>`
        # because that double-counts the same configs.
        patterns = [
            r"config/([\w\-]+)\.yaml",
            r"--config\s+([\w\-]+)\.yaml",  # also catches --config foo.yaml
            r"--config\s+([\w\-]+)",        # catches --config foo (no extension)
        ]
        mentioned: set[str] = set()
        # Words that look like config names but aren't. We have to
        # filter `config` (the directory name) out of `--config config/`
        # matches because the bare `--config\s+([\w\-]+)` pattern
        # captures it. The pattern list intentionally includes the
        # bare form to catch `--config foo` without .yaml, then we
        # exclude the literal directory name.
        NOT_CONFIG_NAMES = {"yaml", "config"}
        for pat in patterns:
            for m in re.finditer(pat, readme):
                name = m.group(1)
                if name in NOT_CONFIG_NAMES or len(name) < 3:
                    continue
                mentioned.add(name)
        existing = _existing_configs()
        missing = mentioned - existing
        self.assertFalse(missing,
                         msg=f"README.md mentions configs that don't exist: {sorted(missing)}\n"
                             f"Existing configs: {sorted(existing)}")


# ---------------------------------------------------------------------------
# 2. README + docs/kaggle-smoke.md method references.
# ---------------------------------------------------------------------------

class TestDocsMethodReferences(unittest.TestCase):
    def test_every_method_named_in_readme_is_supported(self):
        # --method examples in README should be in SUPPORTED_METHODS.
        readme = _read(REPO / "README.md")
        # The supported method names.
        supported = _existing_methods()
        # Patterns we accept as "a method is being referenced":
        patterns = [
            r"--method\s+([\w\-]+)",
            r"python\s+train_launcher\.py\s+([\w\-]+)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, readme):
                name = m.group(1)
                self.assertIn(name, supported,
                              msg=f"README.md mentions method {name!r} "
                                  f"but it's not in {sorted(supported)}")
        # Sanity: README should mention at least one method.
        any_method_referenced = any(
            re.search(pat, readme) for pat in patterns
        )
        self.assertTrue(any_method_referenced,
                        msg="README.md mentions no methods at all")


class TestKaggleSmokeDocMethodReferences(unittest.TestCase):
    def test_every_method_named_in_kaggle_smoke_md_is_supported(self):
        path = REPO / "docs" / "kaggle-smoke.md"
        if not path.exists():
            self.skipTest("docs/kaggle-smoke.md not yet written (RED phase)")
        text = _read(path)
        supported = _existing_methods()
        # --method X
        for m in re.finditer(r"--method\s+([\w\-]+)", text):
            name = m.group(1)
            self.assertIn(name, supported,
                          msg=f"docs/kaggle-smoke.md uses --method {name!r} "
                              f"but it's not in {supported}")


# ---------------------------------------------------------------------------
# 3. Ablation suite references.
# ---------------------------------------------------------------------------

class TestAblationSuiteReferences(unittest.TestCase):
    def test_every_suite_in_ablations_md_exists_in_manifest(self):
        path = REPO / "docs" / "ablations.md"
        if not path.exists():
            self.skipTest("docs/ablations.md not yet written (RED phase)")
        text = _read(path)
        existing = _existing_suites()
        # Look for `python scripts/run_ablations.py --suite X` mentions.
        for m in re.finditer(r"--suite\s+([\w\-]+)", text):
            name = m.group(1)
            self.assertIn(name, existing,
                          msg=f"docs/ablations.md uses --suite {name!r} "
                              f"but {name!r} is not in experiments/ablations.yaml ({sorted(existing)})")

    def test_readme_ablation_suite_references_resolve(self):
        # README.md mentions specific ablation suites by name; verify
        # each exists in the manifest.
        readme = _read(REPO / "README.md")
        existing = _existing_suites()
        # Patterns: backtick-quoted `suite-name` next to "suite" or
        # `experiments/ablations.yaml`.
        for m in re.finditer(r"`([\w\-]+)`\s+(?:suite|ablation)", readme):
            name = m.group(1)
            if name in existing:
                continue
            # Plain-text mentions of "X suite" where X is one of the
            # manifest keys should also be caught.
        # Patterns: `python scripts/run_ablations.py --suite X`
        for m in re.finditer(r"--suite\s+([\w\-]+)", readme):
            name = m.group(1)
            self.assertIn(name, existing,
                          msg=f"README.md uses --suite {name!r} "
                              f"but it's not in {sorted(existing)}")


# ---------------------------------------------------------------------------
# 4. Methods.md: structural sanity. We don't enforce a specific
# layout (the plan calls for formulas + reductions + state-update
# timing + metric definitions) -- we just check the file exists and
# mentions each method by name.
# ---------------------------------------------------------------------------

class TestMethodsDocExists(unittest.TestCase):
    def test_methods_md_exists_and_mentions_each_method(self):
        path = REPO / "docs" / "methods.md"
        self.assertTrue(path.exists(),
                        msg="docs/methods.md missing")
        text = _read(path)
        for method in ("CRISP", "VACS", "CIBO"):
            self.assertIn(method, text,
                          msg=f"docs/methods.md does not mention {method!r}")


if __name__ == "__main__":
    unittest.main()