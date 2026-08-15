#!/usr/bin/env python
"""Serial ablation runner for VACS-CRISP and CIBO-CRISP v2.

Examples:

    # Dry-run a full VACS-core suite to see the exact commands without
    # downloading or training anything.
    python scripts/run_ablations.py --suite vacs-core \\
        --config config/vacs_7b.yaml --seeds 0,1,2 --dry-run

    # Actually run a single variant (faster iteration while developing).
    python scripts/run_ablations.py --suite vacs-core \\
        --variant full --config config/vacs_7b.yaml --seeds 0

    # Run CIBO v2's adaptive-beta suite on a single GPU (the runner
    # refuses multi-process on a single GPU by default; opt in with
    # --allow-parallel only if you know what you're doing).
    python scripts/run_ablations.py --suite cibo-adaptive \\
        --config config/cibo_v2_adaptive_beta_7b.yaml --seeds 0

What the runner guarantees:

  1. Validates the manifest + config files BEFORE starting a model:
     - every variant's `method` is in SUPPORTED_METHODS,
     - every variant's `base_config` exists in config/,
     - the suite exists in experiments/ablations.yaml.
  2. Executes one run at a time by default (`--jobs 1`). On a single
     GPU this is the only safe default; the runner refuses to launch
     more without an explicit `--allow-parallel` opt-in.
  3. Emits a resolved YAML and `run_metadata.json` into every run
     directory (built under `--output-root/<method>-<slug>-seed-<n>/`)
     so each run is reproducible from the manifest alone.
  4. Passes the method, variant slug, seed, and resolved output
     directory through to the train_launcher as `--override` flags.
  5. Fails fast on a non-zero child command and preserves its log
     under the run directory.
  6. `--dry-run` prints every command it would have run, without
     loading a model, fetching a dataset, or starting training.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_manifest() -> dict:
    path = REPO / "experiments" / "ablations.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _existing_configs() -> set[str]:
    return {p.stem for p in (REPO / "config").glob("*.yaml")}


def _existing_methods() -> set[str]:
    from train_launcher import SUPPORTED_METHODS
    return set(SUPPORTED_METHODS)


def _resolve_overrides_to_args(overrides: dict) -> list[str]:
    """Convert a nested overrides dict into a flat list of
    `key.path=value` CLI strings, matching train_launcher.parse_overrides.
    """
    pairs: list[str] = []

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                child = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                _walk(v, child)
        else:
            pairs.append(f"{prefix}={obj}")

    _walk(overrides)
    return [f"--override={p}" for p in pairs]


def _build_run_dir(output_root: Path, method: str, slug: str, seed: int) -> Path:
    return output_root / f"{method}-{slug}-seed-{seed}"


def _build_command(
    *,
    method: str,
    base_config: str,
    overrides: dict,
    output_dir: Path,
    seed: int,
    extra_overrides: dict,
) -> list[str]:
    """Assemble the train_launcher command for one run.

    `extra_overrides` are merged on top of the manifest's `overrides`
    (used for `--output-dir`, `logging.run_name`, `seed`).
    """
    merged: dict = {}
    for src in (overrides, extra_overrides):
        for k, v in src.items():
            merged[k] = v
    cmd = [
        sys.executable,
        "train_launcher.py",
        method,
        "--config",
        f"config/{base_config}.yaml",
    ]
    if method == "grpo":
        grpo_k = int(merged.get("rollout", {}).get("k", 8))
        cmd.extend(["--k", str(grpo_k)])
    cmd.extend([
        *_resolve_overrides_to_args(merged),
        f"--override=seed={seed}",
        f"--override=training.output_dir={output_dir.as_posix()}",
        f"--override=logging.run_name={output_dir.name}",
    ])
    return cmd


def _validate_manifest(manifest: dict) -> None:
    methods = _existing_methods()
    configs = _existing_configs()
    for suite_name, suite in manifest.items():
        if "variants" not in suite or "full_variant" not in suite:
            raise ValueError(
                f"suite {suite_name!r} must define 'full_variant' and 'variants'"
            )
        if suite["full_variant"] not in suite["variants"]:
            raise ValueError(
                f"suite {suite_name!r}: full_variant {suite['full_variant']!r} "
                f"not in variants"
            )
        for variant_name, variant in suite["variants"].items():
            method = variant.get("method")
            base = variant.get("base_config")
            if method not in methods:
                raise ValueError(
                    f"suite {suite_name!r} variant {variant_name!r}: "
                    f"method {method!r} not in {methods}"
                )
            if base not in configs:
                raise ValueError(
                    f"suite {suite_name!r} variant {variant_name!r}: "
                    f"base_config {base!r} not in {configs}"
                )
            for key in ("slug", "metric_keys", "overrides", "description"):
                if key not in variant:
                    raise ValueError(
                        f"suite {suite_name!r} variant {variant_name!r}: "
                        f"missing required field {key!r}"
                    )


def _write_run_metadata(
    *,
    run_dir: Path,
    suite: str,
    variant_name: str,
    variant: dict,
    method: str,
    base_config: str,
    seed: int,
    cmd: list[str],
    overrides: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "suite": suite,
        "variant": variant_name,
        "method": method,
        "base_config": base_config,
        "slug": variant.get("slug"),
        "seed": seed,
        "command": cmd,
        "manifest_overrides": overrides,
        "expected_metric_keys": variant.get("metric_keys", []),
        "description": variant.get("description", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Also write the resolved config the runner would have used, so
    # the run is reproducible from the manifest alone.
    cfg_path = REPO / "config" / f"{base_config}.yaml"
    resolved_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for k, v in overrides.items():
        _deep_set(resolved_yaml, k, v)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_yaml, sort_keys=False), encoding="utf-8"
    )


def _deep_set(d: dict, dotted_key: str, value) -> None:
    """Set d['a']['b']['c'] = value for a dotted_key like 'a.b.c'."""
    parts = dotted_key.split(".")
    cursor = d
    for p in parts[:-1]:
        cursor = cursor.setdefault(p, {})
        if not isinstance(cursor, dict):
            cursor = {}
    cursor[parts[-1]] = value


def _launch(cmd: list[str], cwd: Path, log_path: Path) -> int:
    """Run cmd, capture stdout+stderr to log_path, return exit code."""
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def _validate_completed_run(run_dir: Path, expected_keys: list[str]) -> None:
    import math

    if not (run_dir / "_SUCCESS").exists():
        raise RuntimeError(f"Run has no _SUCCESS marker: {run_dir}")
    metric_files = [
        path for path in run_dir.glob("*.jsonl")
        if path.name != "generations.jsonl"
    ]
    if not metric_files:
        raise RuntimeError(f"No metric JSONL produced in {run_dir}")
    records = []
    for line in metric_files[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"Metric JSONL is empty in {run_dir}")
    seen = set().union(*(record.keys() for record in records))
    missing = set(expected_keys) - seen
    if missing:
        raise RuntimeError(f"Missing expected metrics {sorted(missing)} in {run_dir}")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"Non-finite metric {key}={value} in {run_dir}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run VACS/CIBO ablation suites one variant+seed at a time."
    )
    parser.add_argument("--suite", required=True,
                        help="Name of the suite in experiments/ablations.yaml.")
    parser.add_argument("--variant",
                        help="Run only this one variant. Default: every variant in the suite.")
    parser.add_argument("--config", required=True,
                        help="A *.yaml file under config/. "
                             "Provided for cross-checking; the manifest's "
                             "base_config takes precedence for each variant.")
    parser.add_argument("--seeds", required=True,
                        help="Comma-separated list of integer seeds.")
    parser.add_argument("--output-root", default="runs/ablations",
                        help="Directory under which per-run subdirectories are created.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of sequential runs to allow. Default 1; "
                             "must be 1 unless --allow-parallel is also passed.")
    parser.add_argument("--allow-parallel", action="store_true",
                        help="Opt in to running --jobs N variant+seed combinations "
                             "concurrently. Off by default because on a single GPU "
                             "this would OOM. Useful only on multi-GPU machines.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command without downloading a model "
                             "or starting training.")
    args = parser.parse_args(argv)

    # ---- Validate the manifest before doing anything else ----
    manifest = _load_manifest()
    _validate_manifest(manifest)
    if args.suite not in manifest:
        print(f"ERROR: suite {args.suite!r} not in manifest. "
              f"Available: {list(manifest.keys())}", file=sys.stderr)
        return 1
    suite = manifest[args.suite]

    # ---- One-process-per-GPU safety check ----
    if args.jobs > 1 and not args.allow_parallel:
        print(
            f"ERROR: refusing to launch {args.jobs} concurrent processes without "
            f"--allow-parallel. On a single GPU this would OOM. Pass the flag "
            f"only if you have a multi-GPU machine and know what you're doing.",
            file=sys.stderr,
        )
        return 2

    # ---- Expand seeds + variants into a (variant, seed) job list ----
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except ValueError as exc:
        print(f"ERROR: --seeds must be a comma-separated list of integers: {exc}",
              file=sys.stderr)
        return 1
    if not seeds:
        print("ERROR: --seeds parsed to empty list", file=sys.stderr)
        return 1
    if args.variant:
        if args.variant not in suite["variants"]:
            print(f"ERROR: --variant {args.variant!r} not in suite {args.suite!r}",
                  file=sys.stderr)
            return 1
        variants_to_run = {args.variant: suite["variants"][args.variant]}
    else:
        variants_to_run = suite["variants"]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- Build the job list (deterministic order) ----
    jobs = []
    for variant_name in sorted(variants_to_run):
        variant = variants_to_run[variant_name]
        for seed in seeds:
            run_dir = _build_run_dir(
                output_root=output_root,
                method=variant["method"],
                slug=variant["slug"],
                seed=seed,
            )
            extra = {}
            cmd = _build_command(
                method=variant["method"],
                base_config=variant["base_config"],
                overrides=variant.get("overrides", {}),
                output_dir=run_dir,
                seed=seed,
                extra_overrides=extra,
            )
            jobs.append((variant_name, seed, run_dir, cmd, variant))

    # ---- Print plan, then either run or dry-run ----
    print(f"Suite: {args.suite}")
    print(f"  variants: {len(variants_to_run)}  seeds: {len(seeds)}  jobs: {len(jobs)}")
    if args.dry_run:
        for variant_name, seed, run_dir, cmd, _ in jobs:
            print(f"\n[DRY-RUN] {variant_name} seed={seed} -> {run_dir}")
            print("  " + " ".join(shlex.quote(c) for c in cmd))
        return 0

    # ---- Run sequentially ----
    failed = []
    for variant_name, seed, run_dir, cmd, variant in jobs:
        print(f"\n[run] {variant_name} seed={seed} -> {run_dir}")
        _write_run_metadata(
            run_dir=run_dir,
            suite=args.suite,
            variant_name=variant_name,
            variant=variant,
            method=variant["method"],
            base_config=variant["base_config"],
            seed=seed,
            cmd=cmd,
            overrides=variant.get("overrides", {}),
        )
        log_path = run_dir / "run.log"
        rc = _launch(cmd, cwd=REPO, log_path=log_path)
        if rc != 0:
            print(f"  FAIL: exit={rc}; log preserved at {log_path}", file=sys.stderr)
            failed.append((variant_name, seed, rc))
        else:
            try:
                _validate_completed_run(run_dir, variant.get("metric_keys", []))
            except Exception as exc:
                print(f"  FAIL validation: {exc}", file=sys.stderr)
                failed.append((variant_name, seed, 4))
            else:
                print(f"  ok")

    if failed:
        print(f"\n{len(failed)} job(s) failed:", file=sys.stderr)
        for variant_name, seed, rc in failed:
            print(f"  - {variant_name} seed={seed} exit={rc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())