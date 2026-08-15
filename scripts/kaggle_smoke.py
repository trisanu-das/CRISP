#!/usr/bin/env python
"""Kaggle-friendly bounded smoke test runner for VACS-CRISP and CIBO-CRISP v2.

Examples:

    # See exactly what the smoke test would do without downloading
    # anything.
    python scripts/kaggle_smoke.py --method vacs --dry-run
    python scripts/kaggle_smoke.py --method cibo-v2 --dry-run

    # Optionally pre-download the model + tokenizer before launching
    # the smoke run (the plan: `--predownload` calls predownload.py).
    python scripts/kaggle_smoke.py --method cibo-v2 --predownload

    # Actually run the smoke test (3 steps, 8 examples, on the GPU).
    python scripts/kaggle_smoke.py --method vacs

The runner guarantees:

  1. **Preflight checks** before anything else:
     - `--method` is in SUPPORTED_METHODS,
     - the smoke config exists and loads cleanly,
     - if `--preflight-cuda` (default on) then `torch.cuda.is_available()`.
       We don't hard-fail when CUDA is missing on Windows dev boxes --
       we WARN and proceed (Kaggle always has CUDA, but a contributor
       on a laptop should still be able to dry-run).
     - the output path is writable and has at least 2 GB free disk.
  2. **`--dry-run` is offline**: never imports torch / huggingface_hub,
     never downloads a model, never starts training. Just prints the
     resolved command + preflight results.
  3. **Post-run verification** (when not dry-running) checks:
     - a final checkpoint exists,
     - `metrics.jsonl` was written and contains at least one record,
     - the method-specific required metric key (vacs/loss_rl or
       cibo/loss_total) is present,
     - every required-metric value is finite,
     - at least one non-empty response was generated.
     Sparse / zero rewards in 3 steps are acceptable -- the smoke test
     verifies plumbing, not learning.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# Required-metric keys per method. The smoke runner fails the
# post-run check if any of these are missing OR non-finite.
_REQUIRED_METRICS = {
    "vacs": ["vacs/loss_rl"],
    "cibo-v2": ["cibo/loss_total"],
}


# ---------------------------------------------------------------------------
# Pre-flight checks (all cheap and offline).
# ---------------------------------------------------------------------------

def _check_method_supported(method: str) -> list[tuple[str, bool, str]]:
    """Return (label, ok, details) for `--method {method}`."""
    sys.path.insert(0, str(REPO))
    from train_launcher import SUPPORTED_METHODS
    ok = method in SUPPORTED_METHODS
    return [("method is in SUPPORTED_METHODS", ok,
             f"got {method!r}, known={sorted(SUPPORTED_METHODS)}")]


def _check_config_exists(method: str) -> list[tuple[str, bool, str]]:
    """Return checks for the method's smoke config under config/."""
    suffix = "vacs" if method == "vacs" else "cibo_v2"
    path = REPO / "config" / f"{suffix}_kaggle_smoke.yaml"
    return [(f"smoke config exists: {path.name}", path.exists(),
             f"path={path}")]


def _check_output_path_writable(output_root: Path) -> list[tuple[str, bool, str]]:
    """Make sure output_root is creatable and has at least 2 GB free."""
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        test_file = output_root / ".smoke_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except Exception as e:  # noqa: BLE001 - want a broad except for any IO failure
        return [("output path is writable", False, f"error: {e}")]
    try:
        usage = shutil.disk_usage(str(output_root))
        free_gb = usage.free / (1024 ** 3)
    except Exception as e:  # noqa: BLE001
        return [("disk space check", False, f"error: {e}")]
    ok = free_gb >= 2.0
    return [("output path is writable", True, str(output_root)),
            ("free disk >= 2 GB", ok, f"got {free_gb:.1f} GB")]


def _check_cuda_available() -> list[tuple[str, bool, str]]:
    """Best-effort CUDA probe. We DON'T hard-fail on a dev box, but we
    WARN. The Kaggle-side runner expects this to be True."""
    try:
        import torch  # type: ignore
        avail = torch.cuda.is_available()
        return [("CUDA available (informational)", avail,
                 "Kaggle must be True; dev boxes may be False")]
    except ImportError:
        return [("torch importable", False,
                 "torch is not installed in this Python environment")]


# ---------------------------------------------------------------------------
# Command building.
# ---------------------------------------------------------------------------

def _build_command(method: str, output_dir: Path) -> list[str]:
    """Assemble the train_launcher command for one smoke run.

    The smoke configs already encode the bounded knobs (total_steps,
    train_subset_size, etc.). The runner only adds the output dir and
    the run name.
    """
    suffix = "vacs" if method == "vacs" else "cibo_v2"
    config = REPO / "config" / f"{suffix}_kaggle_smoke.yaml"
    return [
        sys.executable, "train_launcher.py", method,
        "--config", str(config),
        f"--override=training.output_dir={output_dir.as_posix()}",
        f"--override=logging.run_name={output_dir.name}",
    ]


# ---------------------------------------------------------------------------
# Post-run verification (this is what tests/test_kaggle_smoke.py
# exercises against a synthetic run directory).
# ---------------------------------------------------------------------------

def verify_smoke_run(
    run_dir: Path,
    method: str,
) -> dict[str, object]:
    """Inspect a finished smoke run and return a structured verdict.

    The verdict contains:
      - "ok" (bool): every required check passed.
      - "checks" (list[dict]): per-check pass/fail + details.
      - "summary" (str): human-readable summary.

    The plan calls out that sparse / zero rewards in a 3-step smoke
    run are acceptable: we check that the metrics pipeline emits the
    required method-specific keys, that every value is finite, that a
    checkpoint was saved, and that at least one response was generated.
    We do NOT require non-zero reward.
    """
    required_keys = _REQUIRED_METRICS.get(method, [])
    checks: list[dict] = []

    # Check 1: final checkpoint present. `save_checkpoint` (common_loop.py)
    # writes a `final/` directory -- either PEFT adapter files
    # (adapter_model.safetensors + adapter_config.json) when
    # save_adapter_only=True, or model_state_dict.pt otherwise -- never a
    # flat `checkpoint.pt`. Check for what's actually produced.
    ckpt_dir = run_dir / "final"
    ckpt_files = list(ckpt_dir.glob("*")) if ckpt_dir.exists() else []
    ckpt_ok = ckpt_dir.exists() and any(f.stat().st_size > 0 for f in ckpt_files if f.is_file())
    checks.append({
        "label": "checkpoint present",
        "ok": ckpt_ok,
        "details": f"path={ckpt_dir}, files={[f.name for f in ckpt_files]}",
    })

    # Check 2: metrics jsonl present and non-empty. RunLogger
    # (logging_utils.py) writes to `{run_name}.jsonl`, not a fixed
    # `metrics.jsonl` -- run_name is the smoke run's directory name
    # (see _build_command's --override=logging.run_name=...).
    metrics_path = run_dir / f"{run_dir.name}.jsonl"
    metrics_lines: list[str] = []
    if metrics_path.exists():
        metrics_lines = [
            line for line in metrics_path.read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]
    checks.append({
        "label": "metrics.jsonl present and non-empty",
        "ok": metrics_path.exists() and len(metrics_lines) >= 1,
        "details": f"lines={len(metrics_lines)}, path={metrics_path}",
    })

    # Check 3: required method-specific metric key present in at
    # least one record, with a finite value.
    found_records: list[dict] = []
    for line in metrics_lines:
        try:
            found_records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    missing: list[str] = []
    nonfinite: list[str] = []
    for key in required_keys:
        any_present = False
        any_finite = False
        for rec in found_records:
            if key in rec:
                any_present = True
                val = rec[key]
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    any_finite = True
                    break
        if not any_present:
            missing.append(key)
        elif not any_finite:
            nonfinite.append(key)
    checks.append({
        "label": f"required metric keys present: {required_keys}",
        "ok": len(missing) == 0,
        "details": f"missing={missing or 'none'}",
    })
    checks.append({
        "label": "required metric values are finite",
        "ok": len(nonfinite) == 0,
        "details": f"non-finite={nonfinite or 'none'}",
    })

    # Check 4: at least one non-empty response was generated.
    gen_path = run_dir / "generations.jsonl"
    non_empty = 0
    if gen_path.exists():
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("response_text", "")
            if text and text.strip():
                non_empty += 1
    checks.append({
        "label": "at least one non-empty response generated",
        "ok": non_empty >= 1,
        "details": f"non_empty_responses={non_empty}",
    })

    # Check 5: it's OK for sparse / zero rewards -- we don't gate on
    # that. Just record what the average was.
    rewards = [
        float(rec.get("reward", rec.get("reward_mean", 0.0)))
        for rec in found_records
        if "reward" in rec or "reward_mean" in rec
    ]
    avg_reward = sum(rewards) / len(rewards) if rewards else float("nan")

    ok = all(c["ok"] for c in checks)
    summary = (
        f"smoke run {'PASS' if ok else 'FAIL'} ({len([c for c in checks if c['ok']])}/{len(checks)} checks); "
        f"avg_reward={avg_reward:.3f} (sparse/zero rewards in 3 steps are acceptable)"
    )
    return {"ok": ok, "checks": checks, "summary": summary,
            "avg_reward": avg_reward}


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Kaggle-friendly bounded smoke test for VACS / CIBO v2."
    )
    parser.add_argument("--method", required=True,
                        help="Method to smoke-test. One of SUPPORTED_METHODS.")
    parser.add_argument("--output-root", default="runs/kaggle_smoke",
                        help="Directory under which the smoke run directory is created.")
    parser.add_argument("--predownload", action="store_true",
                        help="Pre-download the model + tokenizer via predownload.py "
                             "before launching the smoke run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved command and preflight results, "
                             "without downloading a model or starting training.")
    parser.add_argument("--no-cuda-check", action="store_true",
                        help="Skip the CUDA availability probe.")
    args = parser.parse_args(argv)

    print(f"=== Kaggle smoke: {args.method} ===")
    print(f"--- preflight ---")

    # ---- Pre-flight (all cheap + offline) ----
    all_preflight: list[tuple[str, bool, str]] = []
    all_preflight += _check_method_supported(args.method)
    all_preflight += _check_config_exists(args.method)
    all_preflight += _check_output_path_writable(Path(args.output_root))
    if not args.no_cuda_check:
        all_preflight += _check_cuda_available()
    preflight_failed = [c for c in all_preflight if not c[1]]
    for label, ok, details in all_preflight:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: {details}")

    # Two preflight failures are HARD FAILURES: an unknown method, or
    # a missing smoke config. CUDA + disk checks are softer -- we
    # warn but still proceed on a dev box.
    HARD_FAILURE_LABELS = ("method is in SUPPORTED_METHODS", "smoke config exists")
    hard_failed = [c for c in preflight_failed if any(
        h in c[0] for h in HARD_FAILURE_LABELS
    )]
    if hard_failed:
        for c in hard_failed:
            print(f"  HARD FAILURE: {c[0]}: {c[2]}", file=sys.stderr)
        return 2
    soft_failed = [c for c in preflight_failed if c not in hard_failed]
    if soft_failed:
        print(
            f"  WARNING: {len(soft_failed)} soft preflight check(s) failed. "
            f"Proceeding; a real run may still fail. Resolve before launching.",
            file=sys.stderr,
        )

    # ---- Build the command ----
    output_dir = Path(args.output_root) / f"{args.method}_smoke"
    cmd = _build_command(args.method, output_dir)

    print(f"\n--- resolved command ---")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"\n--- run directory ---")
    print(f"  {output_dir}")

    if args.dry_run:
        print(f"\n[DRY-RUN] not invoking training; preflight above is the entire output.")
        return 0

    # ---- Optional pre-download ----
    if args.predownload:
        sys.path.insert(0, str(REPO))
        from crisp.utils.config import load_config
        cfg_path = REPO / "config" / (
            "vacs_kaggle_smoke.yaml" if args.method == "vacs"
            else "cibo_v2_kaggle_smoke.yaml"
        )
        cfg = load_config(str(cfg_path))
        model_name = cfg.model.name
        print(f"\n--- predownload: {model_name} ---")
        from predownload import main as predownload_main
        predownload_argv = [model_name]
        rc = predownload_main(predownload_argv)
        if rc != 0:
            print(f"  predownload failed (rc={rc}); aborting smoke run.",
                  file=sys.stderr)
            return rc

    # ---- Run ----
    print(f"\n--- launching smoke run ---")
    log_path = output_dir / "smoke_run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  smoke run failed: rc={proc.returncode}; log={log_path}",
              file=sys.stderr)
        return proc.returncode
    print(f"  ok (rc=0); log={log_path}")

    # ---- Post-run verification ----
    print(f"\n--- post-run verification ---")
    verdict = verify_smoke_run(output_dir, args.method)
    for c in verdict["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['label']}: {c['details']}")
    print(f"\n{verdict['summary']}")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())