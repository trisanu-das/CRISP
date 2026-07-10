"""Dataset building utilities for CRISP.

This module formats MATH + AIME-style problems into a uniform schema:

    {
        "prompt": str,
        "answer": str,
        "source": str,
        "year": int | None,
        "split": str | None,
    }

Design goals:
- Keep the training loop simple: it only needs prompt/answer.
- Support MATH from HuggingFace directly.
- Support AIME from local JSON / JSONL / CSV files or directories of such files.
- Preserve metadata for evaluation and ablations.

AIME-specific note:
- AIME datasets are usually not provided as a single canonical HuggingFace dataset.
  This module is therefore source-agnostic and expects local files or preloaded
  records for AIME.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from datasets import Dataset, concatenate_datasets, load_dataset


# -----------------------------
# Prompt templates
# -----------------------------


STUDENT_PROMPT_TEMPLATE = (
    "Solve the following problem step by step.\n"
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)

TEACHER_PROMPT_TEMPLATE = (
    "Solve the following problem step by step.\n"
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "The correct answer is: {answer}\n\n"
    "Now write a complete step-by-step solution:"
)


# -----------------------------
# Normalization helpers
# -----------------------------


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    # Case-insensitive lookup: real-world datasets vary in column casing
    # (e.g. Maxwell-Jia/AIME_2024 uses "Problem"/"Answer", not lowercase).
    lower_map = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        actual_key = lower_map.get(key.lower())
        if actual_key is not None and row[actual_key] not in (None, ""):
            return row[actual_key]
    return None


def normalize_problem_answer_row(
    row: Mapping[str, Any],
    *,
    source: str,
    year: Optional[int] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a raw row into the CRISP prompt/answer schema."""
    problem = _first_present(row, ["problem", "question", "prompt", "input", "statement", "text"])
    answer = _first_present(row, ["answer", "solution", "output", "target", "final_answer", "label"])

    if problem is None or answer is None:
        raise KeyError(
            f"Row is missing a problem or answer field. Available keys: {sorted(row.keys())}"
        )

    row_out: Dict[str, Any] = {
        "prompt": _to_str(problem),
        "answer": _to_str(answer),
        "source": _to_str(source),
    }
    if year is not None:
        row_out["year"] = int(year)
    if split is not None:
        row_out["split"] = _to_str(split)
    return row_out


# -----------------------------
# MATH dataset
# -----------------------------


def load_math_dataset(
    *,
    split: str = "train",
    name: str = "hendrycks/competition_math",
    max_rows: Optional[int] = None,
) -> Dataset:
    """Load and normalize the MATH dataset from HuggingFace.

    The canonical competition_math dataset exposes fields like "problem" and
    "solution", which are mapped into the common schema.
    """
    ds = load_dataset(name, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    def _map(row: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = normalize_problem_answer_row(row, source="math", split=split)
        # Keep the original fields in case downstream users want them.
        normalized["raw_problem"] = _to_str(row.get("problem", row.get("question", "")))
        normalized["raw_answer"] = _to_str(row.get("solution", row.get("answer", "")))
        return normalized

    remove_columns = ds.column_names
    return ds.map(_map, remove_columns=remove_columns)


# -----------------------------
# AIME dataset loading
# -----------------------------


def _read_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Accept either a dict with a top-level records field or a single record.
        if "data" in data and isinstance(data["data"], list):
            return [dict(x) for x in data["data"]]
        return [dict(data)]
    if isinstance(data, list):
        return [dict(x) for x in data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(dict(json.loads(line)))
    return rows


def _read_csv_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_aime_records(source: str | Path | Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Load AIME-style records from files, directories, or in-memory rows.

    Accepted formats:
    - JSON array
    - JSONL
    - CSV
    - directory containing any mixture of the above
    - a list of already-loaded dicts

    Expected fields per row:
    - problem (or question/prompt/input)
    - answer (or solution/final_answer)
    Optional:
    - year
    - split
    - contest
    - id
    """
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, Path)):
        return [dict(x) for x in source]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"AIME source not found: {path}")

    files: List[Path]
    if path.is_dir():
        files = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in {".json", ".jsonl", ".csv"}]
    else:
        files = [path]

    rows: List[Dict[str, Any]] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            rows.extend(_read_json_file(file_path))
        elif suffix == ".jsonl":
            rows.extend(_read_jsonl_file(file_path))
        elif suffix == ".csv":
            rows.extend(_read_csv_file(file_path))
        else:
            raise ValueError(f"Unsupported AIME file format: {file_path}")
    return rows


def load_aime_dataset(
    source: str | Path | Sequence[Mapping[str, Any]],
    *,
    year: Optional[int] = None,
    split: Optional[str] = None,
    source_name: str = "aime",
) -> Dataset:
    """Load and normalize AIME problems from a local source.

    This is intentionally generic so it can serve AIME 2024, 2025, 2026,
    or any other year as long as the records are provided.
    """
    rows = load_aime_records(source)
    if not rows:
        raise ValueError("AIME source yielded no records")

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_year = year
        if row_year is None:
            maybe_year = row.get("year")
            if maybe_year not in (None, ""):
                try:
                    row_year = int(maybe_year)
                except Exception:
                    row_year = None

        row_split = split or _to_str(row.get("split", "")) or None
        normalized = normalize_problem_answer_row(
            row,
            source=source_name,
            year=row_year,
            split=row_split,
        )
        normalized["id"] = _to_str(row.get("id", row.get("problem_id", ""))) or None
        normalized["contest"] = _to_str(row.get("contest", row.get("benchmark", "AIME"))) or "AIME"
        normalized["raw_problem"] = normalized["prompt"]
        normalized["raw_answer"] = normalized["answer"]
        normalized_rows.append(normalized)

    return Dataset.from_list(normalized_rows)


# -----------------------------
# Combined builders
# -----------------------------


def build_prompt_answer_dataset(
    *,
    math_split: str = "train",
    math_name: str = "hendrycks/competition_math",
    math_max_rows: Optional[int] = None,
    aime_source: Optional[str | Path | Sequence[Mapping[str, Any]]] = None,
    aime_year: Optional[int] = None,
    aime_split: Optional[str] = None,
    include_math: bool = True,
    include_aime: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """Build a unified prompt/answer dataset from MATH and AIME.

    The output schema is the same for both sources, so it can be used directly
    by training/crisp_step.py and training/train.py.
    """
    parts: List[Dataset] = []

    if include_math:
        parts.append(
            load_math_dataset(
                split=math_split,
                name=math_name,
                max_rows=math_max_rows,
            )
        )

    if include_aime:
        if aime_source is None:
            raise ValueError("include_aime=True requires aime_source to be provided")
        parts.append(
            load_aime_dataset(
                aime_source,
                year=aime_year,
                split=aime_split,
                source_name="aime",
            )
        )

    if not parts:
        raise ValueError("At least one of include_math or include_aime must be True")

    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if shuffle:
        combined = combined.shuffle(seed=seed)
    return combined


# -----------------------------
# Formatting helpers
# -----------------------------


def format_student_prompt(problem: str) -> str:
    return STUDENT_PROMPT_TEMPLATE.format(problem=_to_str(problem))


def format_teacher_prompt(problem: str, answer: str) -> str:
    return TEACHER_PROMPT_TEMPLATE.format(problem=_to_str(problem), answer=_to_str(answer))


# -----------------------------
# Persistence helpers
# -----------------------------


def save_dataset_jsonl(dataset: Dataset, output_path: str | Path) -> Path:
    """Save a dataset as JSONL with one record per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return output_path


# -----------------------------
# CLI
# -----------------------------


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Build CRISP training/eval datasets")
    parser.add_argument("--output", type=str, required=True, help="Path to write JSONL output")
    parser.add_argument("--include-math", action="store_true", help="Include MATH")
    parser.add_argument("--include-aime", action="store_true", help="Include AIME")
    parser.add_argument("--math-split", type=str, default="train")
    parser.add_argument("--math-name", type=str, default="hendrycks/competition_math")
    parser.add_argument("--math-max-rows", type=int, default=None)
    parser.add_argument("--aime-source", type=str, default=None)
    parser.add_argument("--aime-year", type=int, default=None)
    parser.add_argument("--aime-split", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    ds = build_prompt_answer_dataset(
        math_split=args.math_split,
        math_name=args.math_name,
        math_max_rows=args.math_max_rows,
        aime_source=args.aime_source,
        aime_year=args.aime_year,
        aime_split=args.aime_split,
        include_math=args.include_math,
        include_aime=args.include_aime,
        seed=args.seed,
    )
    save_dataset_jsonl(ds, args.output)
    print(f"Saved {len(ds)} rows to {args.output}")


if __name__ == "__main__":
    main()
