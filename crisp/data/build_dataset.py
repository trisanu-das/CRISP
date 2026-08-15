"""
Dataset loading + normalization into the schema every other module expects:

    {"prompt": <problem statement text>, "answer": <ground-truth final answer as text>}

Column names vary across sources (and drift over time as datasets get
updated upstream), so each loader tries a short list of plausible candidate
column names and fails LOUDLY, listing the columns it actually found, if
none match -- deliberately never silently falling back to an empty/placeholder
dataset. (The reference project's own changelog names exactly this failure
mode -- "AIME eval sources pointing at a placeholder with no loader" -- as a
bug it hit; `_normalize_rows` below is the fix.)

The normalization helpers (`_normalize_rows`, `_extract_gsm8k_answer`,
`_extract_last_boxed`) take plain Python dicts/strings and never touch the
network themselves, so they're covered directly in tests/test_build_dataset.py
without needing a real dataset download.
"""
from __future__ import annotations

from typing import Callable, Optional
import random


def _first_present(keys, candidates: list[str]) -> Optional[str]:
    key_set = set(keys)
    for c in candidates:
        if c in key_set:
            return c
    return None


def extract_gsm8k_answer(raw: str) -> Optional[str]:
    """GSM8K's `answer` field is a full worked solution ending in
    `#### <final number>`; only the final number is a useful reward target.
    """
    marker = "####"
    if marker not in raw:
        return None
    tail = raw.split(marker)[-1].strip()
    return tail.replace(",", "")


def extract_last_boxed(text: str) -> Optional[str]:
    """Same brace-matching boxed-answer extraction as crisp/data/reward.py,
    duplicated here (rather than imported) so dataset normalization has zero
    dependency on the reward module -- they're conceptually separate
    concerns (what's the gold answer vs. was the model's answer correct)
    that happen to share a small piece of parsing logic.
    """
    key = "\\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    depth = 1
    buf: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf) if depth == 0 else None


def normalize_rows(
    rows: list[dict],
    prompt_candidates: list[str],
    answer_candidates: list[str],
    source_name: str,
    answer_transform: Optional[Callable[[str], Optional[str]]] = None,
) -> list[dict]:
    """Turn a list of raw dataset rows (plain dicts) into the
    `{"prompt", "answer"}` schema. Raises ValueError with the actual column
    names found if neither candidate list matches -- callers should never
    catch this and substitute an empty dataset.
    """
    if len(rows) == 0:
        raise ValueError(f"{source_name}: loaded dataset has zero rows.")
    prompt_key = _first_present(rows[0].keys(), prompt_candidates)
    answer_key = _first_present(rows[0].keys(), answer_candidates)
    if prompt_key is None or answer_key is None:
        raise ValueError(
            f"{source_name}: could not find prompt/answer columns. "
            f"Looked for prompt in {prompt_candidates}, answer in {answer_candidates}. "
            f"Actual columns: {sorted(rows[0].keys())}. Add the right names here rather "
            f"than silently training on an empty/placeholder dataset."
        )
    out = []
    for row in rows:
        prompt = row.get(prompt_key)
        answer = row.get(answer_key)
        if prompt is None or answer is None:
            continue
        if answer_transform is not None:
            answer = answer_transform(answer)
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            continue
        out.append({"prompt": str(prompt), "answer": str(answer)})
    if not out:
        raise ValueError(
            f"{source_name}: found prompt/answer columns ({prompt_key!r}/{answer_key!r}) "
            f"but normalization produced zero usable examples -- check answer_transform."
        )
    return out


def _try_load(dataset_ids: list[str]):
    """Return (dataset_id, loaded_object) for the first dataset_id in the
    list that loads successfully. Raises the last error if none do.
    """
    from datasets import load_dataset

    last_err = None
    for dataset_id in dataset_ids:
        try:
            return dataset_id, load_dataset(dataset_id)
        except Exception as e:  # noqa: BLE001 - genuinely want to try the next candidate
            last_err = e
            continue
    raise ValueError(f"None of {dataset_ids} could be loaded. Last error: {last_err}")


def _pick_split(available_splits, preferred: list[str]) -> str:
    for s in preferred:
        if s in available_splits:
            return s
    return next(iter(available_splits))


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "train") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return normalize_rows(list(ds), ["question"], ["answer"], "gsm8k", answer_transform=extract_gsm8k_answer)


def load_gsm8k_validation(subset_size: int = 96, seed: int = 2025) -> list[dict]:
    """Deterministic held-out GSM8K validation subset."""
    rows = load_gsm8k("train")
    if subset_size >= len(rows):
        return rows
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:subset_size]]


def load_math500(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    return normalize_rows(list(ds), ["problem"], ["answer"], "math500")


def load_numina_math_cot(split: str = "train") -> list[dict]:
    from datasets import load_dataset

    # NuminaMath-CoT ships `solution` (a full CoT derivation) rather than a
    # clean final-answer field; a boxed answer is extracted from it, and rows
    # without one are skipped rather than guessed at (see normalize_rows'
    # "zero usable examples" check for the case where this silently breaks).
    ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)
    return normalize_rows(list(ds), ["problem"], ["solution"], "numina_math_cot",
                           answer_transform=extract_last_boxed)



def load_numina_sft(split: str = "train") -> list[dict]:
    """Load full NuminaMath solutions for supervised fine-tuning."""
    from datasets import load_dataset

    ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)
    out = []
    for row in ds:
        problem = row.get("problem")
        solution = row.get("solution")
        if problem is None or solution is None:
            continue
        if not str(problem).strip() or not str(solution).strip():
            continue
        out.append({"prompt": str(problem), "completion": str(solution)})
    if not out:
        raise ValueError("numina_math_cot SFT normalization produced zero examples.")
    return out


def load_hmmt_2025(split: str = "train") -> list[dict]:
    """Load the MathArena HMMT February 2025 evaluation set."""
    dataset_id, ds = _try_load(["MathArena/hmmt_feb_2025"])
    chosen_split = _pick_split(ds.keys(), [split, "train", "test"])
    return normalize_rows(
        list(ds[chosen_split]),
        ["problem", "Problem", "question"],
        ["answer", "Answer", "final_answer"],
        f"hmmt2025 ({dataset_id})",
    )

def load_aime_2024(split: str = "train") -> list[dict]:
    dataset_id, ds = _try_load(["Maxwell-Jia/AIME_2024", "HuggingFaceH4/aime_2024"])
    chosen_split = _pick_split(ds.keys(), [split, "train", "test"])
    return normalize_rows(list(ds[chosen_split]), ["Problem", "problem", "question"],
                           ["Answer", "answer", "final_answer"], f"aime2024 ({dataset_id})")


def load_aime_2025(split: str = "train") -> list[dict]:
    dataset_id, ds = _try_load(["opencompass/AIME2025", "MathArena/aime_2025", "yentinglin/aime_2025"])
    chosen_split = _pick_split(ds.keys(), [split, "train", "test"])
    return normalize_rows(list(ds[chosen_split]), ["Problem", "problem", "question"],
                           ["Answer", "answer", "final_answer"], f"aime2025 ({dataset_id})")


def load_humaneval(split: str = "test") -> list[dict]:
    """Returns HumanEval-schema dicts directly (prompt/test/entry_point),
    NOT the generic {"prompt", "answer"} schema -- correctness here is
    determined by executing `test`, not by comparing an "answer" string.
    See crisp/data/reward.py:code_reward and eval/run_eval.py.
    """
    dataset_id, ds = _try_load(["openai/openai_humaneval", "openai_humaneval"])
    chosen_split = _pick_split(ds.keys(), [split, "test"])
    rows = list(ds[chosen_split])
    required = ["prompt", "canonical_solution", "test", "entry_point"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"humaneval ({dataset_id}): missing expected columns {missing}, found {sorted(rows[0].keys())}")
    return [
        {
            "prompt": row["prompt"],
            "answer": "",
            "task_id": row.get("task_id", ""),
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in rows
    ]


_TRAIN_LOADERS: dict[str, Callable[[Optional[str]], list[dict]]] = {
    "gsm8k": lambda split: load_gsm8k(split or "train"),
    "numina_math_cot": lambda split: load_numina_math_cot(split or "train"),
    "math500": lambda split: load_math500(split or "test"),
}

_EVAL_LOADERS: dict[str, Callable[[], list[dict]]] = {
    "gsm8k": lambda: load_gsm8k("test"),
    "gsm8k_validation": lambda: load_gsm8k_validation(96, 2025),
    "math500": lambda: load_math500("test"),
    "aime2024": load_aime_2024,
    "aime2025": load_aime_2025,
    "hmmt2025": load_hmmt_2025,
    "humaneval": lambda: load_humaneval("test"),
}


def load_train_dataset(
    name: str,
    split: Optional[str] = None,
    train_subset_size: Optional[int] = None,
) -> list[dict]:
    """Load a training dataset, optionally truncated to `train_subset_size`
    rows.

    `train_subset_size` is applied BEFORE the trainer ever sees a batch,
    so a 100k-row dataset sliced to 8 rows returns 8 rows (not 100k rows
    with 7 padded). The subset is the deterministic first-N rows of the
    normalized dataset -- no shuffling here, because the trainer's own
    batching loop is what applies any randomization.

    A `train_subset_size` that exceeds the dataset's length is a no-op
    (the full list is returned) rather than an error: smoke tests with
    `train_subset_size: 1000` against gsm8k's 7k examples should not
    crash if the upstream dataset shrinks.
    """
    if name not in _TRAIN_LOADERS:
        raise ValueError(f"Unknown train dataset '{name}'. Known: {sorted(_TRAIN_LOADERS)}")
    if train_subset_size is not None:
        if train_subset_size < 0:
            raise ValueError(
                f"train_subset_size must be non-negative, got {train_subset_size}"
            )
        if train_subset_size == 0:
            raise ValueError("train_subset_size must be positive.")
        # Hugging Face datasets support split slicing, so smoke tests do not
        # materialize the full source dataset (NuminaMath can be very large).
        base_split = split or {
            "gsm8k": "train",
            "numina_math_cot": "train",
            "math500": "test",
        }.get(name, "train")
        if "[" not in base_split:
            rows = _TRAIN_LOADERS[name](f"{base_split}[:{train_subset_size}]")
        else:
            rows = _TRAIN_LOADERS[name](base_split)
            rows = rows[:train_subset_size]
    else:
        rows = _TRAIN_LOADERS[name](split)
    return rows


def load_eval_dataset(name: str) -> list[dict]:
    if name not in _EVAL_LOADERS:
        raise ValueError(f"Unknown eval dataset '{name}'. Known: {sorted(_EVAL_LOADERS)}")
    return _EVAL_LOADERS[name]()
