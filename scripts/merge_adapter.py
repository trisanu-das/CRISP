#!/usr/bin/env python
"""Merge a PEFT adapter into its base model for external evaluators."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    print(output)


if __name__ == "__main__":
    main()
