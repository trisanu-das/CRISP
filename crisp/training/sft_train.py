"""Completion-only supervised fine-tuning for SFT -> RL experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from crisp.data.build_dataset import load_numina_sft
from crisp.model.load import attach_or_load_lora, load_model_and_tokenizer
from crisp.utils.config import load_config
from crisp.utils.distributed import init_distributed, is_main_process
from crisp.utils.seeding import set_seed


def _tokenize_sft_example(example, tokenizer, cfg):
    prompt_messages = [
        {"role": "system", "content": cfg.model.system_prompt},
        {"role": "user", "content": example["prompt"]},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": example["completion"]}
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    max_length = int(cfg.sft.max_sequence_length)
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    labels = list(encoded["input_ids"])
    if cfg.sft.get("completion_only_loss", True):
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
    encoded["labels"] = labels
    return encoded


def main(config_path: str, overrides: Optional[dict] = None) -> None:
    cfg = load_config(config_path, overrides)
    init_distributed()
    set_seed(cfg.seed)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model, _ = attach_or_load_lora(
        model,
        cfg.lora,
        initial_adapter=cfg.model.get("initial_adapter"),
    )
    model.config.use_cache = False

    rows = load_numina_sft(cfg.data.train_split)
    if cfg.data.train_subset_size is not None:
        rows = rows[: int(cfg.data.train_subset_size)]
    dataset = Dataset.from_list(rows).map(
        lambda row: _tokenize_sft_example(row, tokenizer, cfg),
        remove_columns=["prompt", "completion"],
    )

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_to = ["wandb"] if cfg.logging.backend == "wandb" else []
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(cfg.training.total_steps),
        num_train_epochs=float(cfg.sft.num_train_epochs),
        per_device_train_batch_size=int(cfg.training.micro_batch_size),
        gradient_accumulation_steps=int(cfg.training.grad_accum_steps),
        learning_rate=float(cfg.optimizer.lr),
        warmup_steps=int(cfg.optimizer.warmup_steps),
        weight_decay=float(cfg.optimizer.weight_decay),
        max_grad_norm=float(cfg.optimizer.max_grad_norm),
        logging_steps=max(1, int(cfg.training.log_every)),
        save_steps=max(1, int(cfg.training.save_every)) if cfg.training.save_every else 10**9,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and cfg.model.torch_dtype == "bfloat16",
        fp16=torch.cuda.is_available() and cfg.model.torch_dtype == "float16",
        report_to=report_to,
        run_name=cfg.logging.run_name or "sft",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    resume = cfg.training.get("resume_from")
    trainer.train(resume_from_checkpoint=resume if resume else None)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if is_main_process():
        (output_dir / "_SUCCESS").write_text("completed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    from train_launcher import parse_overrides
    main(args.config, overrides=parse_overrides(args.override))
