#!/usr/bin/env python3
"""
Minimal single-file SFT script for Qwen 3 0.6B on the local Messages JSONL.

This script intentionally keeps everything explicit and close together so it is
easy to read top-to-bottom:
1. Parse CLI args
2. Load and validate JSONL
3. Convert rows into prompt/completion examples
4. Shuffle and split train/val
5. Load model and tokenizer
6. Configure the trainer
7. Train and save
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "training-data"
    / "final"
    / "messages_reply_pairs.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Qwen 3 0.6B SFT script")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the input JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where trainer outputs and final model will be saved",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name to fine-tune",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length used during training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Training learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.02,
        help="Validation split fraction, e.g. 0.02 for 2%%",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling and splitting",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val-size must be between 0 and 1")


def validate_message(message: object, expected_role: str, line_number: int, index: int) -> str:
    if not isinstance(message, dict):
        raise ValueError(f"Line {line_number}: message {index} must be an object")
    role = message.get("role")
    if role != expected_role:
        raise ValueError(
            f"Line {line_number}: message {index} role must be '{expected_role}', got {role!r}"
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Line {line_number}: message {index} content must be a string")
    return content


def load_examples(dataset_path: Path) -> list[dict]:
    examples: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if not isinstance(row, list):
                raise ValueError(f"Line {line_number}: row must be a JSON array")
            if len(row) != 2:
                raise ValueError(f"Line {line_number}: row must contain exactly 2 messages")

            user_text = validate_message(row[0], expected_role="user", line_number=line_number, index=0)
            assistant_text = validate_message(
                row[1], expected_role="assistant", line_number=line_number, index=1
            )

            examples.append(
                {
                    "prompt": [{"role": "user", "content": user_text}],
                    "completion": [{"role": "assistant", "content": assistant_text}],
                }
            )

    if not examples:
        raise ValueError(f"No training examples found in {dataset_path}")
    return examples


def split_examples(examples: list[dict], val_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_size))
    if val_count >= len(shuffled):
        raise ValueError("Validation split leaves no training examples; reduce --val-size")

    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]
    return train_examples, val_examples


def pick_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    set_seed(args.seed)

    examples = load_examples(args.dataset_path)
    train_examples, val_examples = split_examples(examples, val_size=args.val_size, seed=args.seed)

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    effective_batch_size = args.batch_size * args.grad_accum
    print("Starting training with:")
    print(f"  dataset path: {args.dataset_path}")
    print(f"  total rows: {len(examples)}")
    print(f"  train rows: {len(train_dataset)}")
    print(f"  val rows: {len(val_dataset)}")
    print(f"  model name: {args.model_name}")
    print(f"  effective batch size: {effective_batch_size}")
    print(f"  output dir: {args.output_dir}")

    torch_dtype = pick_torch_dtype()
    print(f"  torch dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    )

    # Full fine-tuning on a smaller GPU benefits from checkpointing memory-heavy activations.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
