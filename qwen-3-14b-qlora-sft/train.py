#!/usr/bin/env python3
"""
QLoRA SFT entrypoint for Qwen3-14B on the local Messages JSONL.

The script mirrors the older 0.6B experiment shape, but it trains LoRA
adapters against a 4-bit quantized base model so it can fit on a single
24 GB RTX 3090-class GPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "training-data" / "final" / "messages_reply_pairs.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output-adapter"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-14B"
DEFAULT_WANDB_PROJECT = "imsg-nanochat"
DEFAULT_WANDB_RUN_NAME = "qwen3-14b-qlora-v1"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
IGNORE_INDEX = -100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-14B QLoRA SFT script")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the input nanochat-style JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where LoRA adapter artifacts will be saved",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base Hugging Face model name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length",
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
        default=1e-4,
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
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for smoke-test training rows",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Optional cap for smoke-test eval rows",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=DEFAULT_WANDB_RUN_NAME,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run",
    )
    return parser.parse_args(argv)


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
    if args.lora_r < 1:
        raise ValueError("--lora-r must be >= 1")
    if args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be >= 1")
    if not (0.0 <= args.lora_dropout < 1.0):
        raise ValueError("--lora-dropout must be in [0, 1)")
    for name in ("max_train_rows", "max_eval_rows"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.wandb_project is not None and not args.wandb_project.strip():
        raise ValueError("--wandb-project must not be blank")
    if args.wandb_run_name is not None and not args.wandb_run_name.strip():
        raise ValueError("--wandb-run-name must not be blank")


def setup_wandb(args: argparse.Namespace) -> None:
    if args.no_wandb:
        return

    import os

    wandb_dir = args.output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))


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


def load_examples(dataset_path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
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
            examples.append({"user": user_text, "assistant": assistant_text})

    if not examples:
        raise ValueError(f"No training examples found in {dataset_path}")
    return examples


def split_examples(
    examples: list[dict[str, str]],
    *,
    val_size: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_size))
    if val_count >= len(shuffled):
        raise ValueError("Validation split leaves no training examples; reduce --val-size")

    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]
    return train_examples, val_examples


def apply_row_limits(
    train_examples: list[dict[str, str]],
    eval_examples: list[dict[str, str]],
    *,
    max_train_rows: int | None,
    max_eval_rows: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if max_train_rows is not None:
        train_examples = train_examples[:max_train_rows]
    if max_eval_rows is not None:
        eval_examples = eval_examples[:max_eval_rows]
    if not train_examples:
        raise ValueError("No training rows remain after applying --max-train-rows")
    if not eval_examples:
        raise ValueError("No eval rows remain after applying --max-eval-rows")
    return train_examples, eval_examples


def render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    return list(tokenized["input_ids"])


def build_tokenized_example(
    tokenizer: Any,
    *,
    user_text: str,
    assistant_text: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_messages = [{"role": "user", "content": user_text}]
    full_messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    prompt_text = render_chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = render_chat_text(tokenizer, full_messages, add_generation_prompt=False)
    prompt_ids = tokenize_text(tokenizer, prompt_text)
    input_ids = tokenize_text(tokenizer, full_text)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    labels = [IGNORE_INDEX] * min(len(prompt_ids), len(input_ids))
    labels.extend(input_ids[len(labels):])
    labels = labels[:len(input_ids)]

    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("Tokenized example has no assistant tokens left after truncation")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class TokenizedChatDataset:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def pick_compute_dtype() -> Any:
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(compute_dtype: Any) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(DEFAULT_TARGET_MODULES),
    )


def tokenize_examples(
    tokenizer: Any,
    examples: list[dict[str, str]],
    *,
    max_length: int,
) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    dropped = 0
    for example in examples:
        try:
            rows.append(
                build_tokenized_example(
                    tokenizer,
                    user_text=example["user"],
                    assistant_text=example["assistant"],
                    max_length=max_length,
                )
            )
        except ValueError:
            dropped += 1
    if not rows:
        raise ValueError("No tokenized rows remain; increase --max-length")
    if dropped:
        print(f"Dropped {dropped} rows with no assistant tokens after truncation")
    return rows


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    random.seed(args.seed)
    set_seed(args.seed)

    examples = load_examples(args.dataset_path)
    train_examples, eval_examples = split_examples(examples, val_size=args.val_size, seed=args.seed)
    train_examples, eval_examples = apply_row_limits(
        train_examples,
        eval_examples,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
    )

    compute_dtype = pick_compute_dtype()
    print("Starting Qwen3-14B QLoRA training with:")
    print(f"  dataset path: {args.dataset_path}")
    print(f"  total rows: {len(examples)}")
    print(f"  train rows: {len(train_examples)}")
    print(f"  eval rows: {len(eval_examples)}")
    print(f"  model name: {args.model_name}")
    print(f"  output dir: {args.output_dir}")
    print(f"  max length: {args.max_length}")
    print(f"  effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  compute dtype: {compute_dtype}")
    print(f"  wandb: {'disabled' if args.no_wandb else args.wandb_project + '/' + args.wandb_run_name}")

    setup_wandb(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = TokenizedChatDataset(
        tokenize_examples(tokenizer, train_examples, max_length=args.max_length)
    )
    eval_dataset = TokenizedChatDataset(
        tokenize_examples(tokenizer, eval_examples, max_length=args.max_length)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=build_quantization_config(compute_dtype),
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=None if args.no_wandb else args.wandb_run_name,
        seed=args.seed,
        gradient_checkpointing=True,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
