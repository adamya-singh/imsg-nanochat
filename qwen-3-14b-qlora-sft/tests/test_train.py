from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "qwen-3-14b-qlora-sft" / "train.py"

spec = importlib.util.spec_from_file_location("train_14b", SCRIPT_PATH)
train_14b = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = train_14b
spec.loader.exec_module(train_14b)


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        self.enable_thinking_values.append(enable_thinking)
        parts = [f"{message['role']}:{message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts) + "\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_examples_accepts_two_message_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "messages.jsonl"
    write_jsonl(
        dataset_path,
        [
            [
                {"role": "user", "content": "you free later?"},
                {"role": "assistant", "content": "yea after 7"},
            ]
        ],
    )

    assert train_14b.load_examples(dataset_path) == [
        {"user": "you free later?", "assistant": "yea after 7"}
    ]


def test_load_examples_rejects_invalid_role(tmp_path: Path) -> None:
    dataset_path = tmp_path / "messages.jsonl"
    write_jsonl(
        dataset_path,
        [
            [
                {"role": "system", "content": "oops"},
                {"role": "assistant", "content": "reply"},
            ]
        ],
    )

    with pytest.raises(ValueError, match="role must be 'user'"):
        train_14b.load_examples(dataset_path)


def test_split_examples_is_deterministic() -> None:
    examples = [{"user": str(index), "assistant": str(index)} for index in range(20)]

    first = train_14b.split_examples(examples, val_size=0.2, seed=123)
    second = train_14b.split_examples(examples, val_size=0.2, seed=123)

    assert first == second
    assert len(first[0]) == 16
    assert len(first[1]) == 4


def test_render_chat_text_disables_qwen3_thinking() -> None:
    tokenizer = DummyTokenizer()

    train_14b.render_chat_text(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )

    assert tokenizer.enable_thinking_values == [False]


def test_prompt_masking_ignores_user_tokens_and_trains_assistant_tokens() -> None:
    tokenizer = DummyTokenizer()

    row = train_14b.build_tokenized_example(
        tokenizer,
        user_text="hello",
        assistant_text="hi",
        max_length=200,
    )

    first_trainable = next(
        index for index, label in enumerate(row["labels"]) if label != train_14b.IGNORE_INDEX
    )
    assert all(label == train_14b.IGNORE_INDEX for label in row["labels"][:first_trainable])
    assert row["labels"][first_trainable:] == row["input_ids"][first_trainable:]
    assert tokenizer.enable_thinking_values == [False, False]


def test_parse_defaults_match_14b_plan() -> None:
    args = train_14b.parse_args([])

    assert args.model_name == "Qwen/Qwen3-14B"
    assert args.output_dir.name == "output-adapter"
    assert args.max_length == 512
    assert args.batch_size == 1
    assert args.grad_accum == 16
    assert args.learning_rate == 1e-4
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.05
    assert args.wandb_project == "imsg-nanochat"
    assert args.wandb_run_name == "qwen3-14b-qlora-v1"
    assert args.no_wandb is False
    assert train_14b.DEFAULT_TARGET_MODULES == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_setup_wandb_sets_default_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = train_14b.parse_args([])
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_NAME", raising=False)
    monkeypatch.delenv("WANDB_LOG_MODEL", raising=False)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    args.output_dir = tmp_path / "adapter"

    train_14b.setup_wandb(args)

    import os

    assert os.environ["WANDB_PROJECT"] == "imsg-nanochat"
    assert os.environ["WANDB_NAME"] == "qwen3-14b-qlora-v1"
    assert os.environ["WANDB_LOG_MODEL"] == "false"
    assert os.environ["WANDB_DIR"] == str(tmp_path / "adapter" / "wandb")
    assert (tmp_path / "adapter" / "wandb").is_dir()


def test_setup_wandb_respects_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    args = train_14b.parse_args(["--no-wandb"])
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_DIR", raising=False)

    train_14b.setup_wandb(args)

    import os

    assert "WANDB_PROJECT" not in os.environ
    assert "WANDB_DIR" not in os.environ
