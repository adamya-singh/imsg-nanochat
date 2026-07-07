#!/usr/bin/env python3
"""
Run the Qwen3-14B base model with a local LoRA adapter.

On CUDA, the base model is loaded in 4-bit NF4 and the adapter is applied with
PEFT. CPU/MPS fallback exists for code-path completeness, but the intended
runtime is this cluster's RTX 3090.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen3-14B"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "output-adapter"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qwen3-14B QLoRA adapter")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base Hugging Face model name",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Local PEFT adapter directory",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p nucleus sampling value",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling value",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional one-shot prompt. If omitted, starts an interactive chat loop.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be between 0 and 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    validate_adapter_dir(args.adapter_dir)


def validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter path is not a directory: {adapter_dir}")

    required_files = ["adapter_config.json"]
    missing_required = [name for name in required_files if not (adapter_dir / name).exists()]
    if missing_required:
        joined = ", ".join(missing_required)
        raise FileNotFoundError(f"Adapter directory is missing required files: {joined}")

    weight_names = ("adapter_model.safetensors", "adapter_model.bin")
    if not any((adapter_dir / name).exists() for name in weight_names):
        raise FileNotFoundError(
            "Adapter directory is missing adapter weights. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )


def adapter_has_tokenizer(adapter_dir: Path) -> bool:
    return (adapter_dir / "tokenizer_config.json").exists() and any(
        (adapter_dir / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
        )
    )


def choose_device(torch_module: Any) -> tuple[str, str]:
    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return "cuda", "bfloat16"
        return "cuda", "float16"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


def build_quantization_config(dtype: Any) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_model_and_tokenizer(model_name: str, adapter_dir: Path) -> tuple[Any, Any, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype_name = choose_device(torch)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]

    tokenizer_source = adapter_dir if adapter_has_tokenizer(adapter_dir) else model_name
    print(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {model_name}")
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype_name}")
    if device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=build_quantization_config(dtype),
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=False,
        ).to(device)

    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return tokenizer, model, device


def build_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_reply(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    import torch

    prompt_text = build_prompt_text(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if device == "cuda":
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    else:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return strip_thinking_blocks(
        tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    )


def run_one_shot(args: argparse.Namespace, *, tokenizer: Any, model: Any, device: str) -> None:
    reply = generate_reply(
        tokenizer=tokenizer,
        model=model,
        device=device,
        messages=[{"role": "user", "content": args.prompt}],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(reply)


def run_chat_loop(args: argparse.Namespace, *, tokenizer: Any, model: Any, device: str) -> None:
    print("Interactive chat started. Type `exit` or `quit` to stop. Type `reset` to clear history.")
    history: list[dict[str, str]] = []

    while True:
        user_input = input("you> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            return
        if user_input.lower() == "reset":
            history.clear()
            print("Conversation history cleared.")
            continue

        history.append({"role": "user", "content": user_input})
        reply = generate_reply(
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        print(f"model> {reply}")
        history.append({"role": "assistant", "content": reply})


def main() -> None:
    args = parse_args()
    validate_args(args)

    tokenizer, model, device = load_model_and_tokenizer(args.model_name, args.adapter_dir)
    if args.prompt is not None:
        run_one_shot(args, tokenizer=tokenizer, model=model, device=device)
        return
    run_chat_loop(args, tokenizer=tokenizer, model=model, device=device)


if __name__ == "__main__":
    main()
