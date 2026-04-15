#!/usr/bin/env python3
"""
Pull and run the uploaded Qwen fine-tune locally.

Default behavior:
1. Reuse the local output directory if it already contains a saved model
2. Otherwise download the model from Hugging Face into that same directory
3. Load the model from the local directory
4. Start an interactive chat loop
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import get_token, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO_ID = "adamyathegreat/qwen-3-0.6b-my-texts-finetune"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull and run the Qwen fine-tuned model")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repo to download from if the local model is missing",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local directory to reuse or populate with the downloaded model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional one-shot prompt. If omitted, starts an interactive chat loop.",
    )
    return parser.parse_args()


def validate_generation_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be between 0 and 1")


def local_model_is_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    required_files = ["config.json", "tokenizer_config.json"]
    if any(not (model_dir / name).exists() for name in required_files):
        return False

    has_tokenizer_artifact = any(
        (model_dir / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
        )
    )
    if not has_tokenizer_artifact:
        return False

    weight_patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    has_weights = any(any(model_dir.glob(pattern)) for pattern in weight_patterns)
    return has_weights


def ensure_auth_for_private_repo() -> None:
    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or run `hf auth login` first."
        )


def ensure_local_model(repo_id: str, model_dir: Path) -> Path:
    if local_model_is_ready(model_dir):
        print(f"Using existing local model directory: {model_dir}")
        return model_dir

    ensure_auth_for_private_repo()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Local model not found or incomplete: {model_dir}")
    print(f"Downloading from Hugging Face repo: {repo_id}")
    print(f"Saving into local directory: {model_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    except HfHubHTTPError as error:
        raise RuntimeError(
            "Failed to download the model from Hugging Face. "
            "Check repo access and authentication."
        ) from error

    if not local_model_is_ready(model_dir):
        raise RuntimeError(
            f"Download completed but the local model directory is still incomplete: {model_dir}"
        )

    return model_dir


def pick_device() -> tuple[str, str]:
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float32"


def load_pipeline(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype_name = pick_device()
    dtype = torch.float16 if dtype_name == "float16" else torch.float32

    print(f"Loading model from local directory: {model_dir}")
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    )
    model = model.to(device)
    return tokenizer, model, device


def build_prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generate_reply(
    *,
    tokenizer,
    model,
    device: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    prompt_text = build_prompt_text(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_one_shot(args: argparse.Namespace, *, tokenizer, model, device: str) -> None:
    reply = generate_reply(
        tokenizer=tokenizer,
        model=model,
        device=device,
        messages=[{"role": "user", "content": args.prompt}],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(reply)


def run_chat_loop(args: argparse.Namespace, *, tokenizer, model, device: str) -> None:
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
        )
        print(f"model> {reply}")
        history.append({"role": "assistant", "content": reply})


def main() -> None:
    args = parse_args()
    validate_generation_args(args)

    model_dir = ensure_local_model(args.repo_id, args.model_dir)
    tokenizer, model, device = load_pipeline(model_dir)

    if args.prompt is not None:
        run_one_shot(args, tokenizer=tokenizer, model=model, device=device)
        return

    run_chat_loop(args, tokenizer=tokenizer, model=model, device=device)


if __name__ == "__main__":
    main()
