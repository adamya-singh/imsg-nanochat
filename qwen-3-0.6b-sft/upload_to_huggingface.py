#!/usr/bin/env python3
"""
Upload a completed Qwen fine-tune output directory to the Hugging Face Hub.

This script intentionally keeps the flow explicit:
1. Parse CLI args
2. Validate the local saved-model directory
3. Resolve authentication from HF_TOKEN or prior Hugging Face login
4. Create the remote model repo if needed
5. Upload the full saved-model folder
6. Optionally upload a generated or custom README.md
7. Print the final repo URL
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_TRAIN_SCRIPT_PATH = Path(__file__).resolve().parent / "train.py"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
UPLOAD_EXCLUDE_PATTERNS = [
    ".DS_Store",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.tmp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a saved Qwen fine-tune directory to the Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repo in the form namespace/name",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to a local Transformers save_pretrained directory",
    )
    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default behavior)",
    )
    visibility_group.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model identifier used for fine-tuning",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional local dataset path to include in the model card",
    )
    parser.add_argument(
        "--train-script-path",
        type=Path,
        default=DEFAULT_TRAIN_SCRIPT_PATH,
        help="Optional local training script path to include in the model card",
    )
    parser.add_argument(
        "--readme-path",
        type=Path,
        default=None,
        help="Optional custom README.md file to upload instead of generating one",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload fine-tuned model",
        help="Commit message to use for the Hub upload",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or revision to upload to",
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip uploading any README.md model card",
    )
    return parser.parse_args()


def resolve_visibility(args: argparse.Namespace) -> bool:
    if args.public:
        return False
    return True


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Model directory is not a directory: {model_dir}")

    required_files = ["config.json", "tokenizer_config.json"]
    missing_required = [name for name in required_files if not (model_dir / name).exists()]
    if missing_required:
        joined = ", ".join(missing_required)
        raise FileNotFoundError(f"Model directory is missing required files: {joined}")

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
        raise FileNotFoundError(
            "Model directory is missing tokenizer artifacts. "
            "Expected at least one of tokenizer.json, tokenizer.model, spiece.model, "
            "sentencepiece.bpe.model, or vocab.json."
        )

    weight_patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    has_weights = any(
        any(model_dir.glob(pattern))
        for pattern in weight_patterns
    )
    if not has_weights:
        raise FileNotFoundError(
            "Model directory is missing weight files. "
            "Expected model.safetensors, model-*.safetensors, pytorch_model.bin, "
            "or pytorch_model-*.bin."
        )


def validate_readme_path(readme_path: Path | None) -> None:
    if readme_path is None:
        return
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")
    if not readme_path.is_file():
        raise ValueError(f"README path is not a file: {readme_path}")


def resolve_hf_token() -> str:
    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or run `hf auth login` first."
        )
    return token


def build_model_card(
    *,
    repo_id: str,
    model_dir: Path,
    base_model: str,
    dataset_path: Path | None,
    train_script_path: Path | None,
) -> str:
    dataset_display = str(dataset_path) if dataset_path else "Not provided"
    train_script_display = str(train_script_path) if train_script_path else "Not provided"
    model_dir_display = str(model_dir)

    return textwrap.dedent(
        f"""\
        ---
        library_name: transformers
        base_model: {base_model}
        tags:
          - text-generation
          - transformers
          - personal-style
          - sft
        ---

        # {repo_id}

        Fine-tuned from `{base_model}` using supervised fine-tuning on personal messages reply pairs.

        ## Provenance

        - Base model: `{base_model}`
        - Local model directory: `{model_dir_display}`
        - Local dataset path: `{dataset_display}`
        - Training script: `{train_script_display}`

        ## Intended Use

        This model is intended for personal experimentation around reply style transfer.
        It may reproduce private, narrow, or inconsistent conversational patterns from the fine-tuning data.
        Review generations carefully before any broader use.

        ## Inference

        ```python
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "{repo_id}"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        messages = [{{"role": "user", "content": "you free later?"}}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        ```
        """
    )


def upload_model_card(
    api: HfApi,
    *,
    repo_id: str,
    revision: str | None,
    commit_message: str,
    readme_text: str,
) -> None:
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        revision=revision,
        commit_message=commit_message,
    )


def upload_folder(
    api: HfApi,
    *,
    repo_id: str,
    model_dir: Path,
    revision: str | None,
    commit_message: str,
) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        revision=revision,
        commit_message=commit_message,
        ignore_patterns=UPLOAD_EXCLUDE_PATTERNS,
    )


def main() -> None:
    args = parse_args()
    private = resolve_visibility(args)

    validate_model_dir(args.model_dir)
    validate_readme_path(args.readme_path)

    print(f"Validated model directory: {args.model_dir}")
    print(f"Target repo: {args.repo_id}")
    print(f"Visibility: {'private' if private else 'public'}")

    token = resolve_hf_token()
    api = HfApi(token=token)

    try:
        api.whoami()
    except HfHubHTTPError as error:
        raise RuntimeError(
            "Failed to authenticate with the Hugging Face Hub. "
            "Check HF_TOKEN or your local login state."
        ) from error

    try:
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except HfHubHTTPError as error:
        raise RuntimeError(
            f"Failed to create or access model repo {args.repo_id}. "
            "Check repo permissions and namespace."
        ) from error

    print(f"Remote repo ready: {repo_url}")
    print("Starting model upload...")
    try:
        upload_folder(
            api,
            repo_id=args.repo_id,
            model_dir=args.model_dir,
            revision=args.revision,
            commit_message=args.commit_message,
        )
    except HfHubHTTPError as error:
        raise RuntimeError(
            "Model folder upload failed. Check your network connection, repo permissions, "
            "and local model directory contents."
        ) from error

    if not args.no_model_card:
        if args.readme_path is not None:
            readme_text = args.readme_path.read_text(encoding="utf-8")
            print(f"Uploading custom README from: {args.readme_path}")
        else:
            readme_text = build_model_card(
                repo_id=args.repo_id,
                model_dir=args.model_dir,
                base_model=args.base_model,
                dataset_path=args.dataset_path,
                train_script_path=args.train_script_path,
            )
            print("Uploading generated README.md")

        try:
            upload_model_card(
                api,
                repo_id=args.repo_id,
                revision=args.revision,
                commit_message=args.commit_message,
                readme_text=readme_text,
            )
        except HfHubHTTPError as error:
            raise RuntimeError(
                "Model upload succeeded, but uploading README.md failed."
            ) from error

    final_url = f"https://huggingface.co/{args.repo_id}"
    if args.revision:
        final_url = f"{final_url}/tree/{args.revision}"

    print("Upload complete.")
    print(f"Model URL: {final_url}")


if __name__ == "__main__":
    main()
