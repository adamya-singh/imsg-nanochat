#!/usr/bin/env python3
"""
Upload a Qwen3-14B QLoRA adapter directory to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "output-adapter"
DEFAULT_TRAIN_SCRIPT_PATH = Path(__file__).resolve().parent / "train.py"
DEFAULT_REPO_ID = "adamyathegreat/qwen-3-14b-my-texts-qlora"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-14B"
UPLOAD_EXCLUDE_PATTERNS = [
    ".DS_Store",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.tmp",
    "checkpoint-*/*",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a saved Qwen3-14B QLoRA adapter to the Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target Hugging Face model repo in the form namespace/name",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Path to a local PEFT adapter directory",
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
        help="Base model identifier used for adapter training",
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
        default="Upload Qwen3-14B QLoRA adapter",
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
    return parser.parse_args(argv)


def resolve_visibility(args: argparse.Namespace) -> bool:
    if args.public:
        return False
    return True


def validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter path is not a directory: {adapter_dir}")

    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError("Adapter directory is missing adapter_config.json")

    if not any((adapter_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(
            "Adapter directory is missing adapter weights. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )


def validate_readme_path(readme_path: Path | None) -> None:
    if readme_path is None:
        return
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")
    if not readme_path.is_file():
        raise ValueError(f"README path is not a file: {readme_path}")


def resolve_hf_token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or run `hf auth login` first."
        )
    return token


def build_model_card(
    *,
    repo_id: str,
    adapter_dir: Path,
    base_model: str,
    dataset_path: Path | None,
    train_script_path: Path | None,
) -> str:
    dataset_display = str(dataset_path) if dataset_path else "Not provided"
    train_script_display = str(train_script_path) if train_script_path else "Not provided"
    adapter_dir_display = str(adapter_dir)

    return textwrap.dedent(
        f"""\
        ---
        library_name: peft
        base_model: {base_model}
        tags:
          - text-generation
          - transformers
          - peft
          - qlora
          - personal-style
          - sft
        ---

        # {repo_id}

        QLoRA adapter trained from `{base_model}` on personal message reply pairs.

        ## Provenance

        - Base model: `{base_model}`
        - Local adapter directory: `{adapter_dir_display}`
        - Local dataset path: `{dataset_display}`
        - Training script: `{train_script_display}`
        - Output type: PEFT LoRA adapter only, not merged full model weights

        ## Intended Use

        This adapter is intended for private experimentation around personal reply style transfer.
        It may reproduce private, narrow, or inconsistent conversational patterns from the fine-tuning data.
        Review generations carefully before any broader use.

        ## Inference

        ```python
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_model = "{base_model}"
        adapter_id = "{repo_id}"

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, adapter_id)

        messages = [{{"role": "user", "content": "you free later?"}}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        ```
        """
    )


def upload_model_card(
    api: object,
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


def upload_adapter_folder(
    api: object,
    *,
    repo_id: str,
    adapter_dir: Path,
    revision: str | None,
    commit_message: str,
) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        revision=revision,
        commit_message=commit_message,
        ignore_patterns=UPLOAD_EXCLUDE_PATTERNS,
    )


def main() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    args = parse_args()
    private = resolve_visibility(args)

    validate_adapter_dir(args.adapter_dir)
    validate_readme_path(args.readme_path)

    print(f"Validated adapter directory: {args.adapter_dir}")
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
    print("Starting adapter upload...")
    try:
        upload_adapter_folder(
            api,
            repo_id=args.repo_id,
            adapter_dir=args.adapter_dir,
            revision=args.revision,
            commit_message=args.commit_message,
        )
    except HfHubHTTPError as error:
        raise RuntimeError(
            "Adapter upload failed. Check your network connection, repo permissions, "
            "and local adapter directory contents."
        ) from error

    if not args.no_model_card:
        if args.readme_path is not None:
            readme_text = args.readme_path.read_text(encoding="utf-8")
            print(f"Uploading custom README from: {args.readme_path}")
        else:
            readme_text = build_model_card(
                repo_id=args.repo_id,
                adapter_dir=args.adapter_dir,
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
                "Adapter upload succeeded, but uploading README.md failed."
            ) from error

    final_url = f"https://huggingface.co/{args.repo_id}"
    if args.revision:
        final_url = f"{final_url}/tree/{args.revision}"

    print("Upload complete.")
    print(f"Adapter URL: {final_url}")


if __name__ == "__main__":
    main()
