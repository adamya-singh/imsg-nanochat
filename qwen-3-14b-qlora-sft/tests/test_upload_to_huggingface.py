from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "qwen-3-14b-qlora-sft" / "upload_to_huggingface.py"

spec = importlib.util.spec_from_file_location("upload_14b", SCRIPT_PATH)
upload_14b = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = upload_14b
spec.loader.exec_module(upload_14b)


def create_adapter_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    return path


def test_validate_adapter_dir_accepts_peft_adapter(tmp_path: Path) -> None:
    adapter_dir = create_adapter_dir(tmp_path / "adapter")
    upload_14b.validate_adapter_dir(adapter_dir)


def test_validate_adapter_dir_rejects_full_model_shape(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "model"
    adapter_dir.mkdir()
    (adapter_dir / "config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        upload_14b.validate_adapter_dir(adapter_dir)


def test_validate_readme_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="README file not found"):
        upload_14b.validate_readme_path(tmp_path / "README.md")


def test_build_model_card_describes_adapter_only_output(tmp_path: Path) -> None:
    adapter_dir = create_adapter_dir(tmp_path / "adapter")

    readme = upload_14b.build_model_card(
        repo_id="alice/qwen-adapter",
        adapter_dir=adapter_dir,
        base_model="Qwen/Qwen3-14B",
        dataset_path=tmp_path / "messages.jsonl",
        train_script_path=tmp_path / "train.py",
    )

    assert "# alice/qwen-adapter" in readme
    assert "Qwen/Qwen3-14B" in readme
    assert "adapter only" in readme.lower()
    assert "BitsAndBytesConfig" in readme


def test_parse_defaults_match_private_adapter_upload() -> None:
    args = upload_14b.parse_args([])

    assert args.repo_id == "adamyathegreat/qwen-3-14b-my-texts-qlora"
    assert args.adapter_dir.name == "output-adapter"
    assert upload_14b.resolve_visibility(args) is True
