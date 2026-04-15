from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "qwen-3-0.6b-sft" / "upload_to_huggingface.py"

spec = importlib.util.spec_from_file_location("upload_to_huggingface", SCRIPT_PATH)
upload_to_huggingface = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = upload_to_huggingface
spec.loader.exec_module(upload_to_huggingface)


def create_valid_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_validate_model_dir_accepts_save_pretrained_shape(tmp_path: Path) -> None:
    model_dir = create_valid_model_dir(tmp_path / "model")
    upload_to_huggingface.validate_model_dir(model_dir)


def test_validate_model_dir_rejects_missing_config(tmp_path: Path) -> None:
    model_dir = create_valid_model_dir(tmp_path / "model")
    (model_dir / "config.json").unlink()

    with pytest.raises(FileNotFoundError, match="config.json"):
        upload_to_huggingface.validate_model_dir(model_dir)


def test_validate_model_dir_rejects_missing_weights(tmp_path: Path) -> None:
    model_dir = create_valid_model_dir(tmp_path / "model")
    (model_dir / "model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="weight files"):
        upload_to_huggingface.validate_model_dir(model_dir)


def test_validate_readme_path_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "README.md"

    with pytest.raises(FileNotFoundError, match="README file not found"):
        upload_to_huggingface.validate_readme_path(missing)


def test_build_model_card_includes_repo_and_provenance(tmp_path: Path) -> None:
    model_dir = create_valid_model_dir(tmp_path / "model")
    dataset_path = tmp_path / "dataset.jsonl"
    train_script_path = tmp_path / "train.py"

    readme = upload_to_huggingface.build_model_card(
        repo_id="alice/test-model",
        model_dir=model_dir,
        base_model="Qwen/Qwen3-0.6B",
        dataset_path=dataset_path,
        train_script_path=train_script_path,
    )

    assert "# alice/test-model" in readme
    assert "Qwen/Qwen3-0.6B" in readme
    assert str(dataset_path) in readme
    assert str(train_script_path) in readme
    assert "AutoModelForCausalLM" in readme
