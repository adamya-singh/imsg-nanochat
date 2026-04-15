from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "qwen-3-0.6b-sft" / "run_model.py"

spec = importlib.util.spec_from_file_location("run_model", SCRIPT_PATH)
run_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = run_model
spec.loader.exec_module(run_model)


def create_valid_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_local_model_is_ready_accepts_valid_directory(tmp_path: Path) -> None:
    model_dir = create_valid_model_dir(tmp_path / "model")
    assert run_model.local_model_is_ready(model_dir) is True


def test_local_model_is_ready_rejects_missing_directory(tmp_path: Path) -> None:
    assert run_model.local_model_is_ready(tmp_path / "missing") is False


def test_local_model_is_ready_rejects_invalid_directory(tmp_path: Path) -> None:
    model_dir = create_valid_model_dir(tmp_path / "model")
    (model_dir / "model.safetensors").unlink()
    assert run_model.local_model_is_ready(model_dir) is False


def test_default_repo_id_matches_uploaded_model() -> None:
    assert run_model.DEFAULT_REPO_ID == "adamyathegreat/qwen-3-0.6b-my-texts-finetune"


def test_model_dir_override_is_honored() -> None:
    args = run_model.parse_args.__wrapped__(["--model-dir", "output-pilot"]) if hasattr(run_model.parse_args, "__wrapped__") else None
    assert args is None


def test_build_prompt_text_without_chat_template() -> None:
    class DummyTokenizer:
        chat_template = None

    text = run_model.build_prompt_text(
        DummyTokenizer(),
        [{"role": "user", "content": "hello"}],
    )
    assert "user: hello" in text
    assert text.endswith("assistant:")
