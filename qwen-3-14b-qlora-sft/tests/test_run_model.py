from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "qwen-3-14b-qlora-sft" / "run_model.py"

spec = importlib.util.spec_from_file_location("run_model_14b", SCRIPT_PATH)
run_model_14b = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = run_model_14b
spec.loader.exec_module(run_model_14b)


class DummyMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class DummyCuda:
    def __init__(self, available: bool, bf16: bool) -> None:
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class DummyBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = DummyMps(mps_available)


class DummyTorch:
    def __init__(self, *, cuda_available: bool, bf16: bool, mps_available: bool) -> None:
        self.cuda = DummyCuda(cuda_available, bf16)
        self.backends = DummyBackends(mps_available)


def create_adapter_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    return path


def test_validate_adapter_dir_accepts_peft_adapter(tmp_path: Path) -> None:
    adapter_dir = create_adapter_dir(tmp_path / "adapter")
    run_model_14b.validate_adapter_dir(adapter_dir)


def test_validate_adapter_dir_rejects_missing_weights(tmp_path: Path) -> None:
    adapter_dir = create_adapter_dir(tmp_path / "adapter")
    (adapter_dir / "adapter_model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="adapter weights"):
        run_model_14b.validate_adapter_dir(adapter_dir)


def test_choose_device_prefers_cuda_bf16() -> None:
    torch = DummyTorch(cuda_available=True, bf16=True, mps_available=True)

    assert run_model_14b.choose_device(torch) == ("cuda", "bfloat16")


def test_choose_device_uses_cuda_fp16_when_bf16_is_missing() -> None:
    torch = DummyTorch(cuda_available=True, bf16=False, mps_available=True)

    assert run_model_14b.choose_device(torch) == ("cuda", "float16")


def test_choose_device_falls_back_to_mps_then_cpu() -> None:
    assert run_model_14b.choose_device(
        DummyTorch(cuda_available=False, bf16=False, mps_available=True)
    ) == ("mps", "float16")
    assert run_model_14b.choose_device(
        DummyTorch(cuda_available=False, bf16=False, mps_available=False)
    ) == ("cpu", "float32")


def test_build_prompt_text_disables_qwen3_thinking() -> None:
    class DummyTokenizer:
        def __init__(self) -> None:
            self.enable_thinking = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            enable_thinking,
        ):
            self.enable_thinking = enable_thinking
            return "prompt\n<think>\n\n</think>\n\n"

    tokenizer = DummyTokenizer()
    assert (
        run_model_14b.build_prompt_text(tokenizer, [{"role": "user", "content": "hi"}])
        == "prompt\n<think>\n\n</think>\n\n"
    )
    assert tokenizer.enable_thinking is False


def test_parse_defaults_match_adapter_runner() -> None:
    args = run_model_14b.parse_args([])

    assert args.model_name == "Qwen/Qwen3-14B"
    assert args.adapter_dir.name == "output-adapter"
    assert args.temperature == 0.7
    assert args.top_p == 0.8
    assert args.top_k == 20
