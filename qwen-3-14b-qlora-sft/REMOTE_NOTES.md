# Remote Notes

Record cluster runs, Hugging Face uploads, and evaluation notes for the Qwen3-14B QLoRA experiment here.

## 2026-07-07 Cluster Setup

- Created local Python 3.12 environment at `qwen-3-14b-qlora-sft/.venv` with mamba.
- Installed CUDA training stack: PyTorch 2.12.1+cu130, Transformers 5.13.0, PEFT 0.19.1, bitsandbytes 0.49.2, Accelerate 1.14.0, TRL 1.7.1, and W&B 0.28.0.
- Verified RTX 3090 CUDA visibility and bf16 support.
- Cached official `Qwen/Qwen3-14B` weights locally in the Hugging Face cache.
- Verified dataset load from `training-data/final/messages_reply_pairs.jsonl`: 12,637 rows, split into 12,385 train rows and 252 eval rows with seed 42.
- Ran full local test suite with importlib mode: 59 tests passed.
- Ran 32-row QLoRA smoke training successfully. The smoke adapter was saved under an ignored `output-smoke-adapter*` directory.
- Ran one-shot smoke inference through the adapter runner successfully.
- W&B credentials are available through `WANDB_API_KEY`; default run target is project `imsg-nanochat`, run `qwen3-14b-qlora-v1`.

## Full Run

- Status: ready to start.
- Output directory: `qwen-3-14b-qlora-sft/output-adapter`
- Expected duration: roughly 5-8 hours for one epoch on the RTX 3090.
