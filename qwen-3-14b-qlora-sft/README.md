# qwen-3-14b-qlora-sft

Cluster-focused QLoRA experiment for `Qwen/Qwen3-14B` on the local Messages reply-pair dataset.

This directory is intentionally parallel to `qwen-3-0.6b-sft/`. The old 0.6B path remains the historical M1-friendly full fine-tune; this path targets a CUDA machine such as the RTX 3090 on this cluster.

## Setup

Create a fresh Linux virtual environment. Do not reuse the copied Mac `.venv`.
On this cluster the working setup is a Python 3.12 local conda/mamba prefix at
`qwen-3-14b-qlora-sft/.venv`.

```bash
mamba create -y -p qwen-3-14b-qlora-sft/.venv python=3.12 pip
qwen-3-14b-qlora-sft/.venv/bin/python -m pip install -r qwen-3-14b-qlora-sft/requirements.txt
```

Confirm CUDA is visible:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Weights & Biases

The trainer logs to W&B by default:

- project: `imsg-nanochat`
- run name: `qwen3-14b-qlora-v1`

Authenticate before starting the real run:

```bash
qwen-3-14b-qlora-sft/.venv/bin/wandb login
```

Or provide an API key non-interactively:

```bash
export WANDB_API_KEY=...
```

Disable W&B for throwaway smoke runs with `--no-wandb`.
Local W&B files are written under the ignored adapter output directory.

## Smoke Training

Use a tiny subset first so dependency and CUDA issues show up quickly:

```bash
python qwen-3-14b-qlora-sft/train.py \
  --dataset-path training-data/final/messages_reply_pairs.jsonl \
  --output-dir qwen-3-14b-qlora-sft/output-smoke-adapter \
  --max-train-rows 32 \
  --max-eval-rows 8 \
  --max-length 128 \
  --no-wandb
```

## Full Adapter Training

Default settings are tuned for a single 24 GB RTX 3090:

```bash
python qwen-3-14b-qlora-sft/train.py \
  --dataset-path training-data/final/messages_reply_pairs.jsonl \
  --output-dir qwen-3-14b-qlora-sft/output-adapter \
  --wandb-project imsg-nanochat \
  --wandb-run-name qwen3-14b-qlora-v1
```

For a durable background run with log and PID under the ignored output directory:

```bash
mkdir -p qwen-3-14b-qlora-sft/output-adapter
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup qwen-3-14b-qlora-sft/.venv/bin/python qwen-3-14b-qlora-sft/train.py \
  --dataset-path training-data/final/messages_reply_pairs.jsonl \
  --output-dir qwen-3-14b-qlora-sft/output-adapter \
  --wandb-project imsg-nanochat \
  --wandb-run-name qwen3-14b-qlora-v1 \
  > qwen-3-14b-qlora-sft/output-adapter/train.log 2>&1 &
echo $! > qwen-3-14b-qlora-sft/output-adapter/train.pid
```

The trainer:

- loads `Qwen/Qwen3-14B` in 4-bit NF4
- trains LoRA adapters only
- uses Qwen3 chat formatting with `enable_thinking=False`
- masks prompt tokens so loss is applied only to assistant replies
- saves adapter-only artifacts by default

## Run

One-shot prompt:

```bash
python qwen-3-14b-qlora-sft/run_model.py \
  --adapter-dir qwen-3-14b-qlora-sft/output-adapter \
  --prompt "you free later?"
```

Interactive loop:

```bash
python qwen-3-14b-qlora-sft/run_model.py \
  --adapter-dir qwen-3-14b-qlora-sft/output-adapter
```

## Upload

Authenticate first with either `HF_TOKEN` or `hf auth login`, then upload the adapter:

```bash
python qwen-3-14b-qlora-sft/upload_to_huggingface.py \
  --repo-id adamyathegreat/qwen-3-14b-my-texts-qlora \
  --adapter-dir qwen-3-14b-qlora-sft/output-adapter \
  --dataset-path training-data/final/messages_reply_pairs.jsonl
```

Defaults:

- uploads to a private model repo unless `--public` is passed
- uploads adapter-only PEFT artifacts, not merged full model weights
- generates a minimal adapter model card unless `--no-model-card` is passed

## Tests

```bash
python -m pytest qwen-3-14b-qlora-sft/tests -q
```
