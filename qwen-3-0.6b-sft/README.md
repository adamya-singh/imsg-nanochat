# qwen-3-0.6b-sft

Minimal local fine-tuning and upload helpers for the Qwen 3 0.6B experiment in this repo.

## Training

The training entrypoint is:

```bash
python qwen-3-0.6b-sft/train.py \
  --dataset-path training-data/final/messages_reply_pairs.jsonl \
  --output-dir qwen-3-0.6b-sft/output
```

This saves a standard Transformers `save_pretrained` directory into the output folder.

## Hugging Face Upload

Install the extra Hub dependency in the same environment used for upload:

```bash
pip install huggingface_hub
```

Authenticate with either:

```bash
export HF_TOKEN=hf_...
```

or:

```bash
hf auth login
```

Then upload the trained model directory:

```bash
python qwen-3-0.6b-sft/upload_to_huggingface.py \
  --repo-id your-namespace/your-model-name \
  --model-dir qwen-3-0.6b-sft/output
```

Defaults:

- creates the Hub repo if it does not already exist
- uses a private model repo unless `--public` is passed
- uploads the full saved-model folder
- generates a minimal `README.md` model card unless `--no-model-card` is passed

Use `--readme-path path/to/README.md` to upload a custom model card instead of the generated one.
