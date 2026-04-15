# Remote Notes

## 2026-04-15 Hugging Face Upload

Command used from `/freespace/local/as4487/imsg-nanochat/qwen-3-0.6b-sft`:

```bash
python upload_to_huggingface.py \
  --repo-id adamyathegreat/qwen-3-0.6b-my-texts-finetune \
  --model-dir output-pilot \
  --dataset-path ../training-data/final/messages_reply_pairs.jsonl
```

Uploaded to:

- Hugging Face repo: `adamyathegreat/qwen-3-0.6b-my-texts-finetune`
- URL: [https://huggingface.co/adamyathegreat/qwen-3-0.6b-my-texts-finetune](https://huggingface.co/adamyathegreat/qwen-3-0.6b-my-texts-finetune)
- Visibility at upload time: `private`

Observed upload result:

- Upload completed successfully.
- The generated `README.md` model card was uploaded after the model files.
