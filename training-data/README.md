# training-data

Workspace for data extraction, cleaning, review, and final training artifacts.

## Intended Structure
- `raw/`: copied exports such as `chat.db` or safe derived snapshots
- `intermediate/`: parsed or cleaned outputs that are not final training data
- `review/`: manual review artifacts, notes, and sampling outputs
- `final/`: nanochat-compatible JSONL datasets used for training
- `scripts/`: extraction and cleaning utilities

## `chat.db` Conversion
Use the v1 converter to build reply-only nanochat JSONL directly from Apple Messages:

```bash
python training-data/scripts/build_nanochat_jsonl.py \
  --db-path training-data/raw/chat.db \
  --output-path training-data/final/messages_reply_pairs.jsonl
```

Optional flags:
- `--min-contact-pairs 5`
- `--merge-gap-seconds 600`
- `--seed 42`
- `--limit-chats <int>`

## V1 Rules
- Reads Apple Messages `chat.db` and only keeps one-to-one chats.
- Emits strict 2-message nanochat samples: inbound contact turn as `user`, Adamya reply as `assistant`.
- Formats prompts as:

```json
[
  {
    "role": "user",
    "content": "[MODE: REPLY]\n[CONTACT: <chat_db_handle>]\n<incoming_text>"
  },
  {
    "role": "assistant",
    "content": "<reply_text>"
  }
]
```

- Uses `message.text` first, then best-effort plain-text recovery from `attributedBody`.
- Drops group chats, media-only rows, reactions, message effects, and obvious OTP/service noise.
- Merges consecutive same-speaker texts within 10 minutes into one turn before pairing.
- Excludes contacts with fewer than 5 usable reply pairs after filtering.
