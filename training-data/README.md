# training-data

Workspace for data extraction, cleaning, review, and final training artifacts.

## Implemented Today
The current implemented pipeline is `training-data/scripts/build_nanochat_jsonl.py`, which converts Apple Messages `chat.db` into reply-only nanochat JSONL.

The emitted dataset uses strict two-message samples:
- `user`: inbound contact turn prefixed with `[MODE: REPLY]` and `[CONTACT: ...]`
- `assistant`: Adamya's direct reply turn

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
- Dedupes identical final reply pairs deterministically before writing JSONL.
- Excludes contacts with fewer than 5 usable reply pairs after filtering.

## Verified Behavior
The current script and tests verify the following behavior:
- Only one-to-one chats are considered.
- Same-speaker messages are merged when they fall within `--merge-gap-seconds`.
- `attributedBody` is used as a best-effort text fallback when `message.text` is empty.
- Obvious OTP, verification, service, and other junk-text patterns are excluded.
- Reactions, message effects, and media-only rows without ordinary text are excluded.
- Identical final reply pairs are deduped.
- Contacts below the configured minimum usable pair threshold are dropped.
- The generated JSONL loads cleanly through nanochat's `CustomJSON` task format.

## Testing
Run the focused converter tests from the repo root:

```bash
python -m pytest training-data/tests/test_build_nanochat_jsonl.py -q
```

## Not Implemented Yet
- `SELF_NOTE` dataset generation is still planned and is not part of this converter.
- Broader privacy scrubbing and richer redaction passes are not yet automated here.
- Dataset balancing beyond the current per-contact minimum threshold is still future work.
- Direct wiring from `training-data/final/*.jsonl` into a local nanochat SFT workflow is not yet implemented in this wrapper project.
