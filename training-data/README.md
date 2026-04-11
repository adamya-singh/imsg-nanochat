# training-data

Workspace for data extraction, cleaning, review, and final training artifacts.

## Current Progress
The wrapper repo currently has one implemented data-preparation path: `training-data/scripts/build_nanochat_jsonl.py` converts Apple Messages `chat.db` into reply-only, contact-agnostic nanochat JSONL. The extractor is driven by a local JSON config file, seeded from a checked-in example config, supports explicit per-chat exclusion through `excluded_contact_labels`, and now uses a precision-first `attributedBody` fallback.

## Implemented Today
The current implemented pipeline is `training-data/scripts/build_nanochat_jsonl.py`, which converts Apple Messages `chat.db` into reply-only, contact-agnostic nanochat JSONL.

The emitted dataset uses strict two-message samples:
- `user`: raw inbound message turn
- `assistant`: Adamya's direct reply turn

This README is the source of truth for the currently implemented extraction interface and behavior in this wrapper repo.

## Intended Structure
- `raw/`: copied exports such as `chat.db` or safe derived snapshots
- `config/`: extractor configuration files, including a checked-in example and a local ignored config
- `intermediate/`: parsed or cleaned outputs that are not final training data
- `review/`: manual review artifacts, notes, and sampling outputs
- `final/`: nanochat-compatible JSONL datasets used for training
- `scripts/`: extraction and cleaning utilities

## `chat.db` Conversion
Copy `training-data/config/extraction_config.example.json` to `training-data/config/extraction_config.json`, then edit the local JSON file to set personal extraction defaults. The example file is tracked; your personal `extraction_config.json` is intended to stay local and ignored by git.

Current configurable values include:
- `min_contact_pairs`
- `merge_gap_seconds`
- `seed`
- `limit_chats`
- `excluded_contact_labels`

`excluded_contact_labels` uses exact string matching against the same internal `contact_label` the script derives while scanning one-to-one chats. This is intended for fully ignoring specific one-to-one chats during extraction, even though contact identity is not emitted into the final training samples.

Use the v1 converter to build reply-only nanochat JSONL directly from Apple Messages:

```bash
cp training-data/config/extraction_config.example.json training-data/config/extraction_config.json

python training-data/scripts/build_nanochat_jsonl.py \
  --config-path training-data/config/extraction_config.json \
  --db-path training-data/raw/chat.db \
  --output-path training-data/final/messages_reply_pairs.jsonl
```

Optional flags:
- `--config-path training-data/config/extraction_config.json`
- `--min-contact-pairs 5`
- `--merge-gap-seconds 600`
- `--seed 42`
- `--limit-chats <int>`

If `--config-path` is omitted, the script loads `training-data/config/extraction_config.json` by default. If that file is missing, the script tells you to copy the example config first.

CLI flags override the values set in the loaded config file for that run. `excluded_contact_labels` remains controlled through the JSON config file.

## V1 Rules
- Reads Apple Messages `chat.db` and only keeps one-to-one chats.
- Emits strict 2-message nanochat samples: inbound message turn as `user`, Adamya reply as `assistant`.
- Formats prompts as:

```json
[
  {
    "role": "user",
    "content": "<incoming_text>"
  },
  {
    "role": "assistant",
    "content": "<reply_text>"
  }
]
```

- Uses `message.text` first, then best-effort plain-text recovery from `attributedBody`.
- Favors precision over recall when `message.text` is empty:
  - `message.text` is preferred whenever present.
  - `attributedBody` is only used when the script can isolate clean human-readable text with high confidence.
  - uncertain `attributedBody` rows are dropped instead of guessed.
- Drops group chats, media-only rows, reactions, message effects, and obvious OTP/service noise.
- Merges consecutive same-speaker texts within 10 minutes into one turn before pairing.
- Dedupes identical final reply pairs deterministically before writing JSONL.
- Excludes contacts with fewer than 5 usable reply pairs after filtering.

## Verified Behavior
The current script and tests verify the following behavior:
- Only one-to-one chats are considered.
- Same-speaker messages are merged when they fall within `--merge-gap-seconds`.
- `attributedBody` is used as a best-effort text fallback when `message.text` is empty.
- `attributedBody` recovery is precision-first: the extractor rejects Apple archive metadata such as `streamtyped`, `NSAttributedString`, `__kIM...`, and similar serialization noise instead of passing it through.
- Obvious OTP, verification, service, and other junk-text patterns are excluded.
- Reactions, message effects, and media-only rows without ordinary text are excluded.
- Identical final reply pairs are deduped.
- Contacts below the configured minimum usable pair threshold are dropped.
- Configured `excluded_contact_labels` fully remove matching one-to-one chats from extraction.
- `--config-path` can switch the script to an alternate JSON config file.
- CLI values override loaded config values for that invocation.
- The converter prints extraction stats after each run, including rows using `message.text`, rows using clean `attributedBody`, low-confidence `attributedBody` drops, reaction/effect drops, media-only drops, empty-text drops, and junk-pair drops.
- The generated JSONL loads cleanly through nanochat's `CustomJSON` task format.
- Contact identity is kept internal to extraction for exclusion and minimum-history filtering, but is not present in emitted JSONL.

## Notes On Output Size
Row counts may decrease after extractor cleanup. This is expected when the script prefers clean training text over maximum recovery from noisy `attributedBody` blobs. If the dataset becomes materially smaller after a cleanup change, check `decision_log.md` and the printed extraction stats before assuming the run regressed.

## Testing
Run the focused converter tests from the repo root:

```bash
python -m pytest training-data/tests/test_build_nanochat_jsonl.py -q
```

## Not Implemented Yet
- `SELF_NOTE` dataset generation is still planned and is not part of this converter.
- Broader privacy scrubbing and richer redaction passes are not yet automated here.
- Dataset balancing beyond the current per-contact minimum threshold is still future work.
- Contact-specific conditioning is not part of the current v1 training artifact; any return of contact identity is future work.
- Direct wiring from `training-data/final/*.jsonl` into a local nanochat SFT workflow is not yet implemented in this wrapper project.
- Checkpoint evaluation artifacts and the future frontend are still outside the implemented extraction pipeline.
