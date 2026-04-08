# Decision Log

Canonical memory for important wrapper-repo tradeoffs, deferred choices, and revisit triggers.

## How To Use This Doc

- Use this file for non-obvious project-level decisions that future work should remember.
- Keep entries concise and implementation-relevant.
- Do not use this file for routine progress notes, command transcripts, or `nanochat/` submodule experiment history.
- Prefer one entry per decision. Update the existing entry when the decision changes or is revisited.
- Future plans should link back to relevant entries here when they depend on prior tradeoffs.

---

## 2026-04-08: Prefer precision over recall for `attributedBody` extraction

- Date: 2026-04-08
- Decision: The `chat.db` converter should only keep `attributedBody` text when it can isolate clean human-readable content with high confidence. Uncertain `attributedBody` rows should be dropped.
- Context: The first permissive export produced roughly 56k reply pairs, but many rows were polluted by Apple archive metadata such as `streamtyped`, `NSAttributedString`, `__kIM...`, and `bplist00`. This made the dataset larger but not training-safe.
- Why this choice won: Training-quality text matters more than maximizing row count. The precision-first rerun removed the archive markers entirely and produced a clean dataset of roughly 12.7k reply pairs.
- Cost / downside: The dataset became much smaller, and some recoverable `attributedBody`-only rows were intentionally lost.
- Revisit when: Revisit this only if we add a more specialized Apple attributed-string/archive parser that can safely recover additional rows without leaking serialization metadata.
- Related files or artifacts: `training-data/scripts/build_nanochat_jsonl.py`, `training-data/tests/test_build_nanochat_jsonl.py`, `training-data/final/messages_reply_pairs.jsonl`
