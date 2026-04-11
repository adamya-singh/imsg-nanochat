# Decision Log

Canonical memory for important wrapper-repo tradeoffs, deferred choices, and revisit triggers.

## How To Use This Doc

- Use this file for non-obvious project-level decisions that future work should remember.
- Keep entries concise and implementation-relevant.
- Do not use this file for routine progress notes, command transcripts, or `nanochat/` submodule experiment history.
- Prefer one entry per decision. Update the existing entry when the decision changes or is revisited.
- Future plans should link back to relevant entries here when they depend on prior tradeoffs.

---

## 2026-04-10: Keep reply-only behavior out of the v1 message text

- Date: 2026-04-10
- Decision: The wrapper repo's v1 extracted JSONL should not include `[MODE: REPLY]` in `user.content`. Reply-only behavior remains a property of the dataset, not an in-band text prefix.
- Context: The initial extractor encoded reply mode directly in every training sample. With only one implemented data path, that prefix added schema noise without providing additional signal.
- Why this choice won: The first dataset version should be as close as possible to raw conversational text while the project is still reply-only. This keeps the training artifact simpler and makes extraction defects easier to spot.
- Cost / downside: If a later dataset version introduces `SELF_NOTE` or other modes, the project will need a fresh conditioning contract instead of reusing the old prefix scheme.
- Revisit when: Revisit this only if the repo gains more than one training mode and needs explicit in-band conditioning again.
- Related files or artifacts: `training-data/scripts/build_nanochat_jsonl.py`, `training-data/README.md`, `project_plan.md`

## 2026-04-10: Keep contact identity out of the v1 training contract

- Date: 2026-04-10
- Decision: The wrapper repo's v1 extracted JSONL should remain contact-agnostic. Contact identity is used only inside the extractor for one-to-one chat selection, `excluded_contact_labels`, and minimum-history filtering, and is not emitted into training samples.
- Context: The earlier sample contract included contact headers in the training text. That added setup burden because contact labels would need cleanup or name resolution before the first usable dataset version.
- Why this choice won: The first project version should optimize for getting a clean reply-only dataset with minimal manual metadata work. Local exclusions still cover the immediate privacy and curation need without making contact identity part of the training schema.
- Cost / downside: The v1 training artifact cannot directly support contact-conditioned prompting or contact-specific evaluation prompts.
- Revisit when: Revisit this only if a later dataset version intentionally introduces contact-conditioned behavior and there is a clear plan for stable contact naming or aliasing.
- Related files or artifacts: `training-data/scripts/build_nanochat_jsonl.py`, `training-data/README.md`, `project_plan.md`

## 2026-04-08: Prefer precision over recall for `attributedBody` extraction

- Date: 2026-04-08
- Decision: The `chat.db` converter should only keep `attributedBody` text when it can isolate clean human-readable content with high confidence. Uncertain `attributedBody` rows should be dropped.
- Context: The first permissive export produced roughly 56k reply pairs, but many rows were polluted by Apple archive metadata such as `streamtyped`, `NSAttributedString`, `__kIM...`, and `bplist00`. This made the dataset larger but not training-safe.
- Why this choice won: Training-quality text matters more than maximizing row count. The precision-first rerun removed the archive markers entirely and produced a clean dataset of roughly 12.7k reply pairs.
- Cost / downside: The dataset became much smaller, and some recoverable `attributedBody`-only rows were intentionally lost.
- Revisit when: Revisit this only if we add a more specialized Apple attributed-string/archive parser that can safely recover additional rows without leaking serialization metadata.
- Related files or artifacts: `training-data/scripts/build_nanochat_jsonl.py`, `training-data/tests/test_build_nanochat_jsonl.py`, `training-data/final/messages_reply_pairs.jsonl`
