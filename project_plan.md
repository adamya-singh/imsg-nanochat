# Adamya Clone with Nanochat

## Summary
Create a single repository that serves as the long-lived home for the project: planning, raw and processed training data, a local `nanochat/` checkout, evaluation artifacts, and a later `nextjs-frontend/` interface. The first implementation focus is the training pipeline, not the frontend: extract Messages data, convert it into a clean nanochat-compatible dataset, fine-tune a small chat model, and evaluate whether it captures both general style and contact-specific behavior.

This document stays high level and vision-oriented. Detailed step-by-step implementation plans should be created later for each phase, especially data preparation, training, evaluation, and frontend work.

## Progress So Far
- Phase 1 is complete: this repo now acts as the wrapper workspace for planning, data prep, evaluation prompts, and the nested `nanochat/` codebase.
- Phase 2 is partially complete: a working `chat.db` to nanochat JSONL converter exists at `training-data/scripts/build_nanochat_jsonl.py`, it has focused tests, and it now supports configurable extraction defaults through `DEFAULT_CONFIG`.
- The currently implemented data path is reply-only. It emits strict two-message nanochat samples with `[MODE: REPLY]` and `[CONTACT: ...]`.
- Specific one-to-one chats can now be fully excluded by exact `contact_label` match through `excluded_contact_labels`.
- Later phases remain pending in this wrapper repo: broader privacy/review workflows, richer dataset assembly, cloud training runs, checkpoint evaluation artifacts, and the future frontend.

## Key Changes
- Establish this repo as the canonical project workspace with these top-level responsibilities:
  - `training-data/` for exported raw data, cleaned/intermediate datasets, redaction utilities, and final JSONL outputs
  - `nanochat/` as the local cloned training/inference codebase used for SFT experiments
  - `nextjs-frontend/` reserved for a later lightweight UI to interact with trained checkpoints
  - `project_plan.md` as the stable project vision and roadmap
- Treat the model objective as two related behaviors within one assistant:
  - `REPLY` mode for normal inbound-text to outbound-reply behavior
  - `SELF_NOTE` mode for private note-to-self behavior learned separately rather than mixed into ordinary reply turns
- Keep nanochat’s standard chat schema:
  - `user` always represents the incoming speaker or task prompt
  - `assistant` always represents Adamya
  - contact identity is encoded in message content rather than custom roles
- Define the dataset contract early so all later plans align to it:
  - reply examples include `[MODE: REPLY]` and `[CONTACT: ...]`
  - self-note examples include `[MODE: SELF_NOTE]` once that path is implemented
  - all training data is emitted as nanochat-style JSONL conversation samples
- Set the first training scope to private, high-signal 1:1 conversations only:
  - merge nearby texts into turns
  - filter system/spam/OTP/automation noise
  - allow explicit per-chat exclusion through editable extraction config
  - use minimal redaction at first and leave broader privacy scrubbing/manual review for a later dedicated pass
  - prevent single contacts or self-notes from dominating the dataset
- Plan the workflow in phases rather than one monolithic build:
  - Phase 1: repo setup and data handling conventions. Status: complete.
  - Phase 2: extraction and cleaning pipeline from `chat.db`. Status: partially complete through the reply-only converter, configurable extraction defaults, and explicit per-chat exclusion.
  - Phase 3: dataset assembly and balancing. Status: pending.
  - Phase 4: nanochat integration and cloud GPU training. Status: pending.
  - Phase 5: checkpoint evaluation and prompt/interface design. Status: pending.
  - Phase 6: frontend integration later. Status: pending.

## Public Interfaces / Data Contracts
- Primary training artifact: one or more JSONL files in nanochat-compatible conversation format
- Input source: exported Apple Messages `chat.db`
- Core sample shapes:
  - reply sample: `user` content includes mode and contact header plus incoming text; `assistant` content is Adamya’s reply. This shape is implemented today.
  - self-note sample: `user` content is a self-note instruction/header; `assistant` content is the private note text. This remains planned.
- Evaluation inputs should mirror the same mode-based prompting scheme so training and testing stay aligned

### Reply Sample
```json
[
  {
    "role": "user",
    "content": "[MODE: REPLY]\n[CONTACT: Mahi]\nyou free later?"
  },
  {
    "role": "assistant",
    "content": "yea probably after 7"
  }
]
```

### Self-Note Sample
```json
[
  {
    "role": "user",
    "content": "[MODE: SELF_NOTE]\nWrite a quick private reminder to yourself."
  },
  {
    "role": "assistant",
    "content": "remember to fix openclaw agent"
  }
]
```

## Test Plan
- Validate extraction on a small known subset of conversations before full export
- Check that session splitting and turn merging produce realistic conversational samples
- Confirm junk filtering removes OTPs, service alerts, and obvious automation noise without deleting real conversations
- Review dataset balance by contact and by mode so the model is not dominated by one person or by self-notes
- Run post-training evaluation against a fixed prompt set that checks:
  - overall resemblance to Adamya’s texting tone
  - contact-specific variation
  - clean separation between `REPLY` and `SELF_NOTE` behavior
  - absence of obvious leakage from junk/system text patterns

## Assumptions
- `project_plan.md` is a high-level roadmap, not an implementation spec
- Smaller decision-complete plans will be written later for each major phase
- `nanochat/` will live inside this repository as a sibling to `training-data/` and `nextjs-frontend/`
- Training will happen on a cloud GPU environment, not local low-power hardware
- The frontend is intentionally deferred until the data/training loop is working
- First-pass privacy handling will use minimal redaction plus manual review rather than aggressive automatic scrubbing
