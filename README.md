# imsg-nanochat

Reusable local pipeline for preparing personal texting data for nanochat fine-tuning.

The goal of this repository is to make the local setup ergonomic: collect Apple Messages data, clean and review it, export nanochat-compatible JSONL datasets, and keep evaluation prompts and project plans in one place. Actual model training is still expected to happen later on a GPU machine.

This repository is the top-level project workspace. The `nanochat/` directory is tracked as a Git submodule so project-specific tooling can live here while model/runtime changes can be managed separately in the nanochat fork.

## Current Status
- The workspace structure is in place.
- The Apple Messages to nanochat converter is implemented at `training-data/scripts/build_nanochat_jsonl.py` and covered by focused tests in `training-data/tests/test_build_nanochat_jsonl.py`.
- The wrapper project currently supports preparing reply-only, contact-agnostic nanochat JSONL from `chat.db` as strict two-message samples containing raw inbound text and reply text.
- The extractor now loads personal defaults from a local `training-data/config/extraction_config.json`, with setup starting from the tracked `training-data/config/extraction_config.example.json`.
- Specific one-to-one chats can be fully ignored through `excluded_contact_labels`, which remains a local extraction-only control.
- The extractor now uses a precision-first `attributedBody` fallback: clean human text is kept, but noisy Apple archive payloads are dropped instead of guessed.
- Wrapper-repo tradeoffs and revisit triggers are now recorded in `decision_log.md`.
- Downstream model training, evaluation artifacts, and the future frontend are still pending in this wrapper repo.

## Cloning
Clone recursively so `nanochat/` is populated:

```bash
git clone --recurse-submodules https://github.com/adamya-singh/imsg-nanochat.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Workflow
1. Export or copy local message data into `training-data/raw/`.
2. Copy `training-data/config/extraction_config.example.json` to `training-data/config/extraction_config.json`, then optionally edit it to tune extraction defaults or exclude specific chats with `excluded_contact_labels`.
3. Run the implemented converter in `training-data/scripts/build_nanochat_jsonl.py` to produce reply-only, contact-agnostic nanochat JSONL from `chat.db`.
4. Review outputs and privacy-sensitive samples, including any extraction stats printed by the converter. This is a manual workflow today, not a finished review toolchain.
5. Export final nanochat-compatible JSONL files into `training-data/final/`. The directory exists, but no checked-in dataset artifact is included in this repo.
6. Move the prepared dataset and the `nanochat/` codebase to a later training flow on GPU infrastructure. That downstream loop is still planned, not completed here.

## Current Layout
- `project_plan.md`: high-level roadmap and project vision
- `decision_log.md`: concrete wrapper-repo tradeoffs, decisions, and revisit triggers
- `training-data/`: raw exports, config files, intermediate outputs, review artifacts, final JSONL datasets, prep scripts, and the canonical extraction docs
- `nanochat/`: nanochat fork tracked as a Git submodule for training and inference experimentation
- `evaluation-prompts/`: fixed prompts and notes for checkpoint evaluation
- `nextjs-frontend/`: reserved for a later lightweight interface
