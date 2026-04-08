# imsg-nanochat

Reusable local pipeline for preparing personal texting data for nanochat fine-tuning.

The goal of this repository is to make the local setup ergonomic: collect Apple Messages data, clean and review it, export nanochat-compatible JSONL datasets, and keep evaluation prompts and project plans in one place. Actual model training is still expected to happen later on a GPU machine.

This repository is the top-level project workspace. The `nanochat/` directory is tracked as a Git submodule so project-specific tooling can live here while model/runtime changes can be managed separately in the nanochat fork.

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
2. Run extraction and cleaning scripts from `training-data/scripts/`.
3. Review intermediate outputs and privacy-sensitive samples.
4. Export final nanochat-compatible JSONL files into `training-data/final/`.
5. Move the prepared dataset and the `nanochat/` codebase to a GPU node for training.

## Current Layout
- `project_plan.md`: high-level roadmap and project vision
- `training-data/`: raw exports, intermediate outputs, review artifacts, final JSONL datasets, and prep scripts
- `nanochat/`: nanochat fork tracked as a Git submodule for training and inference experimentation
- `evaluation-prompts/`: fixed prompts and notes for checkpoint evaluation
- `nextjs-frontend/`: reserved for a later lightweight interface
