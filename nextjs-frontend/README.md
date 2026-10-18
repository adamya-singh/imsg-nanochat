# Nanochat Dataset Studio

A private, localhost-only workspace for extracting, reviewing, transforming, balancing, splitting, and exporting nanochat training data.

Requires Node.js 22.5 or newer because the local workspace uses Node's built-in SQLite driver.

## Run locally

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. On first load, the app imports `training-data/final/messages_reply_pairs.jsonl` into the ignored workspace database at `training-data/review/curation.sqlite`.

The source JSONL and `chat.db` are never edited. Exports are written atomically to `training-data/final/runs/<run-id>/`.

## Workflow

1. **Extract** a versioned snapshot from `chat.db`, or use the automatically imported baseline JSONL.
2. **Curate** examples from the overview with search, queues, tags, notes, and review states.
3. **Transform** selected or filtered examples with previewable, reversible rules.
4. **Balance** opaque source groups with deterministic caps.
5. **Split** by source session into deterministic train, validation, and test sets.
6. **Export** nanochat JSONL plus a non-private reproducibility manifest.

## Codex-assisted review

Open **AI Review** to run a 100-example pilot or review every remaining example. Jobs use the authenticated Codex CLI, default to `gpt-5.4-mini`, and allow a different model slug per run. Each example receives an independent include, exclude, or unsure decision with confidence and rationale.

Codex reviews never overwrite human status. Human approved, flagged, and excluded decisions take precedence; an unreviewed example is omitted from downstream output only when its current, non-stale Codex decision is exclude. Starting a job sends the selected message text to OpenAI.

All source IDs and timestamps stay in the local SQLite workspace and are never written into training JSONL.
