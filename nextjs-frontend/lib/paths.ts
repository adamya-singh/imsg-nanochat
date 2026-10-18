import fs from "node:fs";
import path from "node:path";

function discoverRepoRoot(): string {
  const candidates = [process.cwd(), path.resolve(process.cwd(), "..")];
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, "training-data"))) return candidate;
  }
  throw new Error("Could not locate repository root containing training-data/");
}

export const REPO_ROOT = discoverRepoRoot();
export const TRAINING_DATA = path.join(REPO_ROOT, "training-data");
export const REVIEW_DIR = path.join(TRAINING_DATA, "review");
export const WORKSPACE_DB = path.join(REVIEW_DIR, "curation.sqlite");
export const BASELINE_JSONL = path.join(
  TRAINING_DATA,
  "final",
  "messages_reply_pairs.jsonl",
);
export const RAW_CHAT_DB = path.join(TRAINING_DATA, "raw", "chat.db");
export const EXTRACTOR = path.join(
  TRAINING_DATA,
  "scripts",
  "build_nanochat_jsonl.py",
);
export const EXTRACTION_CONFIG = path.join(
  TRAINING_DATA,
  "config",
  "extraction_config.json",
);
export const RUN_ARTIFACTS_DIR = path.join(REVIEW_DIR, "runs");
export const EXPORTS_DIR = path.join(TRAINING_DATA, "final", "runs");
export const CODEX_REVIEW_DIR = path.join(REVIEW_DIR, "codex-review");
export const CODEX_REVIEW_RUBRIC = path.join(
  REPO_ROOT,
  "nextjs-frontend",
  "review-rubrics",
  "training-quality-v1.md",
);
