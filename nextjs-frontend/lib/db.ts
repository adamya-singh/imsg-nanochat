import { DatabaseSync } from "node:sqlite";
import fs from "node:fs";
import { WORKSPACE_DB, REVIEW_DIR } from "@/lib/paths";

declare global {
  var __nanochatDb: StudioDatabase | undefined;
}

type StudioDatabase = DatabaseSync & {
  transaction<T>(fn: () => T): () => T;
  pragma(value: string): void;
};

const schema = `
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('processing','completed','failed','cancelled')),
  created_at TEXT NOT NULL,
  completed_at TEXT,
  parent_run_id TEXT,
  config_json TEXT NOT NULL DEFAULT '{}',
  stats_json TEXT NOT NULL DEFAULT '{}',
  source_jsonl_path TEXT,
  provenance_path TEXT,
  error TEXT,
  reconciliation_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS examples (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  source_key TEXT NOT NULL,
  original_user TEXT NOT NULL,
  original_assistant TEXT NOT NULL,
  current_user TEXT NOT NULL,
  current_assistant TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unreviewed' CHECK(status IN ('unreviewed','approved','flagged','excluded')),
  tags_json TEXT NOT NULL DEFAULT '[]',
  notes TEXT NOT NULL DEFAULT '',
  version INTEGER NOT NULL DEFAULT 1,
  chat_id INTEGER,
  inbound_ids_json TEXT NOT NULL DEFAULT '[]',
  reply_ids_json TEXT NOT NULL DEFAULT '[]',
  start_timestamp REAL,
  end_timestamp REAL,
  session_key TEXT NOT NULL,
  split TEXT CHECK(split IN ('train','validation','test') OR split IS NULL),
  excluded_by_balance INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(run_id, source_key)
);

CREATE INDEX IF NOT EXISTS examples_run_idx ON examples(run_id);
CREATE INDEX IF NOT EXISTS examples_status_idx ON examples(run_id, status);
CREATE INDEX IF NOT EXISTS examples_split_idx ON examples(run_id, split);
CREATE INDEX IF NOT EXISTS examples_source_idx ON examples(run_id, chat_id);
CREATE INDEX IF NOT EXISTS examples_hash_idx ON examples(run_id, content_hash);
CREATE INDEX IF NOT EXISTS examples_updated_idx ON examples(run_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS examples_fts (
  example_id TEXT PRIMARY KEY,
  user_text TEXT NOT NULL,
  assistant_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS examples_search_id_idx ON examples_fts(example_id);

CREATE TABLE IF NOT EXISTS example_terms (
  example_id TEXT NOT NULL,
  term TEXT COLLATE NOCASE NOT NULL,
  PRIMARY KEY(example_id, term)
);
CREATE INDEX IF NOT EXISTS example_terms_lookup_idx ON example_terms(term, example_id);

CREATE TABLE IF NOT EXISTS operations (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TEXT NOT NULL,
  reverted_at TEXT
);

CREATE TABLE IF NOT EXISTS operation_changes (
  operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
  example_id TEXT NOT NULL REFERENCES examples(id) ON DELETE CASCADE,
  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  PRIMARY KEY(operation_id, example_id)
);

CREATE TABLE IF NOT EXISTS recipes (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  position INTEGER NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  spec_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exports (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS codex_review_jobs (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  scope TEXT NOT NULL CHECK(scope IN ('pilot','full')),
  status TEXT NOT NULL CHECK(status IN ('queued','running','paused','completed','failed','cancelled')),
  rubric_version TEXT NOT NULL,
  batch_size INTEGER NOT NULL DEFAULT 50,
  total_count INTEGER NOT NULL,
  processed_count INTEGER NOT NULL DEFAULT 0,
  include_count INTEGER NOT NULL DEFAULT 0,
  exclude_count INTEGER NOT NULL DEFAULT 0,
  unsure_count INTEGER NOT NULL DEFAULT 0,
  retry_count INTEGER NOT NULL DEFAULT 0,
  current_batch INTEGER NOT NULL DEFAULT 0,
  model TEXT NOT NULL DEFAULT 'configured-default',
  cli_version TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS codex_jobs_run_idx ON codex_review_jobs(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS codex_reviews (
  example_id TEXT PRIMARY KEY REFERENCES examples(id) ON DELETE CASCADE,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  content_hash TEXT NOT NULL,
  rubric_version TEXT NOT NULL,
  decision TEXT NOT NULL CHECK(decision IN ('include','exclude','unsure')),
  confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
  reasons_json TEXT NOT NULL DEFAULT '[]',
  rationale TEXT NOT NULL,
  job_id TEXT NOT NULL REFERENCES codex_review_jobs(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  reviewed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS codex_reviews_run_decision_idx ON codex_reviews(run_id, decision);
CREATE INDEX IF NOT EXISTS codex_reviews_hash_idx ON codex_reviews(content_hash);
`;

export function getDb(): StudioDatabase {
  if (globalThis.__nanochatDb) return globalThis.__nanochatDb;
  fs.mkdirSync(REVIEW_DIR, { recursive: true });
  const db = new DatabaseSync(WORKSPACE_DB) as StudioDatabase;
  db.pragma = (value: string) => db.exec(`PRAGMA ${value}`);
  db.transaction =
    <T>(fn: () => T) =>
    () => {
      db.exec("BEGIN IMMEDIATE");
      try {
        const result = fn();
        db.exec("COMMIT");
        return result;
      } catch (error) {
        db.exec("ROLLBACK");
        throw error;
      }
    };
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.pragma("busy_timeout = 5000");
  db.exec(schema);
  globalThis.__nanochatDb = db;
  return db;
}
