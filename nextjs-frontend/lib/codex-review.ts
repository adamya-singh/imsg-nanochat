import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { getDb } from "@/lib/db";
import { CODEX_REVIEW_DIR, REPO_ROOT } from "@/lib/paths";
import type { CodexReviewJob } from "@/lib/types";

const RUBRIC_VERSION = "training-quality-v1";
export const DEFAULT_CODEX_REVIEW_MODEL = "gpt-5.4-mini";

function mapJob(row: Record<string, unknown>): CodexReviewJob {
  return {
    id: String(row.id),
    runId: String(row.run_id),
    scope: row.scope as "pilot" | "full",
    status: row.status as CodexReviewJob["status"],
    rubricVersion: String(row.rubric_version),
    batchSize: Number(row.batch_size),
    totalCount: Number(row.total_count),
    processedCount: Number(row.processed_count),
    includeCount: Number(row.include_count),
    excludeCount: Number(row.exclude_count),
    unsureCount: Number(row.unsure_count),
    retryCount: Number(row.retry_count),
    currentBatch: Number(row.current_batch),
    model: String(row.model),
    cliVersion: row.cli_version ? String(row.cli_version) : null,
    error: row.error ? String(row.error) : null,
    createdAt: String(row.created_at),
    startedAt: row.started_at ? String(row.started_at) : null,
    completedAt: row.completed_at ? String(row.completed_at) : null,
    updatedAt: String(row.updated_at),
  };
}

export function listCodexReviewJobs(runId: string): CodexReviewJob[] {
  const staleBefore = new Date(Date.now() - 15 * 60_000).toISOString();
  getDb()
    .prepare(
      "UPDATE codex_review_jobs SET status='paused',error='Worker heartbeat expired; resume to continue.',updated_at=? WHERE run_id=? AND status='running' AND updated_at<?",
    )
    .run(new Date().toISOString(), runId, staleBefore);
  return (
    getDb()
      .prepare(
        "SELECT * FROM codex_review_jobs WHERE run_id=? ORDER BY created_at DESC",
      )
      .all(runId) as Record<string, unknown>[]
  ).map(mapJob);
}

export function getCodexReviewJob(id: string): CodexReviewJob | null {
  const row = getDb()
    .prepare("SELECT * FROM codex_review_jobs WHERE id=?")
    .get(id) as Record<string, unknown> | undefined;
  return row ? mapJob(row) : null;
}

function spawnWorker(jobId: string): void {
  fs.mkdirSync(CODEX_REVIEW_DIR, { recursive: true });
  const script = path.join(
    REPO_ROOT,
    "nextjs-frontend",
    "scripts",
    "codex-review-worker.mjs",
  );
  const log = fs.openSync(path.join(CODEX_REVIEW_DIR, `${jobId}.log`), "a");
  const child = spawn(process.execPath, [script, jobId], {
    cwd: REPO_ROOT,
    detached: true,
    stdio: ["ignore", log, log],
    env: process.env,
  });
  child.unref();
  fs.closeSync(log);
}

export function createCodexReviewJob(input: {
  runId: string;
  scope: "pilot" | "full";
  model?: string;
}): CodexReviewJob {
  const db = getDb();
  const active = db
    .prepare(
      "SELECT id FROM codex_review_jobs WHERE run_id=? AND status IN ('queued','running')",
    )
    .get(input.runId);
  if (active)
    throw new Error("A Codex review job is already active for this run");
  const model = input.model?.trim() || DEFAULT_CODEX_REVIEW_MODEL;
  if (!/^[a-zA-Z0-9._-]+$/.test(model))
    throw new Error("Model must be a valid model slug");
  const missing = Number(
    (
      db
        .prepare(
          `SELECT COUNT(*) count FROM examples e WHERE e.run_id=? AND NOT EXISTS (SELECT 1 FROM codex_reviews cr WHERE cr.example_id=e.id AND cr.content_hash=e.content_hash AND cr.rubric_version=?)`,
        )
        .get(input.runId, RUBRIC_VERSION) as { count: number }
    ).count,
  );
  const total = input.scope === "pilot" ? Math.min(100, missing) : missing;
  if (!total)
    throw new Error("Every current example already has a Codex review");
  const id = crypto.randomUUID(),
    stamp = new Date().toISOString();
  db.prepare(
    `INSERT INTO codex_review_jobs(id,run_id,scope,status,rubric_version,batch_size,total_count,model,created_at,updated_at) VALUES (?,?,?,'queued',?,50,?,?,?,?)`,
  ).run(
    id,
    input.runId,
    input.scope,
    RUBRIC_VERSION,
    total,
    model,
    stamp,
    stamp,
  );
  spawnWorker(id);
  return getCodexReviewJob(id)!;
}

export function controlCodexReviewJob(
  id: string,
  action: "pause" | "resume" | "cancel" | "retry",
): CodexReviewJob {
  const job = getCodexReviewJob(id);
  if (!job) throw new Error("Review job not found");
  const stamp = new Date().toISOString();
  if (action === "pause") {
    if (!["queued", "running"].includes(job.status))
      throw new Error("Only an active job can be paused");
    getDb()
      .prepare(
        "UPDATE codex_review_jobs SET status='paused',updated_at=? WHERE id=?",
      )
      .run(stamp, id);
  } else if (action === "cancel") {
    if (["completed", "cancelled"].includes(job.status))
      throw new Error("Job is already finished");
    getDb()
      .prepare(
        "UPDATE codex_review_jobs SET status='cancelled',completed_at=?,updated_at=? WHERE id=?",
      )
      .run(stamp, stamp, id);
  } else {
    if (!["paused", "failed"].includes(job.status))
      throw new Error("Only a paused or failed job can resume");
    getDb()
      .prepare(
        "UPDATE codex_review_jobs SET status='queued',error=NULL,completed_at=NULL,updated_at=? WHERE id=?",
      )
      .run(stamp, id);
    spawnWorker(id);
  }
  return getCodexReviewJob(id)!;
}
