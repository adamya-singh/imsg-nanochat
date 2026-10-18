import { spawn, type ChildProcess } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { getDb } from "@/lib/db";
import {
  EXTRACTOR,
  EXTRACTION_CONFIG,
  RAW_CHAT_DB,
  REPO_ROOT,
  RUN_ARTIFACTS_DIR,
} from "@/lib/paths";
import { importRun, listRuns } from "@/lib/workspace";

type Job = { process: ChildProcess; runId: string; cancelled: boolean };
declare global {
  var __nanochatJobs: Map<string, Job> | undefined;
}
const jobs = globalThis.__nanochatJobs ?? new Map<string, Job>();
globalThis.__nanochatJobs = jobs;

export type ExtractionInput = {
  name?: string;
  dbPath?: string;
  minContactPairs?: number;
  mergeGapSeconds?: number;
  seed?: number;
  limitChats?: number | null;
  parentRunId?: string | null;
};

export function startExtraction(input: ExtractionInput): string {
  const runId = `run-${new Date()
    .toISOString()
    .replace(/[-:.TZ]/g, "")
    .slice(0, 14)}-${crypto.randomBytes(3).toString("hex")}`;
  const directory = path.join(RUN_ARTIFACTS_DIR, runId);
  fs.mkdirSync(directory, { recursive: true });
  const dbPath = path.resolve(input.dbPath ?? RAW_CHAT_DB);
  if (!fs.existsSync(dbPath) || path.extname(dbPath) !== ".db")
    throw new Error("Select an existing .db source file");
  const jsonl = path.join(directory, "dataset.jsonl"),
    provenance = path.join(directory, "provenance.jsonl"),
    stats = path.join(directory, "stats.json");
  const config = {
    minContactPairs: input.minContactPairs ?? 5,
    mergeGapSeconds: input.mergeGapSeconds ?? 600,
    seed: input.seed ?? 42,
    limitChats: input.limitChats ?? null,
    dbPath,
  };
  getDb()
    .prepare(
      "INSERT INTO runs(id,name,status,created_at,parent_run_id,config_json,stats_json) VALUES (?,?,?,?,?,?,?)",
    )
    .run(
      runId,
      input.name?.trim() || `Extraction ${new Date().toLocaleString()}`,
      "processing",
      new Date().toISOString(),
      input.parentRunId ??
        listRuns().find((r) => r.status === "completed")?.id ??
        null,
      JSON.stringify(config),
      "{}",
    );
  const args = [
    EXTRACTOR,
    "--db-path",
    dbPath,
    "--output-path",
    jsonl,
    "--provenance-output-path",
    provenance,
    "--stats-output-path",
    stats,
    "--config-path",
    EXTRACTION_CONFIG,
    "--min-contact-pairs",
    String(config.minContactPairs),
    "--merge-gap-seconds",
    String(config.mergeGapSeconds),
    "--seed",
    String(config.seed),
  ];
  if (config.limitChats != null)
    args.push("--limit-chats", String(config.limitChats));
  const child = spawn("python3", args, {
    cwd: REPO_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const job = { process: child, runId, cancelled: false };
  jobs.set(runId, job);
  let stderr = "";
  child.stderr?.on("data", (chunk) => {
    stderr += String(chunk).slice(-8000);
  });
  child.on("exit", (code) => {
    jobs.delete(runId);
    if (job.cancelled) {
      getDb()
        .prepare("UPDATE runs SET status='cancelled',completed_at=? WHERE id=?")
        .run(new Date().toISOString(), runId);
      fs.rmSync(directory, { recursive: true, force: true });
      return;
    }
    if (code !== 0) {
      getDb()
        .prepare(
          "UPDATE runs SET status='failed',completed_at=?,error=? WHERE id=?",
        )
        .run(
          new Date().toISOString(),
          stderr || `Extractor exited with ${code}`,
          runId,
        );
      return;
    }
    try {
      const statsPayload = JSON.parse(fs.readFileSync(stats, "utf8"));
      importRun({
        id: runId,
        name: input.name?.trim() || `Extraction ${runId}`,
        jsonlPath: jsonl,
        provenancePath: provenance,
        config,
        stats: statsPayload,
        parentRunId: input.parentRunId ?? null,
      });
    } catch (error) {
      getDb()
        .prepare(
          "UPDATE runs SET status='failed',completed_at=?,error=? WHERE id=?",
        )
        .run(new Date().toISOString(), (error as Error).message, runId);
    }
  });
  return runId;
}

export function cancelExtraction(runId: string): boolean {
  const job = jobs.get(runId);
  if (!job) return false;
  job.cancelled = true;
  job.process.kill("SIGTERM");
  return true;
}
