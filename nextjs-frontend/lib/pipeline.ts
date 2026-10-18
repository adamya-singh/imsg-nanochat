import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { getDb } from "@/lib/db";
import { EXPORTS_DIR } from "@/lib/paths";
import {
  canonicalHash,
  EFFECTIVE_INCLUDE_SQL,
  getRun,
  hash,
  mapExample,
  reindexExample,
  whereForQuery,
} from "@/lib/workspace";
import type {
  BalanceConfig,
  ExportManifest,
  SplitConfig,
  TrainingExample,
  TransformationPreview,
  TransformationSpec,
} from "@/lib/types";

const now = () => new Date().toISOString();

type MutableState = Pick<
  TrainingExample,
  | "user"
  | "assistant"
  | "status"
  | "tags"
  | "notes"
  | "excludedByBalance"
  | "split"
>;
function stateOf(example: TrainingExample): MutableState {
  return {
    user: example.user,
    assistant: example.assistant,
    status: example.status,
    tags: example.tags,
    notes: example.notes,
    excludedByBalance: example.excludedByBalance,
    split: example.split,
  };
}

function transformExample(
  example: TrainingExample,
  spec: TransformationSpec,
): MutableState {
  const next = stateOf(example);
  if (spec.kind === "replace") {
    if (!spec.find) throw new Error("Find text is required");
    let pattern: RegExp;
    try {
      pattern = spec.regex
        ? new RegExp(spec.find, "gu")
        : new RegExp(spec.find.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gu");
    } catch (error) {
      throw new Error(
        `Invalid regular expression: ${(error as Error).message}`,
      );
    }
    if (spec.field !== "assistant")
      next.user = next.user.replace(pattern, spec.replace ?? "");
    if (spec.field !== "user")
      next.assistant = next.assistant.replace(pattern, spec.replace ?? "");
  } else if (spec.kind === "normalize") {
    const normalize = (value: string) =>
      value
        .replace(/\r\n?/g, "\n")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
    if (spec.field !== "assistant") next.user = normalize(next.user);
    if (spec.field !== "user") next.assistant = normalize(next.assistant);
  } else if (spec.kind === "set-status") {
    if (!spec.status) throw new Error("A review status is required");
    next.status = spec.status;
  } else if (spec.kind === "add-tag") {
    const tag = spec.tag?.trim();
    if (!tag) throw new Error("A tag is required");
    next.tags = [...new Set([...next.tags, tag])];
  }
  return next;
}

function selectedExamples(spec: TransformationSpec): TrainingExample[] {
  const where = whereForQuery(spec.query);
  const rows = getDb()
    .prepare(`SELECT e.* FROM examples e WHERE ${where.sql} ORDER BY e.rowid`)
    .all(...where.params) as Record<string, unknown>[];
  return rows.map(mapExample);
}

export function previewTransformation(
  spec: TransformationSpec,
): TransformationPreview {
  const examples = selectedExamples(spec);
  const changes: TransformationPreview["samples"] = [];
  let affectedCount = 0;
  if (spec.kind === "dedupe") {
    const seen = new Set<string>();
    for (const ex of examples) {
      const key = canonicalHash(ex.user, ex.assistant);
      if (seen.has(key)) {
        affectedCount++;
        if (changes.length < 8)
          changes.push({
            id: ex.id,
            before: stateOf(ex),
            after: { ...stateOf(ex), status: "excluded" },
          });
      } else seen.add(key);
    }
  } else {
    for (const ex of examples) {
      const next = transformExample(ex, spec);
      const before = stateOf(ex);
      if (JSON.stringify(before) !== JSON.stringify(next)) {
        affectedCount++;
        if (changes.length < 8)
          changes.push({ id: ex.id, before, after: next });
      }
    }
  }
  return {
    affectedCount,
    samples: changes,
    warning:
      affectedCount > 5000
        ? "This operation changes more than 5,000 examples. Review the sample carefully."
        : undefined,
  };
}

export function applyTransformation(spec: TransformationSpec): {
  operationId: string;
  affectedCount: number;
} {
  const examples = selectedExamples(spec);
  const db = getDb();
  const operationId = crypto.randomUUID();
  const stamp = now();
  let affectedCount = 0;
  const seen = new Set<string>();
  const transaction = db.transaction(() => {
    db.prepare(
      "INSERT INTO operations(id,run_id,kind,description,created_at) VALUES (?,?,?,?,?)",
    ).run(
      operationId,
      spec.query.runId,
      spec.kind,
      describeTransformation(spec),
      stamp,
    );
    const addChange = db.prepare(
      "INSERT INTO operation_changes(operation_id,example_id,before_json,after_json) VALUES (?,?,?,?)",
    );
    const update = db.prepare(
      `UPDATE examples SET current_user=?,current_assistant=?,status=?,tags_json=?,notes=?,excluded_by_balance=?,split=?,content_hash=?,version=version+1,updated_at=? WHERE id=?`,
    );
    const deleteFts = db.prepare("DELETE FROM examples_fts WHERE example_id=?");
    const addFts = db.prepare(
      "INSERT INTO examples_fts(example_id,user_text,assistant_text) VALUES (?,?,?)",
    );
    for (const ex of examples) {
      let next = stateOf(ex);
      if (spec.kind === "dedupe") {
        const key = canonicalHash(ex.user, ex.assistant);
        if (seen.has(key)) next = { ...next, status: "excluded" };
        else seen.add(key);
      } else next = transformExample(ex, spec);
      const before = stateOf(ex);
      if (JSON.stringify(before) === JSON.stringify(next)) continue;
      affectedCount++;
      addChange.run(
        operationId,
        ex.id,
        JSON.stringify(before),
        JSON.stringify(next),
      );
      update.run(
        next.user,
        next.assistant,
        next.status,
        JSON.stringify(next.tags),
        next.notes,
        next.excludedByBalance ? 1 : 0,
        next.split,
        canonicalHash(next.user, next.assistant),
        stamp,
        ex.id,
      );
      deleteFts.run(ex.id);
      addFts.run(ex.id, next.user, next.assistant);
      if (before.user !== next.user || before.assistant !== next.assistant)
        reindexExample(db, ex.id, next.user, next.assistant);
    }
    if (!affectedCount)
      db.prepare("DELETE FROM operations WHERE id=?").run(operationId);
  });
  transaction();
  return { operationId, affectedCount };
}

function describeTransformation(spec: TransformationSpec): string {
  if (spec.kind === "replace")
    return `${spec.regex ? "Regex" : "Text"} replace in ${spec.field ?? "both"}`;
  if (spec.kind === "normalize")
    return `Normalized ${spec.field ?? "both"} text`;
  if (spec.kind === "set-status") return `Set review state to ${spec.status}`;
  if (spec.kind === "add-tag") return `Added tag “${spec.tag}”`;
  return "Excluded exact duplicates";
}

export function revertOperation(operationId: string): number {
  const db = getDb();
  const operation = db
    .prepare("SELECT * FROM operations WHERE id=?")
    .get(operationId) as { reverted_at: string | null } | undefined;
  if (!operation) throw new Error("Operation not found");
  if (operation.reverted_at)
    throw new Error("Operation has already been reverted");
  const changes = db
    .prepare("SELECT * FROM operation_changes WHERE operation_id=?")
    .all(operationId) as Array<{ example_id: string; before_json: string }>;
  const transaction = db.transaction(() => {
    const stamp = now();
    const update = db.prepare(
      `UPDATE examples SET current_user=?,current_assistant=?,status=?,tags_json=?,notes=?,excluded_by_balance=?,split=?,content_hash=?,version=version+1,updated_at=? WHERE id=?`,
    );
    for (const change of changes) {
      const before = JSON.parse(change.before_json) as MutableState;
      update.run(
        before.user,
        before.assistant,
        before.status,
        JSON.stringify(before.tags),
        before.notes,
        before.excludedByBalance ? 1 : 0,
        before.split,
        canonicalHash(before.user, before.assistant),
        stamp,
        change.example_id,
      );
      db.prepare("DELETE FROM examples_fts WHERE example_id=?").run(
        change.example_id,
      );
      db.prepare(
        "INSERT INTO examples_fts(example_id,user_text,assistant_text) VALUES (?,?,?)",
      ).run(change.example_id, before.user, before.assistant);
      reindexExample(db, change.example_id, before.user, before.assistant);
    }
    db.prepare("UPDATE operations SET reverted_at=? WHERE id=?").run(
      stamp,
      operationId,
    );
  });
  transaction();
  return changes.length;
}

function seededRank(seed: number, value: string): string {
  return hash(`${seed}:${value}`);
}
export function previewBalance(
  runId: string,
  config: BalanceConfig,
): {
  total: number;
  kept: number;
  excluded: number;
  sources: Array<{ source: string; total: number; kept: number }>;
} {
  const rows = getDb()
    .prepare(
      `SELECT e.id,COALESCE(CAST(e.chat_id AS TEXT),'unlinked') source FROM examples e WHERE e.run_id=? AND ${EFFECTIVE_INCLUDE_SQL} ORDER BY source,e.id`,
    )
    .all(runId) as Array<{ id: string; source: string }>;
  const groups = new Map<string, typeof rows>();
  for (const row of rows) {
    const group = groups.get(row.source) ?? [];
    group.push(row);
    groups.set(row.source, group);
  }
  const maxByShare =
    config.maxShare == null
      ? Infinity
      : Math.floor(rows.length * config.maxShare);
  let kept = 0;
  const sources = [];
  for (const [source, group] of groups) {
    const limit = Math.min(config.maxCount ?? Infinity, maxByShare);
    const count = Math.min(group.length, limit);
    kept += count;
    sources.push({ source, total: group.length, kept: count });
  }
  return {
    total: rows.length,
    kept,
    excluded: rows.length - kept,
    sources: sources.sort((a, b) => b.total - a.total).slice(0, 20),
  };
}

export function applyBalance(
  runId: string,
  config: BalanceConfig,
): { operationId: string; excluded: number } {
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT e.* FROM examples e WHERE e.run_id=? AND ${EFFECTIVE_INCLUDE_SQL}`,
    )
    .all(runId) as Record<string, unknown>[];
  const examples = rows.map(mapExample);
  const groups = new Map<string, TrainingExample[]>();
  for (const ex of examples) {
    const source = ex.chatId == null ? "unlinked" : String(ex.chatId);
    const group = groups.get(source) ?? [];
    group.push(ex);
    groups.set(source, group);
  }
  const maxByShare =
    config.maxShare == null
      ? Infinity
      : Math.floor(examples.length * config.maxShare);
  const excluded = new Set<string>();
  for (const group of groups.values()) {
    group.sort((a, b) =>
      seededRank(config.seed, a.sourceKey).localeCompare(
        seededRank(config.seed, b.sourceKey),
      ),
    );
    const limit = Math.min(config.maxCount ?? Infinity, maxByShare);
    for (const ex of group.slice(limit)) excluded.add(ex.id);
  }
  const operationId = crypto.randomUUID(),
    stamp = now();
  const transaction = db.transaction(() => {
    db.prepare(
      "INSERT INTO operations(id,run_id,kind,description,created_at) VALUES (?,?,?,?,?)",
    ).run(
      operationId,
      runId,
      "balance",
      `Balanced sources with count ${config.maxCount ?? "∞"} and share ${config.maxShare ?? "∞"}`,
      stamp,
    );
    const change = db.prepare(
      "INSERT INTO operation_changes(operation_id,example_id,before_json,after_json) VALUES (?,?,?,?)",
    );
    const update = db.prepare(
      "UPDATE examples SET excluded_by_balance=?,split=NULL,version=version+1,updated_at=? WHERE id=?",
    );
    for (const ex of examples) {
      const next = excluded.has(ex.id);
      if (ex.excludedByBalance === next) continue;
      change.run(
        operationId,
        ex.id,
        JSON.stringify(stateOf(ex)),
        JSON.stringify({
          ...stateOf(ex),
          excludedByBalance: next,
          split: null,
        }),
      );
      update.run(next ? 1 : 0, stamp, ex.id);
    }
  });
  transaction();
  return { operationId, excluded: excluded.size };
}

export function applySplit(
  runId: string,
  config: SplitConfig,
): Record<"train" | "validation" | "test", number> {
  const sum = config.train + config.validation + config.test;
  if (Math.abs(sum - 1) > 0.0001)
    throw new Error("Split ratios must add up to 1");
  if (config.sessionGapHours <= 0)
    throw new Error("Session boundary must be greater than zero");
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT e.id,e.session_key,e.content_hash,e.chat_id,e.start_timestamp,e.end_timestamp FROM examples e WHERE e.run_id=? AND ${EFFECTIVE_INCLUDE_SQL} AND e.excluded_by_balance=0 ORDER BY e.chat_id,e.start_timestamp,e.id`,
    )
    .all(runId) as Array<{
    id: string;
    session_key: string;
    content_hash: string;
    chat_id: number | null;
    start_timestamp: number | null;
    end_timestamp: number | null;
  }>;
  const sessionState = new Map<number, { number: number; last: number }>();
  const sessionById = new Map<string, string>();
  for (const row of rows) {
    if (row.chat_id == null || row.start_timestamp == null) {
      sessionById.set(row.id, row.session_key);
      continue;
    }
    const current = sessionState.get(row.chat_id) ?? {
      number: 0,
      last: row.start_timestamp,
    };
    if (row.start_timestamp - current.last > config.sessionGapHours * 3600)
      current.number += 1;
    current.last = row.end_timestamp ?? row.start_timestamp;
    sessionState.set(row.chat_id, current);
    sessionById.set(row.id, `chat:${row.chat_id}:session:${current.number}`);
  }
  const duplicateCounts = new Map<string, number>();
  for (const row of rows)
    duplicateCounts.set(
      row.content_hash,
      (duplicateCounts.get(row.content_hash) ?? 0) + 1,
    );
  const assignments = new Map<string, "train" | "validation" | "test">();
  const counts = { train: 0, validation: 0, test: 0 };
  for (const row of rows) {
    const group =
      (duplicateCounts.get(row.content_hash) ?? 0) > 1
        ? `duplicate:${row.content_hash}`
        : sessionById.get(row.id)!;
    let split = assignments.get(group);
    if (!split) {
      const value =
        parseInt(seededRank(config.seed, group).slice(0, 12), 16) /
        0xffffffffffff;
      split =
        value < config.train
          ? "train"
          : value < config.train + config.validation
            ? "validation"
            : "test";
      assignments.set(group, split);
    }
    counts[split]++;
  }
  const operationId = crypto.randomUUID(),
    stamp = now();
  const transaction = db.transaction(() => {
    db.prepare(
      "INSERT INTO operations(id,run_id,kind,description,created_at) VALUES (?,?,?,?,?)",
    ).run(
      operationId,
      runId,
      "split",
      `Assigned deterministic ${Math.round(config.train * 100)}/${Math.round(config.validation * 100)}/${Math.round(config.test * 100)} splits with a ${config.sessionGapHours}h session boundary`,
      stamp,
    );
    const update = db.prepare(
      "UPDATE examples SET split=?,session_key=?,version=version+1,updated_at=? WHERE id=?",
    );
    for (const row of rows) {
      const group =
        (duplicateCounts.get(row.content_hash) ?? 0) > 1
          ? `duplicate:${row.content_hash}`
          : sessionById.get(row.id)!;
      update.run(
        assignments.get(group)!,
        sessionById.get(row.id)!,
        stamp,
        row.id,
      );
    }
    db.prepare(
      `UPDATE examples AS e SET split=NULL WHERE e.run_id=? AND (NOT ${EFFECTIVE_INCLUDE_SQL} OR e.excluded_by_balance=1)`,
    ).run(runId);
  });
  transaction();
  return counts;
}

async function atomicWrite(filePath: string, content: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  const temp = `${filePath}.${crypto.randomUUID()}.tmp`;
  await fs.promises.writeFile(temp, content, "utf8");
  await fs.promises.rename(temp, filePath);
}
export async function exportRun(
  runId: string,
): Promise<{ directory: string; manifest: ExportManifest }> {
  if (!/^[a-zA-Z0-9_-]+$/.test(runId)) throw new Error("Invalid run id");
  const run = getRun(runId);
  if (!run || run.status !== "completed")
    throw new Error("Only completed runs can be exported");
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT e.current_user,e.current_assistant,e.split FROM examples e WHERE e.run_id=? AND ${EFFECTIVE_INCLUDE_SQL} AND e.excluded_by_balance=0 ORDER BY e.rowid`,
    )
    .all(runId) as Array<{
    current_user: string;
    current_assistant: string;
    split: string | null;
  }>;
  const errors: string[] = [];
  if (rows.some((r) => !r.split))
    errors.push("Every active example must have a split before export");
  if (errors.length) throw new Error(errors.join(". "));
  const directory = path.join(EXPORTS_DIR, runId);
  const files = { train: "", validation: "", test: "" };
  for (const row of rows) {
    const line =
      JSON.stringify([
        { role: "user", content: row.current_user },
        { role: "assistant", content: row.current_assistant },
      ]) + "\n";
    files[row.split as keyof typeof files] += line;
  }
  const operations = db
    .prepare(
      "SELECT kind,description FROM operations WHERE run_id=? AND reverted_at IS NULL ORDER BY created_at",
    )
    .all(runId) as Array<{ kind: string; description: string }>;
  const codexAggregate = db
    .prepare(
      `SELECT MAX(rubric_version) rubric_version,COUNT(*) reviewed,SUM(decision='include') included,SUM(decision='exclude') excluded,SUM(decision='unsure') unsure FROM codex_reviews WHERE run_id=?`,
    )
    .get(runId) as Record<string, string | number | null>;
  const manifest: ExportManifest = {
    runId,
    createdAt: now(),
    counts: {
      train: files.train.split("\n").filter(Boolean).length,
      validation: files.validation.split("\n").filter(Boolean).length,
      test: files.test.split("\n").filter(Boolean).length,
    },
    hashes: {
      train: hash(files.train),
      validation: hash(files.validation),
      test: hash(files.test),
    },
    config: run.config,
    operations,
    validation: { valid: true, errors: [] },
    codexReview: {
      rubricVersion: codexAggregate.rubric_version
        ? String(codexAggregate.rubric_version)
        : null,
      reviewed: Number(codexAggregate.reviewed ?? 0),
      include: Number(codexAggregate.included ?? 0),
      exclude: Number(codexAggregate.excluded ?? 0),
      unsure: Number(codexAggregate.unsure ?? 0),
    },
  };
  await Promise.all([
    atomicWrite(path.join(directory, "train.jsonl"), files.train),
    atomicWrite(path.join(directory, "validation.jsonl"), files.validation),
    atomicWrite(path.join(directory, "test.jsonl"), files.test),
  ]);
  await atomicWrite(
    path.join(directory, "manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  db.prepare(
    "INSERT INTO exports(id,run_id,path,manifest_json,created_at) VALUES (?,?,?,?,?)",
  ).run(
    crypto.randomUUID(),
    runId,
    directory,
    JSON.stringify(manifest),
    manifest.createdAt,
  );
  return { directory, manifest };
}
