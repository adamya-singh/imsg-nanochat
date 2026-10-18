import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { getDb } from "@/lib/db";
import { BASELINE_JSONL } from "@/lib/paths";
import type {
  CodexDecision,
  DatasetRun,
  ExampleQuery,
  Operation,
  Overview,
  ReviewStatus,
  TrainingExample,
} from "@/lib/types";

type Pair = [
  { role: "user"; content: string },
  { role: "assistant"; content: string },
];
type Provenance = {
  source_key: string;
  chat_id: number;
  inbound_message_ids: number[];
  reply_message_ids: number[];
  start_timestamp: number;
  end_timestamp: number;
};

const now = () => new Date().toISOString();
export const hash = (value: string) =>
  crypto.createHash("sha256").update(value).digest("hex");
export const canonicalHash = (user: string, assistant: string) =>
  hash(
    JSON.stringify([
      user.trim().replace(/\r\n?/g, "\n"),
      assistant.trim().replace(/\r\n?/g, "\n"),
    ]),
  );

function searchTerms(user: string, assistant: string): string[] {
  const terms = [
    ...new Set(
      `${user} ${assistant}`.toLocaleLowerCase().match(/[\p{L}\p{N}_']+/gu) ??
        [],
    ),
  ];
  return terms.length ? terms : [""];
}

export function reindexExample(
  db: ReturnType<typeof getDb>,
  exampleId: string,
  user: string,
  assistant: string,
): void {
  db.prepare("DELETE FROM example_terms WHERE example_id=?").run(exampleId);
  const insert = db.prepare(
    "INSERT OR IGNORE INTO example_terms(example_id,term) VALUES (?,?)",
  );
  for (const term of searchTerms(user, assistant)) insert.run(exampleId, term);
}

function ensureTermIndex(): void {
  const db = getDb();
  const examples = Number(
    (
      db.prepare("SELECT COUNT(*) count FROM examples").get() as {
        count: number;
      }
    ).count,
  );
  if (!examples) return;
  const indexed = Number(
    (
      db
        .prepare("SELECT COUNT(DISTINCT example_id) count FROM example_terms")
        .get() as {
        count: number;
      }
    ).count,
  );
  if (indexed === examples) return;
  const rows = db
    .prepare("SELECT id,current_user,current_assistant FROM examples")
    .all() as Array<{
    id: string;
    current_user: string;
    current_assistant: string;
  }>;
  const transaction = db.transaction(() => {
    db.prepare("DELETE FROM example_terms").run();
    for (const row of rows)
      reindexExample(db, row.id, row.current_user, row.current_assistant);
  });
  transaction();
}

function parseJson<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function mapRun(row: Record<string, unknown>): DatasetRun {
  return {
    id: String(row.id),
    name: String(row.name),
    status: row.status as DatasetRun["status"],
    createdAt: String(row.created_at),
    completedAt: row.completed_at ? String(row.completed_at) : null,
    parentRunId: row.parent_run_id ? String(row.parent_run_id) : null,
    config: parseJson(String(row.config_json), {}),
    stats: parseJson(String(row.stats_json), {}),
    error: row.error ? String(row.error) : null,
    exampleCount: Number(row.example_count ?? 0),
  };
}

function mapExample(row: Record<string, unknown>): TrainingExample {
  const codexReview = row.codex_decision
    ? {
        decision: row.codex_decision as CodexDecision,
        confidence: Number(row.codex_confidence),
        reasons: parseJson(String(row.codex_reasons_json), []),
        rationale: String(row.codex_rationale),
        rubricVersion: String(row.codex_rubric_version),
        model: String(row.codex_model),
        reviewedAt: String(row.codex_reviewed_at),
        reviewedContentHash: String(row.codex_content_hash),
        stale: String(row.codex_content_hash) !== String(row.content_hash),
      }
    : null;
  const effectiveDecision =
    row.status === "excluded"
      ? "exclude"
      : row.status === "approved" || row.status === "flagged"
        ? "include"
        : codexReview?.decision === "exclude" && !codexReview.stale
          ? "exclude"
          : "include";
  return {
    id: String(row.id),
    runId: String(row.run_id),
    sourceKey: String(row.source_key),
    originalUser: String(row.original_user),
    originalAssistant: String(row.original_assistant),
    user: String(row.current_user),
    assistant: String(row.current_assistant),
    status: row.status as ReviewStatus,
    tags: parseJson(String(row.tags_json), []),
    notes: String(row.notes),
    version: Number(row.version),
    chatId: row.chat_id == null ? null : Number(row.chat_id),
    inboundMessageIds: parseJson(String(row.inbound_ids_json), []),
    replyMessageIds: parseJson(String(row.reply_ids_json), []),
    startTimestamp:
      row.start_timestamp == null ? null : Number(row.start_timestamp),
    endTimestamp: row.end_timestamp == null ? null : Number(row.end_timestamp),
    sessionKey: String(row.session_key),
    split: (row.split ?? null) as TrainingExample["split"],
    excludedByBalance: Boolean(row.excluded_by_balance),
    codexReview,
    effectiveDecision,
    updatedAt: String(row.updated_at),
  };
}

const codexSelect = `
  cr.decision codex_decision,
  cr.confidence codex_confidence,
  cr.reasons_json codex_reasons_json,
  cr.rationale codex_rationale,
  cr.rubric_version codex_rubric_version,
  cr.model codex_model,
  cr.reviewed_at codex_reviewed_at,
  cr.content_hash codex_content_hash`;

export const EFFECTIVE_INCLUDE_SQL = `(e.status <> 'excluded' AND NOT (
  e.status = 'unreviewed' AND EXISTS (
    SELECT 1 FROM codex_reviews effective_cr
    WHERE effective_cr.example_id=e.id
      AND effective_cr.content_hash=e.content_hash
      AND effective_cr.decision='exclude'
  )
))`;

export function listRuns(): DatasetRun[] {
  ensureBaseline();
  const rows = getDb()
    .prepare(
      `
    SELECT r.*, COUNT(e.id) example_count FROM runs r
    LEFT JOIN examples e ON e.run_id=r.id
    GROUP BY r.id ORDER BY r.created_at DESC
  `,
    )
    .all() as Record<string, unknown>[];
  return rows.map(mapRun);
}

export function getRun(id: string): DatasetRun | null {
  const row = getDb()
    .prepare(
      `SELECT r.*, COUNT(e.id) example_count FROM runs r LEFT JOIN examples e ON e.run_id=r.id WHERE r.id=? GROUP BY r.id`,
    )
    .get(id) as Record<string, unknown> | undefined;
  return row ? mapRun(row) : null;
}

export function ensureBaseline(): void {
  const db = getDb();
  const count = Number(
    (db.prepare("SELECT COUNT(*) count FROM runs").get() as { count: number })
      .count,
  );
  if (count) {
    ensureTermIndex();
    return;
  }
  if (!fs.existsSync(BASELINE_JSONL)) return;
  importRun({
    id: "baseline",
    name: "Imported baseline",
    jsonlPath: BASELINE_JSONL,
    provenancePath: null,
    config: {
      source: path.relative(
        path.resolve(BASELINE_JSONL, "../../.."),
        BASELINE_JSONL,
      ),
    },
    stats: { imported: true },
    parentRunId: null,
  });
}

export function importRun(input: {
  id: string;
  name: string;
  jsonlPath: string;
  provenancePath: string | null;
  config: Record<string, unknown>;
  stats: Record<string, unknown>;
  parentRunId: string | null;
}): { count: number; reconciliations: number } {
  const pairLines = fs
    .readFileSync(input.jsonlPath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean);
  const provenanceLines =
    input.provenancePath && fs.existsSync(input.provenancePath)
      ? fs
          .readFileSync(input.provenancePath, "utf8")
          .split(/\r?\n/)
          .filter(Boolean)
      : [];
  if (provenanceLines.length && provenanceLines.length !== pairLines.length) {
    throw new Error("Provenance sidecar does not align with the dataset JSONL");
  }
  const records = pairLines.map((line, index) => {
    const pair = JSON.parse(line) as Pair;
    if (
      !Array.isArray(pair) ||
      pair.length !== 2 ||
      pair[0]?.role !== "user" ||
      pair[1]?.role !== "assistant"
    ) {
      throw new Error(`Invalid training pair on line ${index + 1}`);
    }
    const provenance = provenanceLines[index]
      ? (JSON.parse(provenanceLines[index]) as Provenance)
      : null;
    return {
      pair,
      provenance,
      sourceKey: provenance?.source_key ?? hash(`${index}:${line}`),
      index,
    };
  });

  const sessions = new Map<number, { number: number; last: number }>();
  const db = getDb();
  let reconciliations = 0;
  const insert = db.prepare(`INSERT INTO examples (
    id,run_id,source_key,original_user,original_assistant,current_user,current_assistant,content_hash,
    status,tags_json,notes,version,chat_id,inbound_ids_json,reply_ids_json,start_timestamp,end_timestamp,
    session_key,split,excluded_by_balance,created_at,updated_at
  ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`);
  const previous = input.parentRunId
    ? db.prepare("SELECT * FROM examples WHERE run_id=? AND source_key=?")
    : null;
  const addFts = db.prepare(
    "INSERT INTO examples_fts(example_id,user_text,assistant_text) VALUES (?,?,?)",
  );
  const previousCodex = db.prepare(
    "SELECT * FROM codex_reviews WHERE example_id=? AND content_hash=?",
  );
  const carryCodex = db.prepare(
    `INSERT INTO codex_reviews(example_id,run_id,content_hash,rubric_version,decision,confidence,reasons_json,rationale,job_id,model,reviewed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
  );
  const transaction = db.transaction(() => {
    db.prepare(
      `INSERT INTO runs(id,name,status,created_at,completed_at,parent_run_id,config_json,stats_json,source_jsonl_path,provenance_path,error,reconciliation_count)
      VALUES (?,?,?,?,?,?,?,?,?,?,NULL,0)
      ON CONFLICT(id) DO UPDATE SET name=excluded.name,status='completed',completed_at=excluded.completed_at,config_json=excluded.config_json,stats_json=excluded.stats_json,source_jsonl_path=excluded.source_jsonl_path,provenance_path=excluded.provenance_path,error=NULL`,
    ).run(
      input.id,
      input.name,
      "completed",
      now(),
      now(),
      input.parentRunId,
      JSON.stringify(input.config),
      JSON.stringify(input.stats),
      input.jsonlPath,
      input.provenancePath,
    );
    for (const record of records) {
      const user = record.pair[0].content;
      const assistant = record.pair[1].content;
      let status: ReviewStatus = "unreviewed",
        tags = "[]",
        notes = "",
        currentUser = user,
        currentAssistant = assistant;
      let oldExample: Record<string, unknown> | undefined;
      if (previous && input.parentRunId) {
        const old = previous.get(input.parentRunId, record.sourceKey) as
          Record<string, unknown> | undefined;
        oldExample = old;
        if (
          old &&
          old.original_user === user &&
          old.original_assistant === assistant
        ) {
          status = old.status as ReviewStatus;
          tags = String(old.tags_json);
          notes = String(old.notes);
          currentUser = String(old.current_user);
          currentAssistant = String(old.current_assistant);
        } else if (old) reconciliations += 1;
      }
      let sessionKey = `source:${record.sourceKey}`;
      if (record.provenance) {
        const state = sessions.get(record.provenance.chat_id) ?? {
          number: 0,
          last: record.provenance.start_timestamp,
        };
        if (record.provenance.start_timestamp - state.last > 6 * 3600)
          state.number += 1;
        state.last = record.provenance.end_timestamp;
        sessions.set(record.provenance.chat_id, state);
        sessionKey = `chat:${record.provenance.chat_id}:session:${state.number}`;
      }
      const id = `${input.id}:${record.sourceKey}`;
      const stamp = now();
      const currentHash = canonicalHash(currentUser, currentAssistant);
      insert.run(
        id,
        input.id,
        record.sourceKey,
        user,
        assistant,
        currentUser,
        currentAssistant,
        currentHash,
        status,
        tags,
        notes,
        1,
        record.provenance?.chat_id ?? null,
        JSON.stringify(record.provenance?.inbound_message_ids ?? []),
        JSON.stringify(record.provenance?.reply_message_ids ?? []),
        record.provenance?.start_timestamp ?? null,
        record.provenance?.end_timestamp ?? null,
        sessionKey,
        null,
        0,
        stamp,
        stamp,
      );
      addFts.run(id, currentUser, currentAssistant);
      reindexExample(db, id, currentUser, currentAssistant);
      if (oldExample) {
        const oldReview = previousCodex.get(
          String(oldExample.id),
          currentHash,
        ) as Record<string, unknown> | undefined;
        if (oldReview)
          carryCodex.run(
            id,
            input.id,
            currentHash,
            String(oldReview.rubric_version),
            String(oldReview.decision),
            Number(oldReview.confidence),
            String(oldReview.reasons_json),
            String(oldReview.rationale),
            String(oldReview.job_id),
            String(oldReview.model),
            String(oldReview.reviewed_at),
          );
      }
    }
    db.prepare("UPDATE runs SET reconciliation_count=? WHERE id=?").run(
      reconciliations,
      input.id,
    );
  });
  transaction();
  return { count: records.length, reconciliations };
}

function whereForQuery(
  query: ExampleQuery,
  alias = "e",
): { sql: string; params: Array<string | number | null> } {
  const clauses = [`${alias}.run_id=?`];
  const params: Array<string | number | null> = [query.runId];
  if (query.search?.trim()) {
    const words = query.search
      .trim()
      .toLocaleLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    for (const word of words) {
      clauses.push(
        `${alias}.id IN (SELECT example_id FROM example_terms WHERE term LIKE ?)`,
      );
      params.push(`${word}%`);
    }
  }
  if (query.status && query.status !== "all") {
    clauses.push(`${alias}.status=?`);
    params.push(query.status);
  }
  if (query.split && query.split !== "all") {
    clauses.push(
      query.split === "unassigned"
        ? `${alias}.split IS NULL`
        : `${alias}.split=?`,
    );
    if (query.split !== "unassigned") params.push(query.split);
  }
  if (query.tag) {
    clauses.push(`${alias}.tags_json LIKE ?`);
    params.push(`%"${query.tag.replaceAll('"', "")}"%`);
  }
  if (query.minLength != null) {
    clauses.push(
      `(length(${alias}.current_user)+length(${alias}.current_assistant))>=?`,
    );
    params.push(query.minLength);
  }
  if (query.maxLength != null) {
    clauses.push(
      `(length(${alias}.current_user)+length(${alias}.current_assistant))<=?`,
    );
    params.push(query.maxLength);
  }
  if (query.queue === "long")
    clauses.push(
      `(length(${alias}.current_user)+length(${alias}.current_assistant))>=300`,
    );
  if (query.queue === "duplicate")
    clauses.push(
      `EXISTS (SELECT 1 FROM examples d WHERE d.run_id=${alias}.run_id AND d.content_hash=${alias}.content_hash AND d.id<>${alias}.id)`,
    );
  if (query.queue === "suspected-junk")
    clauses.push(
      `((lower(${alias}.current_user)||' '||lower(${alias}.current_assistant)) GLOB '*verification code*' OR (lower(${alias}.current_user)||' '||lower(${alias}.current_assistant)) GLOB '*streamtyped*')`,
    );
  if (query.codexDecision && query.codexDecision !== "all") {
    if (query.codexDecision === "unreviewed")
      clauses.push(
        `NOT EXISTS (SELECT 1 FROM codex_reviews c WHERE c.example_id=${alias}.id)`,
      );
    else if (query.codexDecision === "stale")
      clauses.push(
        `EXISTS (SELECT 1 FROM codex_reviews c WHERE c.example_id=${alias}.id AND c.content_hash<>${alias}.content_hash)`,
      );
    else {
      clauses.push(
        `EXISTS (SELECT 1 FROM codex_reviews c WHERE c.example_id=${alias}.id AND c.content_hash=${alias}.content_hash AND c.decision=?)`,
      );
      params.push(query.codexDecision);
    }
  }
  if (query.disagreement)
    clauses.push(
      `EXISTS (SELECT 1 FROM codex_reviews c WHERE c.example_id=${alias}.id AND c.content_hash=${alias}.content_hash AND ((c.decision='exclude' AND ${alias}.status IN ('approved','flagged')) OR (c.decision='include' AND ${alias}.status='excluded')))`,
    );
  if (query.cursor) {
    clauses.push(`${alias}.rowid>?`);
    params.push(Number(query.cursor));
  }
  return { sql: clauses.join(" AND "), params };
}

export function listExamples(query: ExampleQuery): {
  examples: TrainingExample[];
  nextCursor: string | null;
  total: number;
} {
  ensureBaseline();
  const db = getDb();
  const limit = Math.min(Math.max(query.limit ?? 120, 1), 300);
  const where = whereForQuery(query);
  const rows = db
    .prepare(
      `SELECT e.*, e.rowid cursor_rowid, ${codexSelect} FROM examples e LEFT JOIN codex_reviews cr ON cr.example_id=e.id WHERE ${where.sql} ORDER BY e.rowid LIMIT ?`,
    )
    .all(...where.params, limit + 1) as Record<string, unknown>[];
  const total = Number(
    (
      db
        .prepare(`SELECT COUNT(*) count FROM examples e WHERE ${where.sql}`)
        .get(...where.params) as { count: number }
    ).count,
  );
  const hasMore = rows.length > limit;
  const selected = rows.slice(0, limit);
  return {
    examples: selected.map(mapExample),
    nextCursor: hasMore ? String(selected.at(-1)?.cursor_rowid) : null,
    total,
  };
}

export function getExample(id: string): TrainingExample | null {
  const row = getDb()
    .prepare(
      `SELECT e.*, ${codexSelect} FROM examples e LEFT JOIN codex_reviews cr ON cr.example_id=e.id WHERE e.id=?`,
    )
    .get(id) as Record<string, unknown> | undefined;
  return row ? mapExample(row) : null;
}

export function updateExample(
  id: string,
  patch: Partial<
    Pick<TrainingExample, "user" | "assistant" | "status" | "tags" | "notes">
  >,
  expectedVersion: number,
): TrainingExample {
  const db = getDb();
  const current = getExample(id);
  if (!current) throw new Error("Example not found");
  if (current.version !== expectedVersion)
    throw new Error(
      "This example changed in another tab. Reload before saving.",
    );
  const next = {
    ...current,
    ...patch,
    tags: patch.tags
      ? [...new Set(patch.tags.map((t) => t.trim()).filter(Boolean))]
      : current.tags,
  };
  const operationId = crypto.randomUUID();
  const stamp = now();
  const transaction = db.transaction(() => {
    db.prepare(
      "INSERT INTO operations(id,run_id,kind,description,created_at) VALUES (?,?,?,?,?)",
    ).run(
      operationId,
      current.runId,
      "edit",
      "Edited one training example",
      stamp,
    );
    db.prepare(
      "INSERT INTO operation_changes(operation_id,example_id,before_json,after_json) VALUES (?,?,?,?)",
    ).run(operationId, id, JSON.stringify(current), JSON.stringify(next));
    const result = db
      .prepare(
        `UPDATE examples SET current_user=?,current_assistant=?,status=?,tags_json=?,notes=?,content_hash=?,version=version+1,updated_at=? WHERE id=? AND version=?`,
      )
      .run(
        next.user,
        next.assistant,
        next.status,
        JSON.stringify(next.tags),
        next.notes,
        canonicalHash(next.user, next.assistant),
        stamp,
        id,
        expectedVersion,
      );
    if (result.changes !== 1)
      throw new Error(
        "This example changed in another tab. Reload before saving.",
      );
    db.prepare("DELETE FROM examples_fts WHERE example_id=?").run(id);
    db.prepare(
      "INSERT INTO examples_fts(example_id,user_text,assistant_text) VALUES (?,?,?)",
    ).run(id, next.user, next.assistant);
    reindexExample(db, id, next.user, next.assistant);
  });
  transaction();
  return getExample(id)!;
}

function operationRows(runId: string, limit = 12): Operation[] {
  const rows = getDb()
    .prepare(
      `SELECT o.*,COUNT(c.example_id) affected_count FROM operations o LEFT JOIN operation_changes c ON c.operation_id=o.id WHERE o.run_id=? GROUP BY o.id ORDER BY o.created_at DESC LIMIT ?`,
    )
    .all(runId, limit) as Record<string, unknown>[];
  return rows.map((row) => ({
    id: String(row.id),
    runId: String(row.run_id),
    kind: String(row.kind),
    description: String(row.description),
    createdAt: String(row.created_at),
    revertedAt: row.reverted_at ? String(row.reverted_at) : null,
    affectedCount: Number(row.affected_count),
  }));
}

export function getOverview(runId?: string): Overview {
  ensureBaseline();
  const run = runId
    ? getRun(runId)
    : listRuns().find((r) => r.status === "completed");
  if (!run) throw new Error("No completed dataset run is available");
  const db = getDb();
  const countRows = db
    .prepare(
      "SELECT status,COUNT(*) count FROM examples WHERE run_id=? GROUP BY status",
    )
    .all(run.id) as Array<{ status: ReviewStatus; count: number }>;
  const counts = {
    total: 0,
    active: 0,
    balancedOut: 0,
    unreviewed: 0,
    approved: 0,
    flagged: 0,
    excluded: 0,
  };
  for (const row of countRows) {
    counts[row.status] = Number(row.count);
    counts.total += Number(row.count);
  }
  counts.balancedOut = Number(
    (
      db
        .prepare(
          "SELECT COUNT(*) count FROM examples WHERE run_id=? AND excluded_by_balance=1",
        )
        .get(run.id) as { count: number }
    ).count,
  );
  const codexRows = db
    .prepare(
      `SELECT
    SUM(CASE WHEN cr.content_hash=e.content_hash THEN 1 ELSE 0 END) reviewed,
    SUM(CASE WHEN cr.content_hash=e.content_hash AND cr.decision='include' THEN 1 ELSE 0 END) included,
    SUM(CASE WHEN cr.content_hash=e.content_hash AND cr.decision='exclude' THEN 1 ELSE 0 END) excluded,
    SUM(CASE WHEN cr.content_hash=e.content_hash AND cr.decision='unsure' THEN 1 ELSE 0 END) unsure,
    SUM(CASE WHEN cr.example_id IS NOT NULL AND cr.content_hash<>e.content_hash THEN 1 ELSE 0 END) stale,
    SUM(CASE WHEN cr.example_id IS NULL THEN 1 ELSE 0 END) unreviewed,
    SUM(CASE WHEN cr.content_hash=e.content_hash AND ((cr.decision='exclude' AND e.status IN ('approved','flagged')) OR (cr.decision='include' AND e.status='excluded')) THEN 1 ELSE 0 END) disagreements,
    SUM(CASE WHEN e.status='unreviewed' AND cr.content_hash=e.content_hash AND cr.decision='exclude' THEN 1 ELSE 0 END) effective_excluded
    FROM examples e LEFT JOIN codex_reviews cr ON cr.example_id=e.id WHERE e.run_id=?`,
    )
    .get(run.id) as Record<string, number | null>;
  const codex = {
    reviewed: Number(codexRows.reviewed ?? 0),
    include: Number(codexRows.included ?? 0),
    exclude: Number(codexRows.excluded ?? 0),
    unsure: Number(codexRows.unsure ?? 0),
    stale: Number(codexRows.stale ?? 0),
    unreviewed: Number(codexRows.unreviewed ?? 0),
    disagreements: Number(codexRows.disagreements ?? 0),
    effectiveExcluded: Number(codexRows.effective_excluded ?? 0),
  };
  counts.active = Number(
    (
      db
        .prepare(
          `SELECT COUNT(*) count FROM examples e WHERE e.run_id=? AND ${EFFECTIVE_INCLUDE_SQL} AND e.excluded_by_balance=0`,
        )
        .get(run.id) as { count: number }
    ).count,
  );
  const splitRows = db
    .prepare(
      `SELECT COALESCE(e.split,'unassigned') split,COUNT(*) count FROM examples e WHERE e.run_id=? AND ${EFFECTIVE_INCLUDE_SQL} AND e.excluded_by_balance=0 GROUP BY e.split`,
    )
    .all(run.id) as Array<{ split: string; count: number }>;
  const splits = { train: 0, validation: 0, test: 0, unassigned: 0 };
  for (const row of splitRows)
    splits[row.split as keyof typeof splits] = Number(row.count);
  const long = Number(
    (
      db
        .prepare(
          "SELECT COUNT(*) count FROM examples WHERE run_id=? AND length(current_user)+length(current_assistant)>=300",
        )
        .get(run.id) as { count: number }
    ).count,
  );
  const duplicate = Number(
    (
      db
        .prepare(
          "SELECT COALESCE(SUM(count),0) count FROM (SELECT COUNT(*) count FROM examples WHERE run_id=? GROUP BY content_hash HAVING count>1)",
        )
        .get(run.id) as { count: number }
    ).count,
  );
  const suspectedJunk = Number(
    (
      db
        .prepare(
          "SELECT COUNT(*) count FROM examples WHERE run_id=? AND ((lower(current_user)||' '||lower(current_assistant)) GLOB '*verification code*' OR (lower(current_user)||' '||lower(current_assistant)) GLOB '*streamtyped*')",
        )
        .get(run.id) as { count: number }
    ).count,
  );
  const lengthRows = db
    .prepare(
      `SELECT CASE WHEN length(current_user)+length(current_assistant)<50 THEN '0–49' WHEN length(current_user)+length(current_assistant)<100 THEN '50–99' WHEN length(current_user)+length(current_assistant)<200 THEN '100–199' WHEN length(current_user)+length(current_assistant)<400 THEN '200–399' ELSE '400+' END bucket,COUNT(*) count FROM examples WHERE run_id=? GROUP BY bucket ORDER BY MIN(length(current_user)+length(current_assistant))`,
    )
    .all(run.id) as Array<{ bucket: string; count: number }>;
  const sourceRows = db
    .prepare(
      "SELECT COALESCE(CAST(chat_id AS TEXT),'unlinked') source,COUNT(*) count FROM examples WHERE run_id=? GROUP BY chat_id ORDER BY count DESC LIMIT 8",
    )
    .all(run.id) as Array<{ source: string; count: number }>;
  return {
    run,
    counts,
    splits,
    queues: { long, duplicate, suspectedJunk },
    lengths: lengthRows.map((r) => ({
      bucket: r.bucket,
      count: Number(r.count),
    })),
    sources: sourceRows.map((r) => ({
      source: r.source,
      count: Number(r.count),
      share: Number(r.count) / Math.max(1, counts.total),
    })),
    recentOperations: operationRows(run.id),
    codex,
  };
}

export function getOperationHistory(runId: string, limit = 100): Operation[] {
  return operationRows(runId, limit);
}
export { whereForQuery, mapExample };
