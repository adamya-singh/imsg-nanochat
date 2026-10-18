export type ReviewStatus = "unreviewed" | "approved" | "flagged" | "excluded";
export type SplitName = "train" | "validation" | "test" | null;
export type RunStatus = "processing" | "completed" | "failed" | "cancelled";
export type CodexDecision = "include" | "exclude" | "unsure";
export type EffectiveDecision = "include" | "exclude";
export type CodexReviewJobStatus =
  "queued" | "running" | "paused" | "completed" | "failed" | "cancelled";

export interface CodexReview {
  decision: CodexDecision;
  confidence: number;
  reasons: string[];
  rationale: string;
  rubricVersion: string;
  model: string;
  reviewedAt: string;
  reviewedContentHash: string;
  stale: boolean;
}

export interface CodexReviewJob {
  id: string;
  runId: string;
  scope: "pilot" | "full";
  status: CodexReviewJobStatus;
  rubricVersion: string;
  batchSize: number;
  totalCount: number;
  processedCount: number;
  includeCount: number;
  excludeCount: number;
  unsureCount: number;
  retryCount: number;
  currentBatch: number;
  model: string;
  cliVersion: string | null;
  error: string | null;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
  updatedAt: string;
}

export interface DatasetRun {
  id: string;
  name: string;
  status: RunStatus;
  createdAt: string;
  completedAt: string | null;
  parentRunId: string | null;
  config: Record<string, unknown>;
  stats: Record<string, unknown>;
  error: string | null;
  exampleCount: number;
}

export interface TrainingExample {
  id: string;
  runId: string;
  sourceKey: string;
  originalUser: string;
  originalAssistant: string;
  user: string;
  assistant: string;
  status: ReviewStatus;
  tags: string[];
  notes: string;
  version: number;
  chatId: number | null;
  inboundMessageIds: number[];
  replyMessageIds: number[];
  startTimestamp: number | null;
  endTimestamp: number | null;
  sessionKey: string;
  split: SplitName;
  excludedByBalance: boolean;
  codexReview: CodexReview | null;
  effectiveDecision: EffectiveDecision;
  updatedAt: string;
}

export interface ExampleQuery {
  runId: string;
  search?: string;
  status?: ReviewStatus | "all";
  split?: "all" | Exclude<SplitName, null> | "unassigned";
  tag?: string;
  minLength?: number;
  maxLength?: number;
  queue?: "long" | "duplicate" | "suspected-junk";
  codexDecision?: CodexDecision | "unreviewed" | "stale" | "all";
  disagreement?: boolean;
  cursor?: string;
  limit?: number;
}

export type TransformationKind =
  "replace" | "normalize" | "set-status" | "add-tag" | "dedupe";

export interface TransformationSpec {
  kind: TransformationKind;
  query: ExampleQuery;
  field?: "user" | "assistant" | "both";
  find?: string;
  replace?: string;
  regex?: boolean;
  status?: ReviewStatus;
  tag?: string;
}

export interface TransformationPreview {
  affectedCount: number;
  samples: Array<{
    id: string;
    before: {
      user: string;
      assistant: string;
      status: ReviewStatus;
      tags: string[];
    };
    after: {
      user: string;
      assistant: string;
      status: ReviewStatus;
      tags: string[];
    };
  }>;
  warning?: string;
}

export interface BalanceConfig {
  maxCount: number | null;
  maxShare: number | null;
  seed: number;
}

export interface SplitConfig {
  train: number;
  validation: number;
  test: number;
  seed: number;
  sessionGapHours: number;
}

export interface Operation {
  id: string;
  runId: string;
  kind: string;
  description: string;
  createdAt: string;
  revertedAt: string | null;
  affectedCount: number;
}

export interface ExportManifest {
  runId: string;
  createdAt: string;
  counts: Record<"train" | "validation" | "test", number>;
  hashes: Record<"train" | "validation" | "test", string>;
  config: Record<string, unknown>;
  operations: Array<{ kind: string; description: string }>;
  validation: { valid: boolean; errors: string[] };
  codexReview: {
    rubricVersion: string | null;
    reviewed: number;
    include: number;
    exclude: number;
    unsure: number;
  };
}

export interface Overview {
  run: DatasetRun;
  counts: Record<ReviewStatus | "active" | "balancedOut" | "total", number>;
  splits: Record<"train" | "validation" | "test" | "unassigned", number>;
  queues: Record<"long" | "duplicate" | "suspectedJunk", number>;
  lengths: Array<{ bucket: string; count: number }>;
  sources: Array<{ source: string; count: number; share: number }>;
  recentOperations: Operation[];
  codex: {
    reviewed: number;
    include: number;
    exclude: number;
    unsure: number;
    stale: number;
    unreviewed: number;
    disagreements: number;
    effectiveExcluded: number;
  };
}
