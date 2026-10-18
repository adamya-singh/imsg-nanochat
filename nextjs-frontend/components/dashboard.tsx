"use client";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  Flag,
  Search,
  Trash2,
} from "lucide-react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  DatasetRun,
  Overview,
  ReviewStatus,
  TrainingExample,
} from "@/lib/types";

type Result = {
  examples: TrainingExample[];
  nextCursor: string | null;
  total: number;
};
export function Dashboard({
  initialOverview,
  initialExamples,
  runs,
}: {
  initialOverview: Overview;
  initialExamples: Result;
  runs: DatasetRun[];
}) {
  const router = useRouter();
  const [overview, setOverview] = useState(initialOverview);
  const [result, setResult] = useState(initialExamples);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [queue, setQueue] = useState("");
  const [codexDecision, setCodexDecision] = useState("all");
  const [loading, setLoading] = useState(false);
  const parentRef = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(3);
  const rows = useMemo(
    () =>
      Array.from(
        { length: Math.ceil(result.examples.length / columns) },
        (_, i) => result.examples.slice(i * columns, i * columns + columns),
      ),
    [result.examples, columns],
  );
  useEffect(() => {
    const element = parentRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) =>
      setColumns(
        entry.contentRect.width < 650
          ? 1
          : entry.contentRect.width < 1050
            ? 2
            : 3,
      ),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  // TanStack Virtual intentionally returns an imperative controller.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 250,
    overscan: 3,
  });
  async function refresh(
    next: {
      search?: string;
      status?: string;
      queue?: string;
      codexDecision?: string;
      disagreement?: boolean;
    } = {},
  ) {
    setLoading(true);
    const s = next.search ?? search,
      st = next.status ?? status,
      q = next.queue ?? queue;
    const cd = next.codexDecision ?? codexDecision;
    const params = new URLSearchParams({
      run: overview.run.id,
      limit: "120",
      status: st,
    });
    if (s) params.set("search", s);
    if (q) params.set("queue", q);
    if (cd !== "all") params.set("codexDecision", cd);
    if (next.disagreement) params.set("disagreement", "true");
    const [examplesRes, overviewRes] = await Promise.all([
      fetch(`/api/examples?${params}`),
      fetch(`/api/overview?run=${overview.run.id}`),
    ]);
    setResult(await examplesRes.json());
    setOverview(await overviewRes.json());
    setLoading(false);
  }
  async function loadMore() {
    if (loading || !result.nextCursor) return;
    setLoading(true);
    const params = new URLSearchParams({
      run: overview.run.id,
      limit: "120",
      status,
      cursor: result.nextCursor,
    });
    if (search) params.set("search", search);
    if (queue) params.set("queue", queue);
    if (codexDecision !== "all") params.set("codexDecision", codexDecision);
    const next = (await fetch(`/api/examples?${params}`).then((r) =>
      r.json(),
    )) as Result;
    setResult((current) => ({
      ...next,
      examples: [...current.examples, ...next.examples],
      total: current.total,
    }));
    setLoading(false);
  }
  async function setExampleStatus(
    example: TrainingExample,
    next: ReviewStatus,
  ) {
    await fetch(`/api/examples/${encodeURIComponent(example.id)}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        version: example.version,
        patch: { status: next },
      }),
    });
    await refresh();
  }
  function chooseQueue(next: string) {
    setQueue(next);
    setStatus(
      next === "unreviewed" ||
        next === "approved" ||
        next === "flagged" ||
        next === "excluded"
        ? next
        : "all",
    );
    void refresh({
      queue: ["long", "duplicate", "suspected-junk"].includes(next) ? next : "",
      status: ["unreviewed", "approved", "flagged", "excluded"].includes(next)
        ? next
        : "all",
    });
  }
  const reviewPct = Math.round(
    ((overview.counts.approved +
      overview.counts.flagged +
      overview.counts.excluded) /
      Math.max(1, overview.counts.total)) *
      100,
  );
  const maxLength = Math.max(...overview.lengths.map((x) => x.count), 1);
  return (
    <div className="page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Private training data</span>
          <h1>Shape the signal.</h1>
          <p>
            Review the whole corpus at a glance, then move from broad patterns
            to the exact words in a single example.
          </p>
        </div>
        <label className="run-select">
          <span className="field-label">Dataset run</span>
          <select
            className="select"
            value={overview.run.id}
            onChange={(e) => router.push(`/?run=${e.target.value}`)}
          >
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.name} · {run.exampleCount.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
      </div>
      <section className="metrics" aria-label="Dataset summary">
        <Metric label="Total examples" value={overview.counts.total} />
        <Metric
          label="Review progress"
          value={`${reviewPct}%`}
          progress={reviewPct}
        />
        <Metric label="Approved" value={overview.counts.approved} />
        <Metric label="Flagged" value={overview.counts.flagged} />
        <Metric label="Excluded" value={overview.counts.excluded} />
        <Metric label="Active output" value={overview.counts.active} />
        <Metric label="Codex reviewed" value={overview.codex.reviewed} />
      </section>
      <div className="overview-row">
        <section className="panel">
          <div className="panel-head">
            <h2>Review queues</h2>
            <Link href="/pipeline" className="button ghost">
              Open pipeline <ChevronRight size={14} />
            </Link>
          </div>
          <div className="queues">
            <Queue
              label="Unreviewed"
              count={overview.counts.unreviewed}
              onClick={() => chooseQueue("unreviewed")}
            />
            <Queue
              label="Approved"
              count={overview.counts.approved}
              onClick={() => chooseQueue("approved")}
            />
            <Queue
              label="Flagged"
              count={overview.counts.flagged}
              onClick={() => chooseQueue("flagged")}
            />
            <Queue
              label="Very long"
              count={overview.queues.long}
              onClick={() => chooseQueue("long")}
            />
            <Queue
              label="Duplicates"
              count={overview.queues.duplicate}
              onClick={() => chooseQueue("duplicate")}
            />
            <Queue
              label="Suspected junk"
              count={overview.queues.suspectedJunk}
              onClick={() => chooseQueue("suspected-junk")}
            />
            <Queue
              label="Codex exclude"
              count={overview.codex.exclude}
              onClick={() => {
                setCodexDecision("exclude");
                void refresh({ codexDecision: "exclude" });
              }}
            />
            <Queue
              label="Codex unsure"
              count={overview.codex.unsure}
              onClick={() => {
                setCodexDecision("unsure");
                void refresh({ codexDecision: "unsure" });
              }}
            />
            <Queue
              label="AI disagreements"
              count={overview.codex.disagreements}
              onClick={() => void refresh({ disagreement: true })}
            />
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Combined length</h2>
          </div>
          <div className="mini-chart">
            {overview.lengths.map((item) => (
              <div className="bar-wrap" key={item.bucket}>
                <div
                  className="bar"
                  style={{
                    height: `${Math.max(5, (item.count / maxLength) * 100)}%`,
                  }}
                  title={`${item.count} examples`}
                />
                <span>{item.bucket}</span>
              </div>
            ))}
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Source share</h2>
          </div>
          <div className="source-bars">
            {overview.sources.slice(0, 5).map((source) => (
              <div className="source-bar" key={source.source}>
                <div>
                  <span>
                    {source.source === "unlinked"
                      ? "Unlinked baseline"
                      : `Source ${source.source}`}
                  </span>
                  <b>{Math.round(source.share * 100)}%</b>
                </div>
                <i>
                  <span
                    style={{ width: `${Math.max(2, source.share * 100)}%` }}
                  />
                </i>
              </div>
            ))}
          </div>
        </section>
      </div>
      <section className="panel">
        <div className="toolbar">
          <form
            className="search-wrap"
            onSubmit={(e) => {
              e.preventDefault();
              void refresh();
            }}
          >
            <Search size={16} />
            <input
              className="input"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search prompts and replies…"
              aria-label="Search examples"
            />
          </form>
          <select
            className="select"
            value={status}
            onChange={(e) => {
              setStatus(e.target.value);
              setQueue("");
              void refresh({ status: e.target.value, queue: "" });
            }}
            aria-label="Review status"
          >
            <option value="all">All states</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="approved">Approved</option>
            <option value="flagged">Flagged</option>
            <option value="excluded">Excluded</option>
          </select>
          <select
            className="select"
            value={codexDecision}
            onChange={(e) => {
              setCodexDecision(e.target.value);
              void refresh({ codexDecision: e.target.value });
            }}
            aria-label="Codex decision"
          >
            <option value="all">All Codex</option>
            <option value="include">Codex include</option>
            <option value="exclude">Codex exclude</option>
            <option value="unsure">Codex unsure</option>
            <option value="unreviewed">Not AI reviewed</option>
            <option value="stale">Stale AI review</option>
          </select>
          <select
            className="select"
            value={queue}
            onChange={(e) => {
              setQueue(e.target.value);
              void refresh({ queue: e.target.value });
            }}
            aria-label="Issue queue"
          >
            <option value="">All examples</option>
            <option value="long">Very long</option>
            <option value="duplicate">Duplicates</option>
            <option value="suspected-junk">Suspected junk</option>
          </select>
          <span className="result-meta">
            {loading
              ? "Refreshing…"
              : `${result.total.toLocaleString()} matches`}
          </span>
        </div>
        <div
          className="example-viewport"
          ref={parentRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            if (el.scrollHeight - el.scrollTop - el.clientHeight < 600)
              void loadMore();
          }}
        >
          {rows.length ? (
            <div
              style={{
                height: virtualizer.getTotalSize(),
                position: "relative",
              }}
            >
              {virtualizer.getVirtualItems().map((item) => (
                <div
                  className="virtual-row"
                  key={item.key}
                  style={{
                    height: item.size,
                    transform: `translateY(${item.start}px)`,
                  }}
                >
                  <div
                    className="example-grid-row"
                    style={{ "--columns": columns } as React.CSSProperties}
                  >
                    {rows[item.index].map((example) => (
                      <ExampleCard
                        key={example.id}
                        example={example}
                        setStatus={setExampleStatus}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">
              <div>
                <b>No examples found</b>Try changing the search or review queue.
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
function Metric({
  label,
  value,
  progress,
}: {
  label: string;
  value: string | number;
  progress?: number;
}) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>
      </div>
      <strong>
        {typeof value === "number" ? value.toLocaleString() : value}
      </strong>
      {progress != null && (
        <div className="progress">
          <i style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}
function Queue({
  label,
  count,
  onClick,
}: {
  label: string;
  count: number;
  onClick: () => void;
}) {
  return (
    <button className="queue-card" onClick={onClick}>
      <span>{label}</span>
      <b>{count.toLocaleString()}</b>
    </button>
  );
}
function ExampleCard({
  example,
  setStatus,
}: {
  example: TrainingExample;
  setStatus: (example: TrainingExample, status: ReviewStatus) => void;
}) {
  return (
    <article
      className="example-card"
      tabIndex={0}
      aria-label={`Training example, ${example.status}. Press A to approve, F to flag, X to exclude.`}
      onKeyDown={(e) => {
        if (e.key.toLowerCase() === "a") void setStatus(example, "approved");
        if (e.key.toLowerCase() === "f") void setStatus(example, "flagged");
        if (e.key.toLowerCase() === "x") void setStatus(example, "excluded");
        if (e.key === "Enter")
          window.location.href = `/examples/${encodeURIComponent(example.id)}`;
      }}
    >
      <div className="card-top">
        <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
          <span className={`status-pill ${example.status}`}>
            Human · {example.status}
          </span>
          {example.codexReview && (
            <span
              className={`status-pill codex-${example.codexReview.stale ? "stale" : example.codexReview.decision}`}
            >
              Codex ·{" "}
              {example.codexReview.stale
                ? "stale"
                : example.codexReview.decision}
            </span>
          )}
        </div>
        <div className="card-actions">
          <Link
            className="inspect-link"
            href={`/examples/${encodeURIComponent(example.id)}`}
            onKeyDown={(event) => event.stopPropagation()}
          >
            Inspect <ArrowUpRight size={12} />
          </Link>
          <button
            title="Approve"
            aria-label="Approve"
            onClick={() => setStatus(example, "approved")}
          >
            <Check size={15} />
          </button>
          <button
            title="Flag"
            aria-label="Flag"
            onClick={() => setStatus(example, "flagged")}
          >
            <Flag size={15} />
          </button>
          <button
            title="Exclude"
            aria-label="Exclude"
            onClick={() => setStatus(example, "excluded")}
          >
            <Trash2 size={15} />
          </button>
        </div>
      </div>
      <div className="message">
        <label>User</label>
        <p>{example.user}</p>
      </div>
      <div className="message assistant">
        <label>Assistant</label>
        <p>{example.assistant}</p>
      </div>
      <div className="card-foot">
        <span>
          {(example.user.length + example.assistant.length).toLocaleString()}{" "}
          chars
        </span>
        {example.split && <span className="tag">{example.split}</span>}
        {example.tags.slice(0, 1).map((t) => (
          <span className="tag" key={t}>
            {t}
          </span>
        ))}
      </div>
    </article>
  );
}
