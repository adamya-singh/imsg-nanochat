"use client";
import { Bot, CirclePause, CirclePlay, RotateCcw, Square } from "lucide-react";
import { useEffect, useState } from "react";
import type { CodexReviewJob, DatasetRun, Overview } from "@/lib/types";

export function AiReview({
  runs,
  initialRunId,
  initialOverview,
  initialJobs,
}: {
  runs: DatasetRun[];
  initialRunId: string;
  initialOverview: Overview;
  initialJobs: CodexReviewJob[];
}) {
  const [runId, setRunId] = useState(initialRunId);
  const [overview, setOverview] = useState(initialOverview);
  const [jobs, setJobs] = useState(initialJobs);
  const [model, setModel] = useState("gpt-5.4-mini");
  const [message, setMessage] = useState("");
  const active = jobs.find((j) =>
    ["queued", "running", "paused", "failed"].includes(j.status),
  );
  async function refresh(id = runId) {
    const [j, o] = await Promise.all([
      fetch(`/api/codex-review/jobs?run=${id}`).then((r) => r.json()),
      fetch(`/api/overview?run=${id}`).then((r) => r.json()),
    ]);
    setJobs(j.jobs ?? []);
    if (!o.error) setOverview(o);
  }
  useEffect(() => {
    if (!jobs.some((j) => ["queued", "running"].includes(j.status))) return;
    const timer = setInterval(() => void refresh(), 1500);
    return () => clearInterval(timer);
  });
  async function changeRun(id: string) {
    setRunId(id);
    await refresh(id);
  }
  async function start(scope: "pilot" | "full") {
    setMessage("");
    const data = await fetch("/api/codex-review/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ runId, scope, model }),
    }).then((r) => r.json());
    setMessage(
      data.error ??
        `${scope === "pilot" ? "Pilot" : "Full"} review started with ${model}.`,
    );
    await refresh();
  }
  async function control(action: "pause" | "resume" | "retry" | "cancel") {
    if (!active) return;
    const data = await fetch(`/api/codex-review/jobs/${active.id}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action }),
    }).then((r) => r.json());
    setMessage(data.error ?? `Job ${action} requested.`);
    await refresh();
  }
  return (
    <div className="page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Structured Codex judgments</span>
          <h1>AI review</h1>
          <p>
            Review every pair for training quality without overwriting your
            decisions. Human approvals, flags, and exclusions always take
            precedence.
          </p>
        </div>
        <label className="run-select">
          <span className="field-label">Dataset run</span>
          <select
            className="select"
            value={runId}
            onChange={(e) => void changeRun(e.target.value)}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="notice warning" style={{ marginBottom: 18 }}>
        <b>Privacy notice:</b> starting a job sends prompt and reply text to
        OpenAI through your authenticated Codex CLI. Reviews judge training
        quality only; personal or sensitive content is not excluded solely for
        being private.
      </div>
      <section className="metrics">
        <ReviewMetric
          label="Reviewed"
          value={`${overview.codex.reviewed.toLocaleString()} / ${overview.counts.total.toLocaleString()}`}
        />
        <ReviewMetric
          label="Include"
          value={overview.codex.include.toLocaleString()}
        />
        <ReviewMetric
          label="Exclude"
          value={overview.codex.exclude.toLocaleString()}
        />
        <ReviewMetric
          label="Unsure"
          value={overview.codex.unsure.toLocaleString()}
        />
        <ReviewMetric
          label="Stale"
          value={overview.codex.stale.toLocaleString()}
        />
        <ReviewMetric
          label="Disagreements"
          value={overview.codex.disagreements.toLocaleString()}
        />
      </section>
      <div
        className="overview-row"
        style={{ gridTemplateColumns: "minmax(0,1.2fr) minmax(330px,.8fr)" }}
      >
        <section className="panel" style={{ padding: 22 }}>
          <h2 style={{ fontFamily: "Georgia,serif", fontWeight: 500 }}>
            Start a review
          </h2>
          <p style={{ color: "var(--muted)", lineHeight: 1.55 }}>
            Start with 100 examples to calibrate the rubric, then continue
            through every remaining current example. Existing valid reviews are
            skipped.
          </p>
          <label>
            <span className="field-label">Codex model</span>
            <input
              className="input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              list="codex-models"
            />
            <datalist id="codex-models">
              <option value="gpt-5.4-mini" />
              <option value="gpt-5.4" />
              <option value="gpt-5.5" />
            </datalist>
          </label>
          <div className="form-actions">
            <button
              className="button secondary"
              disabled={!!active}
              onClick={() => start("pilot")}
            >
              <Bot size={15} />
              Pilot 100
            </button>
            <button
              className="button"
              disabled={!!active}
              onClick={() => start("full")}
            >
              <Bot size={15} />
              Review all remaining
            </button>
          </div>
          {message && (
            <div className="notice" style={{ marginTop: 15 }}>
              {message}
            </div>
          )}
        </section>
        <section className="panel" style={{ padding: 22 }}>
          <h2 style={{ fontFamily: "Georgia,serif", fontWeight: 500 }}>
            Current job
          </h2>
          {active ? (
            <JobCard job={active} onControl={control} />
          ) : (
            <p className="result-meta">No active review job.</p>
          )}
        </section>
      </div>
      <section className="panel">
        <div className="panel-head">
          <h2>Job history</h2>
          <span className="result-meta">Batch size 50 · one at a time</span>
        </div>
        <table className="runs-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Scope</th>
              <th>Model</th>
              <th>Progress</th>
              <th>Decisions</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id}>
                <td>
                  <span
                    className={`status-pill ${j.status === "completed" ? "approved" : j.status === "failed" ? "excluded" : "flagged"}`}
                  >
                    {j.status}
                  </span>
                </td>
                <td>{j.scope}</td>
                <td>{j.model}</td>
                <td>
                  {j.processedCount.toLocaleString()} /{" "}
                  {j.totalCount.toLocaleString()}
                </td>
                <td>
                  {j.includeCount} include · {j.excludeCount} exclude ·{" "}
                  {j.unsureCount} unsure
                </td>
                <td>{new Date(j.updatedAt).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
function ReviewMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
    </div>
  );
}
function JobCard({
  job,
  onControl,
}: {
  job: CodexReviewJob;
  onControl: (action: "pause" | "resume" | "retry" | "cancel") => void;
}) {
  const pct = Math.round(
    (job.processedCount / Math.max(1, job.totalCount)) * 100,
  );
  return (
    <div>
      <div className="metric" style={{ padding: 0, border: 0 }}>
        <strong>{pct}%</strong>
        <div className="progress">
          <i style={{ width: `${pct}%` }} />
        </div>
      </div>
      <p className="result-meta">
        Batch {job.currentBatch + 1} · {job.processedCount.toLocaleString()}{" "}
        processed · {job.retryCount} retries
      </p>
      {job.error && <div className="notice warning">{job.error}</div>}
      <div className="form-actions">
        {job.status === "running" && (
          <button
            className="button secondary"
            onClick={() => onControl("pause")}
          >
            <CirclePause size={14} />
            Pause
          </button>
        )}
        {job.status === "paused" && (
          <button className="button" onClick={() => onControl("resume")}>
            <CirclePlay size={14} />
            Resume
          </button>
        )}
        {job.status === "failed" && (
          <button className="button" onClick={() => onControl("retry")}>
            <RotateCcw size={14} />
            Retry
          </button>
        )}
        {!["completed", "cancelled"].includes(job.status) && (
          <button className="button danger" onClick={() => onControl("cancel")}>
            <Square size={13} />
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
