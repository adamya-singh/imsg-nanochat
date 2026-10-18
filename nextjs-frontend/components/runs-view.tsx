"use client";
import Link from "next/link";
import { useState } from "react";
import type { DatasetRun, Overview } from "@/lib/types";
export function RunsView({ runs }: { runs: DatasetRun[] }) {
  const completed = runs.filter((r) => r.status === "completed");
  const [left, setLeft] = useState(completed[1]?.id ?? completed[0]?.id ?? "");
  const [right, setRight] = useState(completed[0]?.id ?? "");
  const [comparison, setComparison] = useState<{
    left: Overview;
    right: Overview;
  } | null>(null);
  async function compare() {
    const data = await fetch(
      `/api/runs/compare?left=${left}&right=${right}`,
    ).then((r) => r.json());
    if (!data.error) setComparison(data);
  }
  return (
    <div className="page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Immutable lineage</span>
          <h1>Run history</h1>
          <p>
            Compare extraction snapshots, review coverage, filtering losses, and
            output assignments without losing the decisions attached to earlier
            data.
          </p>
        </div>
        <Link href="/pipeline" className="button">
          Create new run
        </Link>
      </div>
      <section className="panel" style={{ marginBottom: 18 }}>
        <div className="panel-head">
          <h2>All runs</h2>
          <span className="result-meta">{runs.length} snapshots</span>
        </div>
        <table className="runs-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Examples</th>
              <th>Created</th>
              <th>Parent</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td>
                  <b>{run.name}</b>
                  <br />
                  <span className="result-meta">{run.id}</span>
                </td>
                <td>
                  <span
                    className={`status-pill ${run.status === "completed" ? "approved" : run.status === "failed" ? "excluded" : "flagged"}`}
                  >
                    {run.status}
                  </span>
                </td>
                <td>{run.exampleCount.toLocaleString()}</td>
                <td>{new Date(run.createdAt).toLocaleString()}</td>
                <td>{run.parentRunId ?? "—"}</td>
                <td>
                  {run.status === "completed" && (
                    <Link className="button ghost" href={`/?run=${run.id}`}>
                      Open →
                    </Link>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel">
        <div className="panel-head">
          <h2>Compare two runs</h2>
        </div>
        <div style={{ padding: 18 }}>
          <div className="form-grid">
            <label>
              <span className="field-label">Earlier run</span>
              <select
                className="select"
                value={left}
                onChange={(e) => setLeft(e.target.value)}
              >
                {completed.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="field-label">Later run</span>
              <select
                className="select"
                value={right}
                onChange={(e) => setRight(e.target.value)}
              >
                {completed.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="form-actions">
            <button className="button secondary" onClick={compare}>
              Compare snapshots
            </button>
          </div>
          {comparison && (
            <table className="source-table" style={{ marginTop: 18 }}>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>{comparison.left.run.name}</th>
                  <th>{comparison.right.run.name}</th>
                  <th>Change</th>
                </tr>
              </thead>
              <tbody>
                {[
                  [
                    "Examples",
                    comparison.left.counts.total,
                    comparison.right.counts.total,
                  ],
                  [
                    "Approved",
                    comparison.left.counts.approved,
                    comparison.right.counts.approved,
                  ],
                  [
                    "Flagged",
                    comparison.left.counts.flagged,
                    comparison.right.counts.flagged,
                  ],
                  [
                    "Excluded",
                    comparison.left.counts.excluded,
                    comparison.right.counts.excluded,
                  ],
                  [
                    "Active output",
                    comparison.left.counts.active,
                    comparison.right.counts.active,
                  ],
                ].map(([label, a, b]) => (
                  <tr key={String(label)}>
                    <td>{label}</td>
                    <td>{Number(a).toLocaleString()}</td>
                    <td>{Number(b).toLocaleString()}</td>
                    <td>
                      {Number(b) - Number(a) > 0 ? "+" : ""}
                      {(Number(b) - Number(a)).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}
