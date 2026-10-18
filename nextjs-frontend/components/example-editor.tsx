"use client";
import { ArrowLeft, RotateCcw, Save } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import type { Operation, ReviewStatus, TrainingExample } from "@/lib/types";
export function ExampleEditor({
  initialExample,
  operations,
}: {
  initialExample: TrainingExample;
  operations: Operation[];
}) {
  const [example, setExample] = useState(initialExample);
  const [form, setForm] = useState({
    user: example.user,
    assistant: example.assistant,
    status: example.status,
    tags: example.tags.join(", "),
    notes: example.notes,
  });
  const [message, setMessage] = useState("");
  const dirty =
    form.user !== example.user ||
    form.assistant !== example.assistant ||
    form.status !== example.status ||
    form.tags !== example.tags.join(", ") ||
    form.notes !== example.notes;
  async function save() {
    const data = await fetch(
      `/api/examples/${encodeURIComponent(example.id)}`,
      {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          version: example.version,
          patch: {
            user: form.user,
            assistant: form.assistant,
            status: form.status,
            tags: form.tags
              .split(",")
              .map((x) => x.trim())
              .filter(Boolean),
            notes: form.notes,
          },
        }),
      },
    ).then((r) => r.json());
    if (data.error) return setMessage(data.error);
    setExample(data);
    setForm({
      user: data.user,
      assistant: data.assistant,
      status: data.status,
      tags: data.tags.join(", "),
      notes: data.notes,
    });
    setMessage("Saved as a reversible edit.");
  }
  async function acceptCodex() {
    if (
      !example.codexReview ||
      example.codexReview.stale ||
      example.codexReview.decision === "unsure"
    )
      return;
    const status =
      example.codexReview.decision === "include" ? "approved" : "excluded";
    const data = await fetch(
      `/api/examples/${encodeURIComponent(example.id)}`,
      {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ version: example.version, patch: { status } }),
      },
    ).then((r) => r.json());
    if (data.error) return setMessage(data.error);
    setExample(data);
    setForm((current) => ({ ...current, status: data.status }));
    setMessage(
      `Accepted Codex’s ${example.codexReview.decision} decision as your human decision.`,
    );
  }
  return (
    <div className="page">
      <div className="page-title">
        <div>
          <Link
            href={`/?run=${example.runId}`}
            className="button ghost"
            style={{ paddingLeft: 0 }}
          >
            <ArrowLeft size={15} />
            Back to overview
          </Link>
          <h1>Inspect one example</h1>
          <p>
            Edit the training text while keeping the immutable source and
            private provenance close at hand.
          </p>
        </div>
        <button className="button" disabled={!dirty} onClick={save}>
          <Save size={15} />
          Save changes
        </button>
      </div>
      <div className="detail-grid">
        <section className="panel detail-main">
          <div className="editor-block">
            <div className="editor-label">
              <b>User prompt</b>
              <span>{form.user.length} characters</span>
            </div>
            <textarea
              className="textarea detail-textarea"
              value={form.user}
              onChange={(e) => setForm({ ...form, user: e.target.value })}
            />
            {example.originalUser !== form.user && (
              <details>
                <summary
                  className="field-label"
                  style={{ cursor: "pointer", marginTop: 8 }}
                >
                  Show immutable original
                </summary>
                <div className="original-box">{example.originalUser}</div>
              </details>
            )}
          </div>
          <div className="editor-block">
            <div className="editor-label">
              <b>Assistant reply</b>
              <span>{form.assistant.length} characters</span>
            </div>
            <textarea
              className="textarea detail-textarea"
              value={form.assistant}
              onChange={(e) => setForm({ ...form, assistant: e.target.value })}
            />
            {example.originalAssistant !== form.assistant && (
              <details>
                <summary
                  className="field-label"
                  style={{ cursor: "pointer", marginTop: 8 }}
                >
                  Show immutable original
                </summary>
                <div className="original-box">{example.originalAssistant}</div>
              </details>
            )}
          </div>
          {message && <div className="notice">{message}</div>}
        </section>
        <aside className="panel detail-aside">
          <section>
            <h3>Curation</h3>
            <label>
              <span className="field-label">Review state</span>
              <select
                className="select"
                value={form.status}
                onChange={(e) =>
                  setForm({ ...form, status: e.target.value as ReviewStatus })
                }
              >
                <option value="unreviewed">Unreviewed</option>
                <option value="approved">Approved</option>
                <option value="flagged">Flagged</option>
                <option value="excluded">Excluded</option>
              </select>
            </label>
            <label style={{ display: "block", marginTop: 12 }}>
              <span className="field-label">Tags · comma separated</span>
              <input
                className="input"
                value={form.tags}
                onChange={(e) => setForm({ ...form, tags: e.target.value })}
              />
            </label>
            <label style={{ display: "block", marginTop: 12 }}>
              <span className="field-label">Private notes</span>
              <textarea
                className="textarea"
                rows={4}
                value={form.notes}
                onChange={(e) => setForm({ ...form, notes: e.target.value })}
              />
            </label>
          </section>
          <section>
            <h3>Codex review</h3>
            {example.codexReview ? (
              <>
                {example.codexReview.stale && (
                  <div className="notice warning">
                    This review is stale because the training text changed.
                  </div>
                )}
                <dl className="meta-list">
                  <dt>Decision</dt>
                  <dd>
                    <span
                      className={`status-pill codex-${example.codexReview.stale ? "stale" : example.codexReview.decision}`}
                    >
                      {example.codexReview.stale
                        ? "stale"
                        : example.codexReview.decision}
                    </span>
                  </dd>
                  <dt>Confidence</dt>
                  <dd>{Math.round(example.codexReview.confidence * 100)}%</dd>
                  <dt>Reasons</dt>
                  <dd>{example.codexReview.reasons.join(", ") || "—"}</dd>
                  <dt>Rubric</dt>
                  <dd>{example.codexReview.rubricVersion}</dd>
                  <dt>Model</dt>
                  <dd>{example.codexReview.model}</dd>
                  <dt>Reviewed</dt>
                  <dd>
                    {new Date(example.codexReview.reviewedAt).toLocaleString()}
                  </dd>
                </dl>
                <div className="original-box">
                  {example.codexReview.rationale}
                </div>
                {!example.codexReview.stale &&
                  example.codexReview.decision !== "unsure" && (
                    <button
                      className="button secondary"
                      style={{ marginTop: 12 }}
                      onClick={acceptCodex}
                    >
                      Accept Codex decision
                    </button>
                  )}
              </>
            ) : (
              <p className="result-meta">Not reviewed by Codex yet.</p>
            )}
          </section>
          <section>
            <h3>Private provenance</h3>
            <dl className="meta-list">
              <dt>Opaque source</dt>
              <dd>{example.chatId ?? "Unlinked"}</dd>
              <dt>Inbound IDs</dt>
              <dd>{example.inboundMessageIds.join(", ") || "Unavailable"}</dd>
              <dt>Reply IDs</dt>
              <dd>{example.replyMessageIds.join(", ") || "Unavailable"}</dd>
              <dt>Timestamp</dt>
              <dd>
                {example.startTimestamp
                  ? new Date(example.startTimestamp * 1000).toLocaleString()
                  : "Unavailable"}
              </dd>
              <dt>Session</dt>
              <dd>{example.sessionKey}</dd>
              <dt>Split</dt>
              <dd>{example.split ?? "Unassigned"}</dd>
              <dt>Version</dt>
              <dd>{example.version}</dd>
            </dl>
          </section>
          <section>
            <h3>Recent run history</h3>
            <div className="history-list">
              {operations.slice(0, 6).map((op) => (
                <div className="history-item" key={op.id}>
                  <div>
                    <b>{op.description}</b>
                    <br />
                    <span>
                      {new Date(op.createdAt).toLocaleString()} ·{" "}
                      {op.affectedCount} changed
                    </span>
                  </div>
                  {!op.revertedAt && (
                    <button
                      className="button ghost"
                      title="Revert operation"
                      onClick={async () => {
                        const data = await fetch(
                          `/api/operations/${op.id}/revert`,
                          { method: "POST" },
                        ).then((r) => r.json());
                        setMessage(
                          data.error ??
                            `Reverted ${data.reverted} changes. Reload to see the restored version.`,
                        );
                      }}
                    >
                      <RotateCcw size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
