"use client";
import {
  Archive,
  CheckCircle2,
  Database,
  Download,
  RefreshCw,
  Scale,
  Scissors,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import type {
  BalanceConfig,
  DatasetRun,
  Overview,
  SplitConfig,
  TransformationPreview,
  TransformationSpec,
} from "@/lib/types";

const stages = [
  {
    id: "extract",
    label: "Extract",
    hint: "Version the source",
    icon: Database,
  },
  {
    id: "curate",
    label: "Curate",
    hint: "Review the signal",
    icon: CheckCircle2,
  },
  {
    id: "transform",
    label: "Transform",
    hint: "Apply safe rules",
    icon: Sparkles,
  },
  { id: "balance", label: "Balance", hint: "Shape sources", icon: Scale },
  { id: "split", label: "Split", hint: "Prevent leakage", icon: Scissors },
  { id: "export", label: "Export", hint: "Publish artifacts", icon: Download },
] as const;
type Stage = (typeof stages)[number]["id"];
export function PipelineStudio({
  runs: initialRuns,
  initialOverview,
}: {
  runs: DatasetRun[];
  initialOverview: Overview;
}) {
  const [runs, setRuns] = useState(initialRuns);
  const [runId, setRunId] = useState(initialOverview.run.id);
  const [overview, setOverview] = useState(initialOverview);
  const [stage, setStage] = useState<Stage>("extract");
  const [message, setMessage] = useState("");
  async function reload() {
    const [r, o] = await Promise.all([
      fetch("/api/runs").then((x) => x.json()),
      fetch(`/api/overview?run=${runId}`).then((x) => x.json()),
    ]);
    setRuns(r.runs);
    if (!o.error) setOverview(o);
  }
  useEffect(() => {
    const active = runs.some((r) => r.status === "processing");
    if (!active) return;
    const id = setInterval(reload, 1500);
    return () => clearInterval(id);
  });
  async function changeRun(id: string) {
    setRunId(id);
    const data = await fetch(`/api/overview?run=${id}`).then((r) => r.json());
    if (!data.error) setOverview(data);
  }
  return (
    <div className="page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Guided data workflow</span>
          <h1>Pipeline studio</h1>
          <p>
            Each stage is inspectable and reversible. Raw messages stay
            untouched; only a versioned workspace moves forward.
          </p>
        </div>
        <label className="run-select">
          <span className="field-label">Working run</span>
          <select
            className="select"
            value={runId}
            onChange={(e) => void changeRun(e.target.value)}
          >
            {runs
              .filter((r) => r.status === "completed")
              .map((r) => (
                <option value={r.id} key={r.id}>
                  {r.name}
                </option>
              ))}
          </select>
        </label>
      </div>
      <div className="pipeline-shell">
        <aside className="panel stage-list">
          {stages.map((item, i) => (
            <button
              className={`stage-button ${stage === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => {
                setStage(item.id);
                setMessage("");
              }}
            >
              <span className="stage-number">{i + 1}</span>
              <span>
                <b>{item.label}</b>
                <span>{item.hint}</span>
              </span>
            </button>
          ))}
        </aside>
        <section className="panel stage-content">
          {stage === "extract" && (
            <ExtractStage
              runs={runs}
              onStarted={async (text) => {
                setMessage(text);
                await reload();
              }}
            />
          )}
          {stage === "curate" && <CurateStage overview={overview} />}{" "}
          {stage === "transform" && (
            <TransformStage
              key={runId}
              runId={runId}
              onDone={async (text) => {
                setMessage(text);
                await reload();
              }}
            />
          )}
          {stage === "balance" && (
            <BalanceStage
              runId={runId}
              onDone={async (text) => {
                setMessage(text);
                await reload();
              }}
            />
          )}
          {stage === "split" && (
            <SplitStage
              runId={runId}
              overview={overview}
              onDone={async (text) => {
                setMessage(text);
                await reload();
              }}
            />
          )}
          {stage === "export" && (
            <ExportStage
              runId={runId}
              overview={overview}
              onDone={setMessage}
            />
          )}{" "}
          {message && (
            <div className="notice" style={{ marginTop: 18 }}>
              {message}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
function ExtractStage({
  runs,
  onStarted,
}: {
  runs: DatasetRun[];
  onStarted: (m: string) => void;
}) {
  const [form, setForm] = useState({
    name: "",
    dbPath: "",
    minContactPairs: 5,
    mergeGapSeconds: 600,
    seed: 42,
    limitChats: "",
  });
  const [busy, setBusy] = useState(false);
  const active = runs.find((r) => r.status === "processing");
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    const res = await fetch("/api/runs/extract", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        ...form,
        dbPath: form.dbPath || undefined,
        limitChats: form.limitChats ? Number(form.limitChats) : null,
        parentRunId: runs.find((r) => r.status === "completed")?.id,
      }),
    });
    const data = await res.json();
    setBusy(false);
    onStarted(
      data.error ??
        `Extraction ${data.runId} started. You can leave this page while it runs.`,
    );
  }
  return (
    <>
      <h2>Extract a versioned snapshot</h2>
      <p>
        Read the Apple Messages database with the existing precision-first
        rules. Output and provenance are written privately to a new immutable
        run.
      </p>
      {active && (
        <div className="notice warning" style={{ marginBottom: 18 }}>
          Extraction in progress: <b>{active.name}</b>. The run appears only
          after all files publish successfully.
        </div>
      )}
      <form onSubmit={submit}>
        <div className="form-grid">
          <label>
            <span className="field-label">Run name</span>
            <input
              className="input"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="e.g. July cleanup"
            />
          </label>
          <label>
            <span className="field-label">chat.db path</span>
            <input
              className="input"
              value={form.dbPath}
              onChange={(e) => setForm({ ...form, dbPath: e.target.value })}
              placeholder="Default: training-data/raw/chat.db"
            />
          </label>
          <label>
            <span className="field-label">Minimum pairs per source</span>
            <input
              className="input"
              type="number"
              min="1"
              value={form.minContactPairs}
              onChange={(e) =>
                setForm({ ...form, minContactPairs: Number(e.target.value) })
              }
            />
          </label>
          <label>
            <span className="field-label">Turn merge gap · seconds</span>
            <input
              className="input"
              type="number"
              min="0"
              value={form.mergeGapSeconds}
              onChange={(e) =>
                setForm({ ...form, mergeGapSeconds: Number(e.target.value) })
              }
            />
          </label>
          <label>
            <span className="field-label">Deterministic seed</span>
            <input
              className="input"
              type="number"
              value={form.seed}
              onChange={(e) =>
                setForm({ ...form, seed: Number(e.target.value) })
              }
            />
          </label>
          <label>
            <span className="field-label">Limit sources · optional</span>
            <input
              className="input"
              type="number"
              min="1"
              value={form.limitChats}
              onChange={(e) => setForm({ ...form, limitChats: e.target.value })}
              placeholder="All sources"
            />
          </label>
        </div>
        <div className="form-actions">
          <button className="button" disabled={busy || !!active}>
            <Database size={16} />
            {busy ? "Starting…" : "Start extraction"}
          </button>
          {active && (
            <button
              type="button"
              className="button danger"
              onClick={async () => {
                await fetch(`/api/runs/${active.id}/cancel`, {
                  method: "POST",
                });
                onStarted("Cancellation requested.");
              }}
            >
              Cancel run
            </button>
          )}
        </div>
      </form>
    </>
  );
}
function CurateStage({ overview }: { overview: Overview }) {
  const reviewed =
    overview.counts.approved +
    overview.counts.flagged +
    overview.counts.excluded;
  return (
    <>
      <h2>Review the training signal</h2>
      <p>
        Move through inbox-style queues, edit the exact prompt and reply, and
        leave private notes without changing the source snapshot.
      </p>
      <div className="metrics" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
        <SmallStat
          label="Reviewed"
          value={`${reviewed.toLocaleString()} / ${overview.counts.total.toLocaleString()}`}
        />
        <SmallStat
          label="Approved"
          value={overview.counts.approved.toLocaleString()}
        />
        <SmallStat
          label="Needs attention"
          value={overview.counts.flagged.toLocaleString()}
        />
      </div>
      <div className="notice">
        Edits use version checks, so a second browser tab cannot silently
        overwrite newer work.
      </div>
      <div className="form-actions">
        <Link href={`/?run=${overview.run.id}`} className="button">
          Open review workspace
        </Link>
      </div>
    </>
  );
}
function TransformStage({
  runId,
  onDone,
}: {
  runId: string;
  onDone: (m: string) => void;
}) {
  const [spec, setSpec] = useState<TransformationSpec>({
    kind: "normalize",
    query: { runId, status: "all" },
    field: "both",
  });
  const [preview, setPreview] = useState<TransformationPreview | null>(null);
  const [busy, setBusy] = useState(false);
  async function previewRule() {
    setBusy(true);
    const data = await fetch("/api/transformations/preview", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => r.json());
    setBusy(false);
    if (data.error) onDone(data.error);
    else setPreview(data);
  }
  async function apply() {
    if (!preview) return;
    setBusy(true);
    const data = await fetch("/api/transformations/apply", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => r.json());
    setBusy(false);
    setPreview(null);
    onDone(
      data.error ??
        `Applied to ${data.affectedCount.toLocaleString()} examples. The operation can be reverted from run history.`,
    );
  }
  return (
    <>
      <h2>Build a safe transformation</h2>
      <p>
        Target the entire run or a review state, preview representative
        before/after pairs, then commit one atomic and reversible operation.
      </p>
      <div className="form-grid">
        <label>
          <span className="field-label">Rule</span>
          <select
            className="select"
            value={spec.kind}
            onChange={(e) => {
              setSpec({
                ...spec,
                kind: e.target.value as TransformationSpec["kind"],
              });
              setPreview(null);
            }}
          >
            <option value="normalize">Normalize whitespace</option>
            <option value="replace">Find and replace</option>
            <option value="set-status">Set review state</option>
            <option value="add-tag">Add tag</option>
            <option value="dedupe">Exclude exact duplicates</option>
          </select>
        </label>
        <label>
          <span className="field-label">Target state</span>
          <select
            className="select"
            value={spec.query.status}
            onChange={(e) =>
              setSpec({
                ...spec,
                query: {
                  ...spec.query,
                  status: e.target
                    .value as TransformationSpec["query"]["status"],
                },
              })
            }
          >
            <option value="all">All examples</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="approved">Approved</option>
            <option value="flagged">Flagged</option>
            <option value="excluded">Excluded</option>
          </select>
        </label>
        {["normalize", "replace"].includes(spec.kind) && (
          <label>
            <span className="field-label">Text field</span>
            <select
              className="select"
              value={spec.field}
              onChange={(e) =>
                setSpec({
                  ...spec,
                  field: e.target.value as TransformationSpec["field"],
                })
              }
            >
              <option value="both">Prompt and reply</option>
              <option value="user">Prompt only</option>
              <option value="assistant">Reply only</option>
            </select>
          </label>
        )}
        {spec.kind === "replace" && (
          <>
            <label>
              <span className="field-label">Find</span>
              <input
                className="input"
                value={spec.find ?? ""}
                onChange={(e) => setSpec({ ...spec, find: e.target.value })}
              />
            </label>
            <label>
              <span className="field-label">Replace with</span>
              <input
                className="input"
                value={spec.replace ?? ""}
                onChange={(e) => setSpec({ ...spec, replace: e.target.value })}
              />
            </label>
            <label style={{ alignSelf: "end" }}>
              <input
                type="checkbox"
                checked={spec.regex ?? false}
                onChange={(e) => setSpec({ ...spec, regex: e.target.checked })}
              />{" "}
              Use regular expression
            </label>
          </>
        )}
        {spec.kind === "set-status" && (
          <label>
            <span className="field-label">New state</span>
            <select
              className="select"
              value={spec.status ?? "approved"}
              onChange={(e) =>
                setSpec({
                  ...spec,
                  status: e.target.value as TransformationSpec["status"],
                })
              }
            >
              <option value="approved">Approved</option>
              <option value="flagged">Flagged</option>
              <option value="excluded">Excluded</option>
              <option value="unreviewed">Unreviewed</option>
            </select>
          </label>
        )}
        {spec.kind === "add-tag" && (
          <label>
            <span className="field-label">Tag</span>
            <input
              className="input"
              value={spec.tag ?? ""}
              onChange={(e) => setSpec({ ...spec, tag: e.target.value })}
            />
          </label>
        )}
      </div>
      <div className="form-actions">
        <button
          className="button secondary"
          disabled={busy}
          onClick={previewRule}
        >
          <RefreshCw size={15} />
          Preview changes
        </button>
        {preview && (
          <button
            className="button"
            disabled={busy || !preview.affectedCount}
            onClick={apply}
          >
            Apply {preview.affectedCount.toLocaleString()} changes
          </button>
        )}
      </div>
      {preview && (
        <div className="preview-list">
          <div className={`notice ${preview.warning ? "warning" : ""}`}>
            {preview.affectedCount.toLocaleString()} examples will change.{" "}
            {preview.warning}
          </div>
          {preview.samples.slice(0, 4).map((item) => (
            <div className="preview-diff" key={item.id}>
              <div>
                <span className="field-label">Before</span>
                <del>
                  {item.before.user}
                  <br />
                  {item.before.assistant}
                </del>
              </div>
              <div>
                <span className="field-label">After</span>
                <ins>
                  {item.after.user}
                  <br />
                  {item.after.assistant}
                </ins>
              </div>
            </div>
          ))}
        </div>
      )}
      <RecipeLibrary
        runId={runId}
        currentSpec={spec}
        onLoad={(saved) => {
          setSpec({ ...saved, query: { ...saved.query, runId } });
          setPreview(null);
          onDone("Recipe step loaded. Preview it before applying.");
        }}
      />
    </>
  );
}

type SavedRecipe = {
  id: string;
  name: string;
  position: number;
  enabled: number;
  spec: string;
};

function RecipeLibrary({
  runId,
  currentSpec,
  onLoad,
}: {
  runId: string;
  currentSpec: TransformationSpec;
  onLoad: (spec: TransformationSpec) => void;
}) {
  const [recipes, setRecipes] = useState<SavedRecipe[]>([]);
  async function refresh() {
    const data = await fetch(`/api/recipes?run=${runId}`).then((r) => r.json());
    setRecipes(data.recipes ?? []);
  }
  useEffect(() => {
    let active = true;
    void fetch(`/api/recipes?run=${runId}`)
      .then((response) => response.json())
      .then((data) => {
        if (active) setRecipes(data.recipes ?? []);
      });
    return () => {
      active = false;
    };
  }, [runId]);
  async function save() {
    const labels: Record<TransformationSpec["kind"], string> = {
      normalize: "Normalize whitespace",
      replace: "Find and replace",
      "set-status": "Set review state",
      "add-tag": "Add tag",
      dedupe: "Exclude duplicates",
    };
    await fetch("/api/recipes", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        runId,
        name: labels[currentSpec.kind],
        spec: currentSpec,
      }),
    });
    await refresh();
  }
  async function patch(id: string, body: Record<string, unknown>) {
    await fetch(`/api/recipes/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    await refresh();
  }
  async function move(index: number, delta: number) {
    const other = recipes[index + delta];
    if (!other) return;
    await Promise.all([
      fetch(`/api/recipes/${recipes[index].id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ position: other.position }),
      }),
      fetch(`/api/recipes/${other.id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ position: recipes[index].position }),
      }),
    ]);
    await refresh();
  }
  async function clone(recipe: SavedRecipe) {
    await fetch("/api/recipes", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        runId,
        name: `${recipe.name} copy`,
        spec: JSON.parse(recipe.spec),
      }),
    });
    await refresh();
  }
  return (
    <div
      style={{
        marginTop: 28,
        borderTop: "1px solid var(--line)",
        paddingTop: 20,
      }}
    >
      <div className="panel-head" style={{ padding: 0, border: 0 }}>
        <div>
          <span className="field-label">Reusable recipe</span>
          <h2 style={{ fontSize: 20 }}>Ordered steps</h2>
        </div>
        <button className="button secondary" onClick={save}>
          Save current rule
        </button>
      </div>
      {!recipes.length ? (
        <p className="result-meta">No saved steps yet.</p>
      ) : (
        <div className="history-list" style={{ marginTop: 12 }}>
          {recipes.map((recipe, index) => (
            <div className="history-item" key={recipe.id}>
              <div>
                <b>
                  {index + 1}. {recipe.name}
                </b>
                <br />
                <span>{recipe.enabled ? "Enabled" : "Disabled"}</span>
              </div>
              <div className="card-actions">
                <button
                  aria-label="Move up"
                  disabled={!index}
                  onClick={() => move(index, -1)}
                >
                  ↑
                </button>
                <button
                  aria-label="Move down"
                  disabled={index === recipes.length - 1}
                  onClick={() => move(index, 1)}
                >
                  ↓
                </button>
                <button
                  onClick={() => patch(recipe.id, { enabled: !recipe.enabled })}
                >
                  {recipe.enabled ? "Disable" : "Enable"}
                </button>
                <button onClick={() => clone(recipe)}>Clone</button>
                <button
                  onClick={() =>
                    onLoad(JSON.parse(recipe.spec) as TransformationSpec)
                  }
                >
                  Load
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
function BalanceStage({
  runId,
  onDone,
}: {
  runId: string;
  onDone: (m: string) => void;
}) {
  const [config, setConfig] = useState<BalanceConfig>({
    maxCount: null,
    maxShare: null,
    seed: 42,
  });
  const [preview, setPreview] = useState<{
    total: number;
    kept: number;
    excluded: number;
    sources: Array<{ source: string; total: number; kept: number }>;
  } | null>(null);
  async function submit(apply: boolean) {
    const data = await fetch("/api/pipeline/balance", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ runId, config, apply }),
    }).then((r) => r.json());
    if (data.error) return onDone(data.error);
    if (apply) {
      setPreview(null);
      onDone(
        `${data.excluded.toLocaleString()} examples are now held out by balancing. The operation is reversible.`,
      );
    } else setPreview(data);
  }
  return (
    <>
      <h2>Balance opaque sources</h2>
      <p>
        Limit outsized source groups without revealing contact identity. Nothing
        is removed unless you explicitly apply a cap.
      </p>
      <div className="form-grid">
        <label>
          <span className="field-label">Maximum per source · optional</span>
          <input
            className="input"
            type="number"
            min="1"
            value={config.maxCount ?? ""}
            onChange={(e) =>
              setConfig({
                ...config,
                maxCount: e.target.value ? Number(e.target.value) : null,
              })
            }
            placeholder="No count cap"
          />
        </label>
        <label>
          <span className="field-label">Maximum share · optional</span>
          <input
            className="input"
            type="number"
            min="0.01"
            max="1"
            step="0.01"
            value={config.maxShare ?? ""}
            onChange={(e) =>
              setConfig({
                ...config,
                maxShare: e.target.value ? Number(e.target.value) : null,
              })
            }
            placeholder="e.g. 0.10"
          />
        </label>
        <label>
          <span className="field-label">Seed</span>
          <input
            className="input"
            type="number"
            value={config.seed}
            onChange={(e) =>
              setConfig({ ...config, seed: Number(e.target.value) })
            }
          />
        </label>
      </div>
      <div className="form-actions">
        <button className="button secondary" onClick={() => submit(false)}>
          Preview balance
        </button>
        {preview && (
          <button className="button" onClick={() => submit(true)}>
            Apply balance
          </button>
        )}
      </div>
      {preview && (
        <>
          <div className="split-preview">
            <SmallStat
              label="Active before"
              value={preview.total.toLocaleString()}
            />
            <SmallStat label="Kept" value={preview.kept.toLocaleString()} />
            <SmallStat
              label="Held out"
              value={preview.excluded.toLocaleString()}
            />
          </div>
          <table className="source-table">
            <thead>
              <tr>
                <th>Opaque source</th>
                <th>Current</th>
                <th>After cap</th>
              </tr>
            </thead>
            <tbody>
              {preview.sources.slice(0, 8).map((row) => (
                <tr key={row.source}>
                  <td>
                    {row.source === "unlinked"
                      ? "Unlinked baseline"
                      : `Source ${row.source}`}
                  </td>
                  <td>{row.total}</td>
                  <td>{row.kept}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  );
}
function SplitStage({
  runId,
  overview,
  onDone,
}: {
  runId: string;
  overview: Overview;
  onDone: (m: string) => void;
}) {
  const [config, setConfig] = useState<SplitConfig>({
    train: 0.9,
    validation: 0.05,
    test: 0.05,
    seed: 42,
    sessionGapHours: 6,
  });
  async function apply() {
    const data = await fetch("/api/pipeline/split", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ runId, config }),
    }).then((r) => r.json());
    onDone(
      data.error ??
        `Assigned ${data.counts.train.toLocaleString()} train, ${data.counts.validation.toLocaleString()} validation, and ${data.counts.test.toLocaleString()} test examples.`,
    );
  }
  return (
    <>
      <h2>Create leakage-safe splits</h2>
      <p>
        Nearby examples from the same opaque chat stay together. Exact
        duplicates are also forced into one split.
      </p>
      <div className="form-grid">
        <label>
          <span className="field-label">Train ratio</span>
          <input
            className="input"
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.train}
            onChange={(e) =>
              setConfig({ ...config, train: Number(e.target.value) })
            }
          />
        </label>
        <label>
          <span className="field-label">Validation ratio</span>
          <input
            className="input"
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.validation}
            onChange={(e) =>
              setConfig({ ...config, validation: Number(e.target.value) })
            }
          />
        </label>
        <label>
          <span className="field-label">Test ratio</span>
          <input
            className="input"
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={config.test}
            onChange={(e) =>
              setConfig({ ...config, test: Number(e.target.value) })
            }
          />
        </label>
        <label>
          <span className="field-label">Seed</span>
          <input
            className="input"
            type="number"
            value={config.seed}
            onChange={(e) =>
              setConfig({ ...config, seed: Number(e.target.value) })
            }
          />
        </label>
        <label>
          <span className="field-label">Session boundary · hours</span>
          <input
            className="input"
            type="number"
            min="0.1"
            step="0.5"
            value={config.sessionGapHours}
            onChange={(e) =>
              setConfig({ ...config, sessionGapHours: Number(e.target.value) })
            }
          />
        </label>
      </div>
      <div className="split-preview">
        <SmallStat
          label="Train now"
          value={overview.splits.train.toLocaleString()}
        />
        <SmallStat
          label="Validation now"
          value={overview.splits.validation.toLocaleString()}
        />
        <SmallStat
          label="Test now"
          value={overview.splits.test.toLocaleString()}
        />
      </div>
      <button className="button" onClick={apply}>
        <Scissors size={15} />
        Assign deterministic splits
      </button>
    </>
  );
}
function ExportStage({
  runId,
  overview,
  onDone,
}: {
  runId: string;
  overview: Overview;
  onDone: (m: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [output, setOutput] = useState<{
    directory: string;
    manifest: { counts: Record<string, number> };
  } | null>(null);
  async function exportData() {
    setBusy(true);
    const data = await fetch("/api/pipeline/export", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ runId }),
    }).then((r) => r.json());
    setBusy(false);
    if (data.error) return onDone(data.error);
    setOutput(data);
    onDone(
      "Export completed atomically. Private provenance and curation metadata were omitted.",
    );
  }
  const unassigned = overview.splits.unassigned;
  return (
    <>
      <h2>Export training artifacts</h2>
      <p>
        Validate the active examples and publish three nanochat-compatible JSONL
        files plus a non-private reproducibility manifest.
      </p>
      <div className="split-preview">
        <SmallStat
          label="Train"
          value={overview.splits.train.toLocaleString()}
        />
        <SmallStat
          label="Validation"
          value={overview.splits.validation.toLocaleString()}
        />
        <SmallStat label="Test" value={overview.splits.test.toLocaleString()} />
      </div>
      {unassigned > 0 && (
        <div className="notice warning">
          {unassigned.toLocaleString()} active examples are unassigned. Run the
          Split stage before exporting.
        </div>
      )}
      <div className="form-actions">
        <button
          className="button"
          disabled={busy || unassigned > 0}
          onClick={exportData}
        >
          <Archive size={15} />
          {busy ? "Validating…" : "Export run"}
        </button>
      </div>
      {output && (
        <div className="notice" style={{ marginTop: 18 }}>
          <b>Export ready</b>
          <br />
          {output.directory}
          <br />
          {Object.entries(output.manifest.counts)
            .map(([k, v]) => `${k}: ${v.toLocaleString()}`)
            .join(" · ")}
        </div>
      )}
    </>
  );
}
function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="split-card">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}
