import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { DatabaseSync } from "node:sqlite";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const port = 3199;
const base = `http://127.0.0.1:${port}`;
let server;
let reviewJobId;

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`${base}/api/runs`);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("The test server did not become ready");
}

test.before(async () => {
  const fakeCodex = path.join(process.cwd(), "tests", "fake-codex.mjs");
  fs.chmodSync(fakeCodex, 0o755);
  server = spawn(
    process.execPath,
    [
      "node_modules/next/dist/bin/next",
      "start",
      "--hostname",
      "127.0.0.1",
      "--port",
      String(port),
    ],
    {
      cwd: process.cwd(),
      stdio: "ignore",
      env: {
        ...process.env,
        PATH: `${path.dirname(process.execPath)}:${process.env.PATH}`,
        CODEX_BIN: fakeCodex,
      },
    },
  );
  await waitForServer();
});

test.after(() => {
  server?.kill("SIGTERM");
  if (reviewJobId) {
    const db = new DatabaseSync(
      path.join(
        process.cwd(),
        "..",
        "training-data",
        "review",
        "curation.sqlite",
      ),
    );
    db.exec("PRAGMA foreign_keys=ON");
    db.prepare("DELETE FROM codex_reviews WHERE job_id=?").run(reviewJobId);
    db.prepare("DELETE FROM codex_review_jobs WHERE id=?").run(reviewJobId);
    db.close();
  }
});

test("baseline run and cursor-paginated examples are available", async () => {
  const runs = await fetch(`${base}/api/runs`).then((response) =>
    response.json(),
  );
  const baseline = runs.runs.find((run) => run.id === "baseline");
  assert.equal(baseline.status, "completed");
  assert.equal(baseline.exampleCount, 12637);

  const page = await fetch(
    `${base}/api/examples?run=baseline&limit=3&status=all`,
  ).then((response) => response.json());
  assert.equal(page.examples.length, 3);
  assert.equal(page.total, 12637);
  assert.ok(page.nextCursor);
});

test("transformation previews reject invalid regular expressions", async () => {
  const response = await fetch(`${base}/api/transformations/preview`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      kind: "replace",
      query: { runId: "baseline", status: "all" },
      field: "both",
      find: "[",
      replace: "",
      regex: true,
    }),
  });
  assert.equal(response.status, 400);
  const body = await response.json();
  assert.match(body.error, /regular expression/i);
});

test("overview exposes aggregate metrics without message content", async () => {
  const overview = await fetch(`${base}/api/overview?run=baseline`).then(
    (response) => response.json(),
  );
  assert.equal(overview.counts.total, 12637);
  assert.equal(JSON.stringify(overview).includes("current_user"), false);
  assert.equal(JSON.stringify(overview).includes("contact_label"), false);
});

test("mock Codex worker completes a structured, resumable pilot", async () => {
  const created = await fetch(`${base}/api/codex-review/jobs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      runId: "baseline",
      scope: "pilot",
      model: "fake-mini",
    }),
  }).then((r) => r.json());
  assert.ok(created.id);
  reviewJobId = created.id;
  let job = created;
  for (
    let attempt = 0;
    attempt < 100 && !["completed", "failed"].includes(job.status);
    attempt += 1
  ) {
    await new Promise((resolve) => setTimeout(resolve, 100));
    job = await fetch(`${base}/api/codex-review/jobs/${reviewJobId}`).then(
      (r) => r.json(),
    );
  }
  assert.equal(job.status, "completed");
  assert.equal(job.processedCount, 100);
  assert.equal(job.model, "fake-mini");
  assert.equal(job.includeCount + job.excludeCount + job.unsureCount, 100);
});

test("human decisions override current and stale Codex exclusions", async () => {
  const page = await fetch(
    `${base}/api/examples?run=baseline&limit=1&status=unreviewed&codexDecision=exclude`,
  ).then((response) => response.json());
  assert.equal(page.examples.length, 1);
  const example = page.examples[0];
  assert.equal(example.effectiveDecision, "exclude");

  const db = new DatabaseSync(
    path.join(
      process.cwd(),
      "..",
      "training-data",
      "review",
      "curation.sqlite",
    ),
  );
  db.prepare("UPDATE examples SET status='approved' WHERE id=?").run(
    example.id,
  );
  let updated = await fetch(
    `${base}/api/examples/${encodeURIComponent(example.id)}`,
  ).then((response) => response.json());
  assert.equal(updated.effectiveDecision, "include");

  db.prepare(
    "UPDATE examples SET status='unreviewed',content_hash=content_hash||'-edited' WHERE id=?",
  ).run(example.id);
  updated = await fetch(
    `${base}/api/examples/${encodeURIComponent(example.id)}`,
  ).then((response) => response.json());
  assert.equal(updated.codexReview.stale, true);
  assert.equal(updated.effectiveDecision, "include");
  db.prepare("UPDATE examples SET content_hash=? WHERE id=?").run(
    example.codexReview.reviewedContentHash,
    example.id,
  );
  db.close();

  const overview = await fetch(`${base}/api/overview?run=baseline`).then(
    (response) => response.json(),
  );
  const balance = await fetch(`${base}/api/pipeline/balance`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      runId: "baseline",
      config: { maxCount: null, maxShare: null, seed: 42 },
      apply: false,
    }),
  }).then((response) => response.json());
  assert.equal(balance.total, overview.counts.active);
});
