import { fail, ok } from "@/lib/http";
import type { CodexDecision, ExampleQuery, ReviewStatus } from "@/lib/types";
import { listExamples } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  try {
    const p = new URL(request.url).searchParams;
    const runId = p.get("run");
    if (!runId) throw new Error("A run is required");
    const query: ExampleQuery = {
      runId,
      search: p.get("search") ?? undefined,
      status: (p.get("status") ?? "all") as ReviewStatus | "all",
      split: (p.get("split") ?? "all") as ExampleQuery["split"],
      tag: p.get("tag") ?? undefined,
      queue: (p.get("queue") ?? undefined) as ExampleQuery["queue"],
      codexDecision: (p.get("codexDecision") ?? "all") as
        CodexDecision | "unreviewed" | "stale" | "all",
      disagreement: p.get("disagreement") === "true",
      cursor: p.get("cursor") ?? undefined,
      limit: Number(p.get("limit") ?? 120),
    };
    return ok(listExamples(query));
  } catch (error) {
    return fail(error);
  }
}
