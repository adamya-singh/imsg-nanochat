import { createCodexReviewJob, listCodexReviewJobs } from "@/lib/codex-review";
import { fail, ok } from "@/lib/http";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  try {
    const runId = new URL(request.url).searchParams.get("run");
    if (!runId) throw new Error("A run is required");
    return ok({ jobs: listCodexReviewJobs(runId) });
  } catch (error) {
    return fail(error);
  }
}
export async function POST(request: Request) {
  try {
    return ok(createCodexReviewJob(await request.json()), { status: 202 });
  } catch (error) {
    return fail(error);
  }
}
