import { controlCodexReviewJob, getCodexReviewJob } from "@/lib/codex-review";
import { fail, ok } from "@/lib/http";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const job = getCodexReviewJob(id);
  return job ? ok(job) : fail(new Error("Review job not found"), 404);
}
export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const { action } = (await request.json()) as {
      action: "pause" | "resume" | "cancel" | "retry";
    };
    return ok(controlCodexReviewJob(id, action));
  } catch (error) {
    return fail(error);
  }
}
