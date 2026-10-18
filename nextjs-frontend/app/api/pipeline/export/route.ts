import { fail, ok } from "@/lib/http";
import { exportRun } from "@/lib/pipeline";
export const runtime = "nodejs";
export async function POST(request: Request) {
  try {
    const { runId } = (await request.json()) as { runId: string };
    return ok(await exportRun(runId));
  } catch (error) {
    return fail(error);
  }
}
