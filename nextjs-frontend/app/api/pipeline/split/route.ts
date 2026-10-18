import { fail, ok } from "@/lib/http";
import { applySplit } from "@/lib/pipeline";
import type { SplitConfig } from "@/lib/types";
export const runtime = "nodejs";
export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      runId: string;
      config: SplitConfig;
    };
    return ok({ counts: applySplit(body.runId, body.config) });
  } catch (error) {
    return fail(error);
  }
}
