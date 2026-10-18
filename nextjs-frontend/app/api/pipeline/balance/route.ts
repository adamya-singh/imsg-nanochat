import { fail, ok } from "@/lib/http";
import { applyBalance, previewBalance } from "@/lib/pipeline";
import type { BalanceConfig } from "@/lib/types";
export const runtime = "nodejs";
type Body = { runId: string; config: BalanceConfig; apply?: boolean };
export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Body;
    return ok(
      body.apply
        ? applyBalance(body.runId, body.config)
        : previewBalance(body.runId, body.config),
    );
  } catch (error) {
    return fail(error);
  }
}
