import { ok } from "@/lib/http";
import { listRuns } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET() {
  return ok({ runs: listRuns() });
}
