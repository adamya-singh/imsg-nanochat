import { fail, ok } from "@/lib/http";
import { getOverview } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  try {
    return ok(
      getOverview(new URL(request.url).searchParams.get("run") ?? undefined),
    );
  } catch (error) {
    return fail(error);
  }
}
