import { fail, ok } from "@/lib/http";
import { getOverview } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  try {
    const url = new URL(request.url),
      left = url.searchParams.get("left"),
      right = url.searchParams.get("right");
    if (!left || !right) throw new Error("Select two runs");
    return ok({ left: getOverview(left), right: getOverview(right) });
  } catch (error) {
    return fail(error);
  }
}
