import { fail, ok } from "@/lib/http";
import { cancelExtraction } from "@/lib/jobs";
export const runtime = "nodejs";
export async function POST(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  return cancelExtraction(id)
    ? ok({ cancelled: true })
    : fail(new Error("No active extraction job found"), 404);
}
