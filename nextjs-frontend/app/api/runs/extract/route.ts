import { fail, ok } from "@/lib/http";
import { startExtraction, type ExtractionInput } from "@/lib/jobs";
export const runtime = "nodejs";
export async function POST(request: Request) {
  try {
    return ok(
      { runId: startExtraction((await request.json()) as ExtractionInput) },
      { status: 202 },
    );
  } catch (error) {
    return fail(error);
  }
}
