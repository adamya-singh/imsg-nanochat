import { fail, ok } from "@/lib/http";
import { previewTransformation } from "@/lib/pipeline";
import type { TransformationSpec } from "@/lib/types";
export const runtime = "nodejs";
export async function POST(request: Request) {
  try {
    return ok(
      previewTransformation((await request.json()) as TransformationSpec),
    );
  } catch (error) {
    return fail(error);
  }
}
