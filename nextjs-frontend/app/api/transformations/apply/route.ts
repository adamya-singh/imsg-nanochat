import { fail, ok } from "@/lib/http";
import { applyTransformation } from "@/lib/pipeline";
import type { TransformationSpec } from "@/lib/types";
export const runtime = "nodejs";
export async function POST(request: Request) {
  try {
    return ok(
      applyTransformation((await request.json()) as TransformationSpec),
    );
  } catch (error) {
    return fail(error);
  }
}
