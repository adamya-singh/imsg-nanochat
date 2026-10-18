import { fail, ok } from "@/lib/http";
import { getExample, updateExample } from "@/lib/workspace";
import type { TrainingExample } from "@/lib/types";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const example = getExample(decodeURIComponent(id));
  return example ? ok(example) : fail(new Error("Example not found"), 404);
}
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const body = (await request.json()) as {
      version: number;
      patch: Partial<
        Pick<
          TrainingExample,
          "user" | "assistant" | "status" | "tags" | "notes"
        >
      >;
    };
    return ok(updateExample(decodeURIComponent(id), body.patch, body.version));
  } catch (error) {
    return fail(error, 409);
  }
}
