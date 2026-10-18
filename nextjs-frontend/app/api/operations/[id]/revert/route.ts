import { fail, ok } from "@/lib/http";
import { revertOperation } from "@/lib/pipeline";
export const runtime = "nodejs";
export async function POST(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    return ok({ reverted: revertOperation(id) });
  } catch (error) {
    return fail(error);
  }
}
