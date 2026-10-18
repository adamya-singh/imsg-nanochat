import { getDb } from "@/lib/db";
import { fail, ok } from "@/lib/http";
export const runtime = "nodejs";
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const body = (await request.json()) as {
      name?: string;
      position?: number;
      enabled?: boolean;
      spec?: unknown;
    };
    const current = getDb()
      .prepare("SELECT * FROM recipes WHERE id=?")
      .get(id) as Record<string, unknown> | undefined;
    if (!current) throw new Error("Recipe step not found");
    getDb()
      .prepare(
        "UPDATE recipes SET name=?,position=?,enabled=?,spec_json=?,updated_at=? WHERE id=?",
      )
      .run(
        body.name ?? String(current.name),
        body.position ?? Number(current.position),
        body.enabled == null ? Number(current.enabled) : body.enabled ? 1 : 0,
        body.spec == null
          ? String(current.spec_json)
          : JSON.stringify(body.spec),
        new Date().toISOString(),
        id,
      );
    return ok({ updated: true });
  } catch (error) {
    return fail(error);
  }
}
export async function DELETE(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  getDb().prepare("DELETE FROM recipes WHERE id=?").run(id);
  return ok({ deleted: true });
}
