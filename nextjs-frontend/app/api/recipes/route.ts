import crypto from "node:crypto";
import { getDb } from "@/lib/db";
import { fail, ok } from "@/lib/http";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  const runId = new URL(request.url).searchParams.get("run");
  if (!runId) return fail(new Error("A run is required"));
  return ok({
    recipes: getDb()
      .prepare(
        "SELECT id,name,position,enabled,spec_json spec FROM recipes WHERE run_id=? ORDER BY position",
      )
      .all(runId),
  });
}
export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      runId: string;
      name: string;
      spec: unknown;
      cloneId?: string;
    };
    const stamp = new Date().toISOString(),
      id = crypto.randomUUID();
    const position = Number(
      (
        getDb()
          .prepare(
            "SELECT COALESCE(MAX(position),-1)+1 next FROM recipes WHERE run_id=?",
          )
          .get(body.runId) as { next: number }
      ).next,
    );
    getDb()
      .prepare(
        "INSERT INTO recipes(id,run_id,name,position,enabled,spec_json,created_at,updated_at) VALUES (?,?,?,?,1,?,?,?)",
      )
      .run(
        id,
        body.runId,
        body.name,
        position,
        JSON.stringify(body.spec),
        stamp,
        stamp,
      );
    return ok({ id });
  } catch (error) {
    return fail(error);
  }
}
