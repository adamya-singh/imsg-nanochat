import { RunsView } from "@/components/runs-view";
import { listRuns } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export default function RunsPage() {
  return <RunsView runs={listRuns()} />;
}
