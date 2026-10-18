import { AiReview } from "@/components/ai-review";
import { listCodexReviewJobs } from "@/lib/codex-review";
import { getOverview, listRuns } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export default function AiReviewPage() {
  const runs = listRuns().filter((r) => r.status === "completed");
  const overview = getOverview();
  return (
    <AiReview
      runs={runs}
      initialRunId={overview.run.id}
      initialOverview={overview}
      initialJobs={listCodexReviewJobs(overview.run.id)}
    />
  );
}
