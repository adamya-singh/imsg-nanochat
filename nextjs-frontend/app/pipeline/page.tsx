import { PipelineStudio } from "@/components/pipeline-studio";
import { getOverview, listRuns } from "@/lib/workspace";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export default function PipelinePage() {
  const runs = listRuns();
  const overview = getOverview();
  return <PipelineStudio runs={runs} initialOverview={overview} />;
}
