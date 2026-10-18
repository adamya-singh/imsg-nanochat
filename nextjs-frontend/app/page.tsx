import { Dashboard } from "@/components/dashboard";
import { getOverview, listExamples, listRuns } from "@/lib/workspace";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ run?: string }>;
}) {
  const { run } = await searchParams;
  const overview = getOverview(run);
  const examples = listExamples({
    runId: overview.run.id,
    status: "all",
    limit: 120,
  });
  const runs = listRuns();
  return (
    <Dashboard
      initialOverview={overview}
      initialExamples={examples}
      runs={runs}
    />
  );
}
