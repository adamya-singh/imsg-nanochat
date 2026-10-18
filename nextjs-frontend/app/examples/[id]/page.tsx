import { ExampleEditor } from "@/components/example-editor";
import { getExample, getOperationHistory } from "@/lib/workspace";
import { notFound } from "next/navigation";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export default async function ExamplePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const example = getExample(decodeURIComponent(id));
  if (!example) notFound();
  return (
    <ExampleEditor
      initialExample={example}
      operations={getOperationHistory(example.runId, 30)}
    />
  );
}
