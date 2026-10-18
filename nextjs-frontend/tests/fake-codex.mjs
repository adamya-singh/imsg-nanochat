#!/usr/bin/env node
import fs from "node:fs";

if (process.argv.includes("--version")) {
  console.log("fake-codex 1.0");
  process.exit(0);
}
let input = "";
for await (const chunk of process.stdin) input += chunk;
const marker = "EXAMPLES_JSON:\n";
const examples = JSON.parse(
  input.slice(input.lastIndexOf(marker) + marker.length),
);
const outputIndex = process.argv.indexOf("--output-last-message");
const output = process.argv[outputIndex + 1];
const decisions = ["include", "exclude", "unsure"];
const results = examples.map((example, index) => ({
  id: example.id,
  decision: decisions[index % 3],
  confidence: 0.8,
  reasons: [
    index % 3 === 0
      ? "natural_reply"
      : index % 3 === 1
        ? "incoherent_reply"
        : "insufficient_context",
  ],
  rationale:
    index % 3 === 0
      ? "Natural reply pair."
      : index % 3 === 1
        ? "Reply is not useful for training."
        : "More context is needed.",
}));
fs.writeFileSync(output, JSON.stringify({ results }));
