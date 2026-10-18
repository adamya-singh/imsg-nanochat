import type { Metadata } from "next";
import Link from "next/link";
import { Bot, Database, GitCompareArrows, Workflow } from "lucide-react";
import "./globals.css";
import "./accessibility.css";

export const metadata: Metadata = {
  title: "Dataset Studio · nanochat",
  description: "A private local workbench for shaping nanochat training data.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="app-header">
          <Link href="/" className="brand" aria-label="Dataset Studio home">
            <span className="brand-mark">
              <Database size={18} />
            </span>
            <span>
              Dataset <b>Studio</b>
            </span>
          </Link>
          <nav aria-label="Main navigation">
            <Link href="/">
              <Database size={16} /> Overview
            </Link>
            <Link href="/pipeline">
              <Workflow size={16} /> Pipeline
            </Link>
            <Link href="/ai-review">
              <Bot size={16} /> AI Review
            </Link>
            <Link href="/runs">
              <GitCompareArrows size={16} /> Runs
            </Link>
          </nav>
          <div className="local-chip">
            <i /> Local only
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
