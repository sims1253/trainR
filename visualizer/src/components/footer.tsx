"use client";

import { BenchmarkData } from "@/lib/types";
import { ProvenancePanel } from "@/components/provenance-panel";
import { FingerprintIcon, ExternalLinkIcon } from "lucide-react";

interface FooterProps {
  data?: BenchmarkData;
}

export function Footer({ data }: FooterProps) {
  const lastUpdated = data
    ? new Date(data.metadata.last_updated).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : null;

  const totalTasks = data?.metadata.total_tasks;
  const modelCount = data?.models.length;

  // Extract provenance summary
  const provenance = data?.metadata.provenance;
  const hasFingerprint = provenance?.manifest_fingerprint !== undefined;
  const fingerprintShort = hasFingerprint
    ? provenance!.manifest_fingerprint!.substring(0, 8)
    : null;

  return (
    <footer className="border-t bg-muted/30 mt-auto">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          {/* Left side - Stats */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
            {lastUpdated && (
              <>
                <span>Last updated: {lastUpdated}</span>
                <span className="hidden md:inline text-muted-foreground/50">
                  |
                </span>
              </>
            )}
            {totalTasks !== undefined && (
              <>
                <span>{totalTasks} tasks</span>
                <span className="hidden md:inline text-muted-foreground/50">
                  |
                </span>
              </>
            )}
            {modelCount !== undefined && <span>{modelCount} models</span>}
          </div>

          {/* Right side - Provenance and credits */}
          <div className="flex flex-wrap items-center gap-4">
            {/* Provenance link */}
            {data && (
              <div className="flex items-center gap-2">
                {fingerprintShort ? (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <FingerprintIcon className="w-3 h-3" />
                    <code className="font-mono">{fingerprintShort}...</code>
                  </div>
                ) : null}
                <ProvenancePanel
                  metadata={data.metadata}
                  trigger={
                    <button className="text-xs text-primary hover:underline flex items-center gap-1">
                      {fingerprintShort ? "Full provenance" : "View provenance"}
                      <ExternalLinkIcon className="w-3 h-3" />
                    </button>
                  }
                />
              </div>
            )}

            {/* Divider */}
            <span className="hidden md:inline text-muted-foreground/50">
              |
            </span>

            {/* Credits */}
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>
                Inspired by{" "}
                <a
                  href="https://github.com/OpenCode-AI/skatebench"
                  target="_blank"
                  rel="noreferrer"
                  className="underline hover:text-foreground"
                >
                  skatebench
                </a>
              </span>
              <span className="text-muted-foreground/50">|</span>
              <span>Built with Next.js & shadcn/ui</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
