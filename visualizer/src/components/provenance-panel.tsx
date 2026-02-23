"use client";

import { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import type { ProvenanceV1, MetadataV1 } from "@/lib/types";
import {
  FingerprintIcon,
  DatabaseIcon,
  BoxIcon,
  FileJsonIcon,
  GitBranchIcon,
  ExternalLinkIcon,
  InfoIcon,
} from "lucide-react";

interface ProvenancePanelProps {
  metadata: MetadataV1;
  trigger?: React.ReactNode;
}

/**
 * Helper component to display a provenance field with graceful handling of missing values
 */
function ProvenanceField({
  label,
  value,
  icon: Icon,
  mono = false,
  copyable = false,
}: {
  label: string;
  value: string | undefined | null;
  icon?: React.ComponentType<{ className?: string }>;
  mono?: boolean;
  copyable?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (value) {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const isAvailable = value !== undefined && value !== null && value !== "";

  return (
    <div className="flex items-start gap-3 py-2">
      {Icon && (
        <Icon className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
      )}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-muted-foreground">{label}</div>
        {isAvailable ? (
          <div className="flex items-center gap-2">
            <span
              className={`text-sm break-all ${mono ? "font-mono text-xs" : ""}`}
            >
              {value}
            </span>
            {copyable && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={handleCopy}
              >
                {copied ? "Copied" : "Copy"}
              </Button>
            )}
          </div>
        ) : (
          <span className="text-sm text-muted-foreground italic">
            not available
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Helper component to display a download link
 */
function ArtifactLink({
  label,
  path,
  icon: Icon,
}: {
  label: string;
  path: string | undefined;
  icon?: React.ComponentType<{ className?: string }>;
}) {
  if (!path) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground italic">
        {Icon && <Icon className="w-4 h-4" />}
        <span>{label}: not available</span>
      </div>
    );
  }

  return (
    <a
      href={`/${path}`}
      target="_blank"
      rel="noreferrer"
      className="flex items-center gap-2 text-sm text-primary hover:underline"
    >
      {Icon && <Icon className="w-4 h-4" />}
      <span>{label}</span>
      <ExternalLinkIcon className="w-3 h-3" />
    </a>
  );
}

/**
 * ProvenancePanel - Displays provenance information for the benchmark run
 *
 * Shows:
 * - Run manifest fingerprint
 * - Schema version
 * - Dataset fingerprint
 * - Environment metadata (image digest, lock hashes, versions)
 * - Downloadable links to artifacts
 */
export function ProvenancePanel({ metadata, trigger }: ProvenancePanelProps) {
  const [open, setOpen] = useState(false);
  const provenance: ProvenanceV1 | undefined = metadata.provenance;

  // Calculate a summary for the footer
  const hasProvenance = provenance !== undefined;
  const hasFingerprint = provenance?.manifest_fingerprint !== undefined;
  const hasArtifacts = provenance?.artifact_paths !== undefined;

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        {trigger || (
          <Button variant="outline" size="sm" className="gap-2">
            <FingerprintIcon className="w-4 h-4" />
            Provenance
          </Button>
        )}
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[540px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <FingerprintIcon className="w-5 h-5" />
            Provenance & Reproducibility
          </SheetTitle>
          <SheetDescription>
            Trace this dashboard back to exact run artifacts
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Core Provenance */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Run Identification</CardTitle>
              <CardDescription>
                Unique identifiers for this benchmark run
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-1">
              <ProvenanceField
                label="Manifest Fingerprint"
                value={provenance?.manifest_fingerprint}
                icon={FingerprintIcon}
                mono
                copyable
              />
              <ProvenanceField
                label="Schema Version"
                value={provenance?.schema_version?.toString()}
                icon={DatabaseIcon}
              />
              <ProvenanceField
                label="Dataset Fingerprint"
                value={provenance?.dataset_fingerprint}
                icon={DatabaseIcon}
                mono
                copyable
              />
            </CardContent>
          </Card>

          {/* Git Information */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <GitBranchIcon className="w-4 h-4" />
                Git Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              <ProvenanceField
                label="Commit"
                value={provenance?.git_commit}
                icon={GitBranchIcon}
                mono
                copyable
              />
              <ProvenanceField
                label="Branch"
                value={provenance?.git_branch}
                icon={GitBranchIcon}
              />
            </CardContent>
          </Card>

          {/* Environment */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <BoxIcon className="w-4 h-4" />
                Environment
              </CardTitle>
              <CardDescription>
                Runtime environment for reproducibility
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-1">
              <ProvenanceField
                label="Image Digest"
                value={provenance?.environment?.image_digest}
                icon={BoxIcon}
                mono
                copyable
              />
              <ProvenanceField
                label="Python Version"
                value={provenance?.environment?.python_version}
                icon={BoxIcon}
              />

              {/* Lock Hashes */}
              {provenance?.environment?.lock_hashes &&
                Object.keys(provenance.environment.lock_hashes).length > 0 && (
                  <div className="pt-2">
                    <div className="text-sm font-medium text-muted-foreground mb-2">
                      Lock File Hashes
                    </div>
                    <div className="space-y-1">
                      {Object.entries(provenance.environment.lock_hashes).map(
                        ([file, hash]) => (
                          <div
                            key={file}
                            className="flex items-center gap-2 text-xs"
                          >
                            <Badge variant="outline" className="font-normal">
                              {file}
                            </Badge>
                            <code className="font-mono text-muted-foreground truncate flex-1">
                              {hash}
                            </code>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

              {/* Package Versions */}
              {provenance?.environment?.package_versions &&
                Object.keys(provenance.environment.package_versions).length >
                  0 && (
                  <div className="pt-2">
                    <div className="text-sm font-medium text-muted-foreground mb-2">
                      Package Versions
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(provenance.environment.package_versions).map(
                        ([pkg, version]) => (
                          <Badge
                            key={pkg}
                            variant="secondary"
                            className="font-mono text-xs"
                          >
                            {pkg}=={version}
                          </Badge>
                        )
                      )}
                    </div>
                  </div>
                )}

              {/* Show "not available" if no environment data */}
              {(!provenance?.environment ||
                (!provenance.environment.image_digest &&
                  !provenance.environment.python_version &&
                  (!provenance.environment.lock_hashes ||
                    Object.keys(provenance.environment.lock_hashes).length ===
                      0) &&
                  (!provenance.environment.package_versions ||
                    Object.keys(provenance.environment.package_versions)
                      .length === 0))) && (
                <div className="text-sm text-muted-foreground italic">
                  No environment metadata available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Artifacts */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <FileJsonIcon className="w-4 h-4" />
                Source Artifacts
              </CardTitle>
              <CardDescription>
                Download the original run artifacts
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <ArtifactLink
                label="manifest.json"
                path={provenance?.artifact_paths?.manifest}
                icon={FileJsonIcon}
              />
              <ArtifactLink
                label="summary.json"
                path={provenance?.artifact_paths?.summary}
                icon={FileJsonIcon}
              />
              <ArtifactLink
                label="results.jsonl"
                path={provenance?.artifact_paths?.results}
                icon={FileJsonIcon}
              />

              {!hasArtifacts && (
                <div className="text-sm text-muted-foreground italic">
                  No artifact paths available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Run Summary */}
          <Card className="bg-muted/30">
            <CardContent className="pt-4">
              <div className="flex items-start gap-3">
                <InfoIcon className="w-4 h-4 text-muted-foreground mt-0.5" />
                <div className="text-sm text-muted-foreground">
                  <p className="font-medium text-foreground mb-1">
                    Run Summary
                  </p>
                  <p>
                    {metadata.runs_included} run(s) included across{" "}
                    {metadata.total_tasks} tasks
                  </p>
                  <p className="mt-1">
                    Last updated:{" "}
                    {new Date(metadata.last_updated).toLocaleString()}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </SheetContent>
    </Sheet>
  );
}
