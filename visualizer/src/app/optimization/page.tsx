"use client";

import { useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { OptimizationTrajectory } from "@/components/charts/optimization-trajectory";
import type {
  OptimizationTrajectoryV1,
  BestCandidateV1,
  HoldoutSummaryV1,
  SupportProfileRefV1,
} from "@/lib/types";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";
import {
  ArrowUpIcon,
  ArrowDownIcon,
  CheckCircleIcon,
  XCircleIcon,
  AlertTriangleIcon,
  TargetIcon,
  TrendingUpIcon,
  ShieldCheckIcon,
  BarChart3Icon,
  GitCompareIcon,
  InfoIcon,
} from "lucide-react";

// Import sample data
import optimizationTrajectoryData from "@/data/optimization-trajectory.json";
import holdoutSummaryData from "@/data/holdout-summary.json";

/**
 * Format a number as percentage
 */
function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format delta with color and arrow
 */
function DeltaIndicator({ value, format = "percent" }: { value: number; format?: "percent" | "pp" }) {
  const displayValue = format === "percent" ? `${(value * 100).toFixed(1)}%` : `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)}pp`;

  if (value === 0) {
    return <span className="text-muted-foreground">{displayValue}</span>;
  }

  const isPositive = value > 0;
  const Icon = isPositive ? ArrowUpIcon : ArrowDownIcon;
  const colorClass = isPositive ? "text-green-600" : "text-red-600";

  return (
    <span className={`flex items-center gap-1 font-medium ${colorClass}`}>
      <Icon className="w-3 h-3" />
      {displayValue}
    </span>
  );
}

/**
 * Confidence indicator component
 */
function ConfidenceIndicator({ low, high, value }: { low?: number; high?: number; value: number }) {
  if (low === undefined || high === undefined) {
    return <span className="text-muted-foreground">N/A</span>;
  }

  const width = ((high - low) / value) * 100;
  const clampedWidth = Math.min(Math.max(width, 5), 50);

  return (
    <div className="flex flex-col gap-1">
      <span className="font-mono text-sm">{formatPercent(value)}</span>
      <div className="flex items-center gap-1">
        <div className="w-16 h-2 bg-muted rounded-full relative">
          <div
            className="absolute h-full bg-primary/50 rounded-full"
            style={{
              left: `${50 - clampedWidth / 2}%`,
              width: `${clampedWidth}%`,
            }}
          />
        </div>
        <span className="text-xs text-muted-foreground font-mono">
          [{formatPercent(low)}, {formatPercent(high)}]
        </span>
      </div>
    </div>
  );
}

/**
 * Best Candidate Card Component
 */
function BestCandidateCard({
  candidate,
  seedProfile,
}: {
  candidate: BestCandidateV1;
  seedProfile: SupportProfileRefV1;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TargetIcon className="w-5 h-5" />
          Best Candidate
        </CardTitle>
        <CardDescription>
          Found at iteration {candidate.found_at_iteration}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-muted/30 rounded-lg">
            <div className="text-xs text-muted-foreground">Candidate ID</div>
            <code className="text-sm font-mono truncate block">
              {candidate.candidate_id}
            </code>
          </div>
          <div className="p-3 bg-muted/30 rounded-lg">
            <div className="text-xs text-muted-foreground">Final Score</div>
            <div className="text-lg font-semibold font-mono">
              {formatPercent(candidate.score)}
            </div>
          </div>
        </div>

        <Separator />

        <div className="space-y-2">
          <h4 className="text-sm font-medium">Seed vs Best Comparison</h4>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2 bg-muted/20 rounded">
              <div className="text-xs text-muted-foreground">Seed Profile</div>
              <Badge variant="outline" className="mt-1 font-mono text-xs">
                {seedProfile.profile_id}
              </Badge>
            </div>
            <div className="flex items-center justify-center">
              <GitCompareIcon className="w-4 h-4 text-muted-foreground" />
            </div>
            <div className="p-2 bg-green-500/10 rounded border border-green-500/20">
              <div className="text-xs text-muted-foreground">Best Profile</div>
              <Badge
                variant="outline"
                className="mt-1 font-mono text-xs border-green-500/50"
              >
                {candidate.profile.profile_id}
              </Badge>
            </div>
          </div>
        </div>

        <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Improvement vs Seed</span>
            <DeltaIndicator value={candidate.delta_vs_seed} />
          </div>
        </div>

        {candidate.content && (
          <>
            <Separator />
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Candidate Configuration</h4>
              <pre className="p-3 bg-muted/30 rounded-lg text-xs font-mono overflow-auto max-h-[200px]">
                {JSON.stringify(candidate.content, null, 2)}
              </pre>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Holdout Summary Card Component
 */
function HoldoutSummaryCard({ summary }: { summary: HoldoutSummaryV1 }) {
  const isOverfitting = !summary.generalization_ok;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldCheckIcon className="w-5 h-5" />
          Holdout Validation
        </CardTitle>
        <CardDescription>
          Validation on {summary.holdout_size} held-out samples
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Generalization Status */}
        <div
          className={`p-4 rounded-lg border ${
            isOverfitting
              ? "bg-yellow-500/10 border-yellow-500/30"
              : "bg-green-500/10 border-green-500/30"
          }`}
        >
          <div className="flex items-center gap-2">
            {isOverfitting ? (
              <>
                <AlertTriangleIcon className="w-5 h-5 text-yellow-600" />
                <div>
                  <div className="font-medium text-yellow-700 dark:text-yellow-400">
                    Potential Overfitting Detected
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Generalization gap: {formatPercent(summary.generalization_gap)}
                  </div>
                </div>
              </>
            ) : (
              <>
                <CheckCircleIcon className="w-5 h-5 text-green-600" />
                <div>
                  <div className="font-medium text-green-700 dark:text-green-400">
                    Generalization OK
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Gap within acceptable threshold
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Train vs Holdout Comparison */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-muted/30 rounded-lg text-center">
            <div className="text-xs text-muted-foreground mb-1">Train Pass Rate</div>
            <div className="text-xl font-semibold font-mono">
              {formatPercent(summary.train_pass_rate)}
            </div>
          </div>
          <div className="p-3 bg-muted/30 rounded-lg text-center">
            <div className="text-xs text-muted-foreground mb-1">Holdout Pass Rate</div>
            <div className="text-xl font-semibold font-mono">
              {formatPercent(summary.holdout_pass_rate)}
            </div>
          </div>
        </div>

        <Separator />

        {/* Detailed Metrics */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <BarChart3Icon className="w-4 h-4" />
            Detailed Metrics
          </h4>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead className="text-right">Train</TableHead>
                <TableHead className="text-right">Holdout</TableHead>
                <TableHead className="text-right">95% CI</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {summary.metrics.map((metric, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium">{metric.metric_name}</TableCell>
                  <TableCell className="text-right font-mono">
                    {formatPercent(metric.train_value)}
                  </TableCell>
                  <TableCell className="text-right">
                    <ConfidenceIndicator
                      low={metric.ci_low}
                      high={metric.ci_high}
                      value={metric.holdout_value}
                    />
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground text-xs">
                    {metric.std_error !== undefined
                      ? `SE: ${metric.std_error.toFixed(4)}`
                      : "N/A"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <div className="text-xs text-muted-foreground flex items-center gap-1">
          <InfoIcon className="w-3 h-3" />
          Validated at: {new Date(summary.validated_at).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * No Data Fallback Component
 */
function NoDataState() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 container mx-auto py-8 px-4">
        <div className="max-w-2xl mx-auto p-6 bg-muted/30 border rounded-lg text-center">
          <TrendingUpIcon className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">No Optimization Data Available</h2>
          <p className="text-muted-foreground mb-4">
            Run optimization to generate trajectory and holdout validation data.
          </p>
          <code className="block p-3 bg-muted rounded text-sm font-mono">
            uv run python scripts/run_optimization.py --model gpt-4o
          </code>
        </div>
      </main>
      <Footer />
    </div>
  );
}

export default function OptimizationPage() {
  const [selectedRun, setSelectedRun] = useState<string | null>(null);

  // Cast imported data to types
  const trajectory = optimizationTrajectoryData as unknown as OptimizationTrajectoryV1;
  const holdoutSummary = holdoutSummaryData as unknown as HoldoutSummaryV1;

  // Check if we have valid data
  const hasData = trajectory && trajectory.trajectory && trajectory.trajectory.length > 0;

  // Extract best candidate from trajectory (simulated for demo)
  const bestCandidate: BestCandidateV1 = useMemo(() => {
    if (!trajectory?.trajectory) {
      return {
        candidate_id: "unknown",
        profile: trajectory?.seed_profile || { profile_id: "unknown", mode: "none" },
        score: 0,
        delta_vs_seed: 0,
        found_at_iteration: 0,
      };
    }

    const bestPoint =
      trajectory.trajectory.find((p) => p.candidate_id === trajectory.best_candidate_id) ||
      trajectory.trajectory[trajectory.trajectory.length - 1];

    const seedScore = trajectory.trajectory[0].score;

    return {
      candidate_id: bestPoint.candidate_id,
      profile: {
        profile_id: `optimized_${bestPoint.iteration}`,
        mode: "system_plus_agents",
        name: `Optimized Profile (Iter ${bestPoint.iteration})`,
      },
      score: bestPoint.score,
      delta_vs_seed: bestPoint.score - seedScore,
      found_at_iteration: bestPoint.iteration,
      content: {
        mode: "system_plus_agents",
        parameters: {
          temperature: 0.7,
          max_tokens: 4096,
          system_prompt_version: "v2.3",
        },
      },
    };
  }, [trajectory]);

  if (!hasData) {
    return <NoDataState />;
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 container mx-auto py-8 px-4 space-y-8">
        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <TrendingUpIcon className="w-6 h-6" />
              Optimization Insights
            </h2>
            <p className="text-muted-foreground">
              Profile optimization trajectory and holdout validation results
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-normal">
              Run: {trajectory.run_id}
            </Badge>
            <Badge variant="secondary" className="font-normal">
              {trajectory.model_name}
            </Badge>
          </div>
        </div>

        {/* Trajectory Chart */}
        <OptimizationTrajectory
          trajectory={trajectory}
          title="Score Over Iterations"
          description="Candidate score progression during optimization"
        />

        {/* Best Candidate and Holdout Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <BestCandidateCard
            candidate={bestCandidate}
            seedProfile={trajectory.seed_profile}
          />
          {holdoutSummary && <HoldoutSummaryCard summary={holdoutSummary} />}
        </div>

        {/* Additional Info */}
        <Card className="bg-muted/30">
          <CardContent className="pt-4">
            <div className="flex items-start gap-3">
              <InfoIcon className="w-4 h-4 text-muted-foreground mt-0.5" />
              <div className="text-sm text-muted-foreground">
                <p className="font-medium text-foreground mb-1">
                  Optimization Summary
                </p>
                <p>
                  Started: {new Date(trajectory.started_at).toLocaleString()}
                  {trajectory.completed_at && (
                    <> | Completed: {new Date(trajectory.completed_at).toLocaleString()}</>
                  )}
                </p>
                <p className="mt-1">
                  Total iterations: {trajectory.total_iterations} | Best found at: iteration{" "}
                  {bestCandidate.found_at_iteration}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>

      <Footer />
    </div>
  );
}
