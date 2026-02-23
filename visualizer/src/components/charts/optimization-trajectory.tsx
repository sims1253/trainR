"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Dot,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { OptimizationTrajectoryV1 } from "@/lib/types";
import { TrendingUpIcon, TargetIcon, MinusIcon } from "lucide-react";

interface OptimizationTrajectoryProps {
  trajectory: OptimizationTrajectoryV1;
  title?: string;
  description?: string;
}

/**
 * Custom dot component to highlight the best candidate
 */
function CustomDot(props: {
  cx?: number;
  cy?: number;
  payload?: { candidate_id: string; isBest: boolean; score: number };
}) {
  const { cx, cy, payload } = props;
  if (!cx || !cy || !payload) return null;

  if (payload.isBest) {
    return (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={8}
          fill="hsl(var(--primary))"
          stroke="hsl(var(--background))"
          strokeWidth={2}
        />
        <circle cx={cx} cy={cy} r={3} fill="hsl(var(--background))" />
      </g>
    );
  }

  return (
    <Dot
      cx={cx}
      cy={cy}
      r={4}
      fill="hsl(var(--muted-foreground))"
      stroke="hsl(var(--background))"
      strokeWidth={1}
    />
  );
}

/**
 * Calculate improvement trend (linear regression slope approximation)
 */
function calculateTrend(data: { score: number }[]): number {
  if (data.length < 2) return 0;
  const n = data.length;
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += data[i].score;
    sumXY += i * data[i].score;
    sumX2 += i * i;
  }
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  return slope;
}

/**
 * Optimization Trajectory Chart Component
 *
 * Displays the optimization progress over iterations, showing score improvements
 * and highlighting the best candidate found.
 */
export function OptimizationTrajectory({
  trajectory,
  title = "Optimization Trajectory",
  description,
}: OptimizationTrajectoryProps) {
  if (!trajectory || !trajectory.trajectory || trajectory.trajectory.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>No trajectory data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[200px] text-muted-foreground">
            Run optimization to generate trajectory data
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare chart data
  const chartData = trajectory.trajectory.map((point) => ({
    iteration: point.iteration,
    score: point.score * 100, // Convert to percentage
    candidate_id: point.candidate_id,
    isBest: point.candidate_id === trajectory.best_candidate_id,
    timestamp: point.timestamp,
  }));

  // Calculate statistics
  const scores = trajectory.trajectory.map((p) => p.score);
  const initialScore = scores[0];
  const finalScore = scores[scores.length - 1];
  const bestScore = Math.max(...scores);
  const improvement = ((bestScore - initialScore) / initialScore) * 100;
  const trend = calculateTrend(trajectory.trajectory);

  // Find best iteration
  const bestIteration =
    trajectory.trajectory.find((p) => p.candidate_id === trajectory.best_candidate_id)
      ?.iteration ?? trajectory.trajectory.length - 1;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUpIcon className="w-5 h-5" />
          {title}
        </CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
        <div className="flex flex-wrap gap-2 mt-2">
          <Badge variant="outline" className="font-normal">
            {trajectory.model_name}
          </Badge>
          <Badge variant="secondary" className="font-normal">
            {trajectory.total_iterations} iterations
          </Badge>
          <Badge variant="outline" className="font-normal">
            Seed: {trajectory.seed_profile.name || trajectory.seed_profile.profile_id}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-muted/30 rounded-lg">
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Initial Score</div>
            <div className="text-lg font-semibold font-mono">
              {(initialScore * 100).toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Best Score</div>
            <div className="text-lg font-semibold font-mono text-green-600">
              {(bestScore * 100).toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Improvement</div>
            <div
              className={`text-lg font-semibold flex items-center justify-center gap-1 ${
                improvement > 0 ? "text-green-600" : improvement < 0 ? "text-red-600" : "text-muted-foreground"
              }`}
            >
              {improvement > 0 ? (
                <TrendingUpIcon className="w-4 h-4" />
              ) : improvement < 0 ? (
                <TrendingUpIcon className="w-4 h-4 rotate-180" />
              ) : (
                <MinusIcon className="w-4 h-4" />
              )}
              {improvement > 0 ? "+" : ""}
              {improvement.toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Best at Iter</div>
            <div className="text-lg font-semibold font-mono">{bestIteration}</div>
          </div>
        </div>

        {/* Line Chart */}
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="iteration"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                label={{
                  value: "Iteration",
                  position: "insideBottom",
                  offset: -5,
                  fontSize: 11,
                  fill: "hsl(var(--muted-foreground))",
                }}
              />
              <YAxis
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
                domain={[
                  (dataMin: number) => Math.max(0, dataMin - 5),
                  (dataMax: number) => Math.min(100, dataMax + 5),
                ]}
                label={{
                  value: "Score",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 11,
                  fill: "hsl(var(--muted-foreground))",
                }}
              />
              <Tooltip
                cursor={{ fill: "hsl(var(--muted) / 0.3)" }}
                contentStyle={{
                  borderRadius: "8px",
                  border: "none",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                  backgroundColor: "hsl(var(--background))",
                }}
                formatter={(value, _name, props) => {
                  const data = props.payload as { isBest: boolean; candidate_id: string };
                  return [
                    <>
                      <span className="font-mono">{Number(value).toFixed(1)}%</span>
                      {data.isBest && (
                        <Badge variant="default" className="ml-2 text-xs">
                          Best
                        </Badge>
                      )}
                    </>,
                    "Score",
                  ];
                }}
                labelFormatter={(label, payload) => {
                  if (payload && payload[0]?.payload) {
                    const data = payload[0].payload as { iteration: number; candidate_id: string };
                    return `Iteration ${data.iteration} (${data.candidate_id.slice(0, 8)}...)`;
                  }
                  return `Iteration ${label}`;
                }}
              />
              <ReferenceLine
                y={initialScore * 100}
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="5 5"
                label={{
                  value: "Seed",
                  position: "right",
                  fontSize: 10,
                  fill: "hsl(var(--muted-foreground))",
                }}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={<CustomDot />}
                activeDot={{ r: 6, fill: "hsl(var(--primary))" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Trend Indicator */}
        <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
          <TargetIcon className="w-4 h-4" />
          <span>
            Trend:{" "}
            <span
              className={`font-medium ${
                trend > 0 ? "text-green-600" : trend < 0 ? "text-red-600" : ""
              }`}
            >
              {trend > 0 ? "Improving" : trend < 0 ? "Declining" : "Stable"}
            </span>{" "}
            ({trend > 0 ? "+" : ""}
            {(trend * 100).toFixed(2)}% per iteration)
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
