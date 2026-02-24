"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ClientOnly } from "@/components/client-only";
import type { PairedDeltaV1 } from "@/lib/types";
import { ArrowUpIcon, ArrowDownIcon, MinusIcon } from "lucide-react";

interface DeltaComparisonProps {
  pairedDeltas: PairedDeltaV1[];
  title?: string;
  description?: string;
  /** Dimension being compared: 'support' or 'tool' */
  dimension?: "support" | "tool";
}

/**
 * Format a number as a percentage point delta
 */
function formatDeltaPP(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)}pp`;
}

/**
 * Format cost delta
 */
function formatCostDelta(value: number | undefined): string {
  if (value === undefined) return "N/A";
  const sign = value >= 0 ? "+" : "";
  return `${sign}$${value.toFixed(4)}`;
}

/**
 * Format latency delta
 */
function formatLatencyDelta(value: number | undefined): string {
  if (value === undefined) return "N/A";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(0)}ms`;
}

/**
 * Render delta indicator
 */
function DeltaIndicator({ value, format = "pp" }: { value: number; format?: "pp" | "cost" | "latency" }) {
  if (value === 0) {
    return (
      <span className="flex items-center gap-1 text-muted-foreground">
        <MinusIcon className="w-3 h-3" />
        {format === "pp" ? "0.0pp" : format === "cost" ? "$0.00" : "0ms"}
      </span>
    );
  }

  const isPositive = value > 0;
  const Icon = isPositive ? ArrowUpIcon : ArrowDownIcon;
  const colorClass = isPositive ? "text-green-600" : "text-red-600";
  const displayValue =
    format === "pp"
      ? formatDeltaPP(value)
      : format === "cost"
      ? formatCostDelta(value)
      : formatLatencyDelta(value);

  return (
    <span className={`flex items-center gap-1 font-medium ${colorClass}`}>
      <Icon className="w-3 h-3" />
      {displayValue}
    </span>
  );
}

/**
 * Delta Comparison Chart Component
 *
 * Displays paired delta comparisons for A/B testing of support/tool profiles.
 */
export function DeltaComparison({
  pairedDeltas,
  title = "Paired Delta Comparison",
  description,
  dimension = "support",
}: DeltaComparisonProps) {
  if (!pairedDeltas || pairedDeltas.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>No paired delta data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[200px] text-muted-foreground">
            Run A/B comparison benchmarks to generate paired delta data
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare chart data
  const chartData = pairedDeltas.map((delta, index) => ({
    name: delta.model_name || `Comparison ${index + 1}`,
    profileA: delta.profile_a,
    profileB: delta.profile_b,
    deltaPassRate: (delta.delta_pass_rate || 0) * 100,
    deltaCost: delta.delta_cost,
    deltaLatency: delta.delta_latency_ms,
    sampleCount: delta.sample_count,
    pValue: delta.p_value,
  }));

  // Calculate aggregate stats
  const avgDelta =
    pairedDeltas.reduce((sum, d) => sum + (d.delta_pass_rate || 0), 0) / pairedDeltas.length;
  const positiveCount = pairedDeltas.filter((d) => (d.delta_pass_rate || 0) > 0).length;
  const negativeCount = pairedDeltas.filter((d) => (d.delta_pass_rate || 0) < 0).length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
        <div className="flex gap-2 mt-2">
          <Badge variant="outline" className="font-normal">
            {dimension === "support" ? "Support Profile" : "Tool Profile"} A/B Test
          </Badge>
          <Badge variant="secondary" className="font-normal">
            {pairedDeltas.length} comparisons
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 p-4 bg-muted/30 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Avg Delta</div>
            <div className="text-lg font-semibold">
              <DeltaIndicator value={avgDelta} />
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Improved</div>
            <div className="text-lg font-semibold text-green-600">{positiveCount}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Regressed</div>
            <div className="text-lg font-semibold text-red-600">{negativeCount}</div>
          </div>
        </div>

        {/* Bar Chart */}
        <div className="h-[250px] w-full">
          <ClientOnly
            fallback={
              <div className="h-full w-full flex items-center justify-center text-muted-foreground">
                Loading chart...
              </div>
            }
          >
            <ResponsiveContainer width="100%" height="100%" minWidth={300} minHeight={200}>
              <BarChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="name"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `${value.toFixed(0)}pp`}
                />
                <Tooltip
                  cursor={{ fill: "transparent" }}
                  contentStyle={{
                    borderRadius: "8px",
                    border: "none",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                  }}
                  formatter={(value, name) => {
                    if (name === "deltaPassRate") return [`${Number(value).toFixed(1)}pp`, "Pass Rate Delta"];
                    return [value, String(name)];
                  }}
                  labelFormatter={(label, payload) => {
                    if (payload && payload[0]?.payload) {
                      const data = payload[0].payload as { name?: string; profileA?: string; profileB?: string };
                      return `${data.name}: ${data.profileA} vs ${data.profileB}`;
                    }
                    return String(label);
                  }}
                />
                <ReferenceLine y={0} stroke="#888" strokeDasharray="3 3" />
                <Bar
                  dataKey="deltaPassRate"
                  name="Pass Rate Delta"
                  fill="#8b5cf6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </ClientOnly>
        </div>

        {/* Detail Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 px-2">Model</th>
                <th className="text-left py-2 px-2">Profile A</th>
                <th className="text-left py-2 px-2">Profile B</th>
                <th className="text-right py-2 px-2">Pass Rate</th>
                <th className="text-right py-2 px-2">Cost</th>
                <th className="text-right py-2 px-2">Latency</th>
                <th className="text-right py-2 px-2">Samples</th>
              </tr>
            </thead>
            <tbody>
              {pairedDeltas.map((delta, index) => (
                <tr key={index} className="border-b hover:bg-muted/50">
                  <td className="py-2 px-2 font-medium">{delta.model_name || `Test ${index + 1}`}</td>
                  <td className="py-2 px-2">
                    <Badge variant="outline" className="font-mono text-xs">
                      {delta.profile_a}
                    </Badge>
                  </td>
                  <td className="py-2 px-2">
                    <Badge variant="outline" className="font-mono text-xs">
                      {delta.profile_b}
                    </Badge>
                  </td>
                  <td className="py-2 px-2 text-right">
                    <DeltaIndicator value={delta.delta_pass_rate || 0} />
                  </td>
                  <td className="py-2 px-2 text-right text-muted-foreground">
                    {formatCostDelta(delta.delta_cost)}
                  </td>
                  <td className="py-2 px-2 text-right text-muted-foreground">
                    {formatLatencyDelta(delta.delta_latency_ms)}
                  </td>
                  <td className="py-2 px-2 text-right text-muted-foreground">
                    {delta.sample_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
