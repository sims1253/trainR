"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ClientOnly } from "@/components/client-only";
import { ModelResult } from "@/lib/types";

interface PackageChartProps {
  model: ModelResult;
  skill: string;
}

interface PackageChartData {
  name: string;
  no_skill?: number;
  posit_skill?: number;
}

export function PackageChart({ model, skill }: PackageChartProps) {
  const skillsToShow = skill === "all" ? ["no_skill", "posit_skill"] : [skill];
  
  // Extract all package names from the results
  const allPackages = Array.from(new Set([
    ...Object.keys(model.results.no_skill?.by_package || {}),
    ...Object.keys(model.results.posit_skill?.by_package || {}),
  ]));

  const chartData: PackageChartData[] = allPackages.map(pkg => {
    const data: PackageChartData = { name: pkg };
    if (skillsToShow.includes("no_skill")) {
      data.no_skill = (model.results.no_skill?.by_package?.[pkg] || 0) * 100;
    }
    if (skillsToShow.includes("posit_skill")) {
      data.posit_skill = (model.results.posit_skill?.by_package?.[pkg] || 0) * 100;
    }
    return data;
  });

  return (
    <Card className="col-span-1">
      <CardHeader>
        <CardTitle>Performance by Package: {model.display_name}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] w-full">
          <ClientOnly
            fallback={
              <div className="h-full w-full flex items-center justify-center text-muted-foreground">
                Loading chart...
              </div>
            }
          >
            <ResponsiveContainer width="100%" height="100%" minWidth={300} minHeight={200}>
              <BarChart
                layout="vertical"
                data={chartData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 40,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis 
                  type="number"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `${value}%`}
                  domain={[0, 100]}
                />
                <YAxis 
                  dataKey="name" 
                  type="category"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  width={80}
                />
                <Tooltip 
                  cursor={{ fill: 'transparent' }}
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  formatter={(value) => value !== undefined ? [`${(value as number).toFixed(1)}%`, ''] : ['', '']}
                />
                <Legend />
                {skillsToShow.includes("no_skill") && (
                  <Bar 
                    dataKey="no_skill" 
                    name="No Skill" 
                    fill="#3b82f6" 
                    radius={[0, 4, 4, 0]} 
                  />
                )}
                {skillsToShow.includes("posit_skill") && (
                  <Bar 
                    dataKey="posit_skill" 
                    name="Posit Skill" 
                    fill="#8b5cf6" 
                    radius={[0, 4, 4, 0]} 
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
          </ClientOnly>
        </div>
      </CardContent>
    </Card>
  );
}
