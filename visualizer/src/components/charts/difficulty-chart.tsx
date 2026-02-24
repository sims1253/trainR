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

interface DifficultyChartProps {
  models: ModelResult[];
  skill: string;
}

export function DifficultyChart({ models, skill }: DifficultyChartProps) {
  const chartData = models.map((model) => {
    const results = model.results[skill as keyof typeof model.results];
    return {
      name: model.display_name,
      easy: (results?.by_difficulty?.easy || 0) * 100,
      medium: (results?.by_difficulty?.medium || 0) * 100,
      hard: (results?.by_difficulty?.hard || 0) * 100,
    };
  });

  return (
    <Card className="col-span-1">
      <CardHeader>
        <CardTitle>Performance by Difficulty</CardTitle>
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
                data={chartData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
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
                  tickFormatter={(value) => `${value}%`}
                  domain={[0, 100]}
                />
                <Tooltip 
                  cursor={{ fill: 'transparent' }}
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  formatter={(value) => value !== undefined ? [`${(value as number).toFixed(1)}%`, ''] : ['', '']}
                />
                <Legend />
                <Bar 
                  dataKey="easy" 
                  name="Easy" 
                  fill="#22c55e" 
                  radius={[4, 4, 0, 0]} 
                />
                <Bar 
                  dataKey="medium" 
                  name="Medium" 
                  fill="#eab308" 
                  radius={[4, 4, 0, 0]} 
                />
                <Bar 
                  dataKey="hard" 
                  name="Hard" 
                  fill="#ef4444" 
                  radius={[4, 4, 0, 0]} 
                />
              </BarChart>
            </ResponsiveContainer>
          </ClientOnly>
        </div>
      </CardContent>
    </Card>
  );
}
