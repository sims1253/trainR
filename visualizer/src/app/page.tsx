"use client";

import { useState, useMemo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatPercent } from "@/lib/utils";
import benchmarkDataRaw from "@/data/benchmark-results.json";
import { BenchmarkData, ModelResult, validateVisualizerDataV1 } from "@/lib/types";
import { DifficultyChart } from "@/components/charts/difficulty-chart";
import { PackageChart } from "@/components/charts/package-chart";
import { ArrowUpIcon, ArrowDownIcon, AlertTriangleIcon } from "lucide-react";

import { Header } from "@/components/header";
import { Footer } from "@/components/footer";

/**
 * Error display component for invalid data
 */
function DataError({ error }: { error: string }) {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 container mx-auto py-8 px-4">
        <div className="max-w-2xl mx-auto p-6 bg-destructive/10 border border-destructive rounded-lg">
          <div className="flex items-start gap-4">
            <AlertTriangleIcon className="w-8 h-8 text-destructive flex-shrink-0 mt-1" />
            <div className="space-y-2">
              <h1 className="text-xl font-semibold text-destructive">
                Failed to Load Benchmark Data
              </h1>
              <p className="text-muted-foreground">
                The benchmark data file is invalid or corrupted.
              </p>
              <div className="mt-4 p-4 bg-muted rounded-md font-mono text-sm whitespace-pre-wrap overflow-auto max-h-[300px]">
                {error}
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                To fix this issue, regenerate the benchmark data:
              </p>
              <code className="block p-2 bg-muted rounded text-sm">
                uv run python scripts/export_visualizer_data.py --sample
              </code>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

/**
 * Dashboard content component - only rendered when data is valid
 */
function DashboardContent({ data }: { data: BenchmarkData }) {
  const [skillFilter, setSkillFilter] = useState<string>("posit_skill");
  const [difficultyFilter, setDifficultyFilter] = useState<string>("all");
  const [packageFilter, setPackageFilter] = useState<string>("all");
  const [selectedModelIndex, setSelectedModelIndex] = useState<number>(0);

  const filteredModels = useMemo(() => {
    return data.models;
  }, [data.models]);

  const selectedModel = filteredModels[selectedModelIndex] || filteredModels[0];

  const getPassRate = (model: ModelResult, skill: string) => {
    const skillData = model.results[skill as keyof typeof model.results];
    if (!skillData) return null;

    if (difficultyFilter !== "all") {
      return skillData.by_difficulty[difficultyFilter as keyof typeof skillData.by_difficulty] ?? null;
    }

    if (packageFilter !== "all") {
      return skillData.by_package[packageFilter] ?? null;
    }

    return skillData.overall.pass_rate;
  };

  const calculateDelta = (model: ModelResult) => {
    const noSkillRate = getPassRate(model, "no_skill");
    const positSkillRate = getPassRate(model, "posit_skill");

    if (noSkillRate === null || positSkillRate === null) return null;
    return positSkillRate - noSkillRate;
  };

  const renderDelta = (delta: number | null) => {
    if (delta === null) return <span className="text-muted-foreground">-</span>;
    
    const value = (delta * 100).toFixed(1);
    if (delta > 0) {
      return (
        <span className="text-green-600 flex items-center justify-end gap-1 font-medium">
          <ArrowUpIcon className="w-3 h-3" />
          +{value}pp
        </span>
      );
    } else if (delta < 0) {
      return (
        <span className="text-red-600 flex items-center justify-end gap-1 font-medium">
          <ArrowDownIcon className="w-3 h-3" />
          {value}pp
        </span>
      );
    } else {
      return <span className="text-muted-foreground font-medium">0.0pp</span>;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-1 container mx-auto py-8 px-4 space-y-8">
        <div className="flex flex-wrap gap-4 items-end bg-muted/30 p-4 rounded-lg border">
          <div className="space-y-2">
            <label className="text-sm font-medium">Skill Filter</label>
            <Select value={skillFilter} onValueChange={setSkillFilter}>
              <SelectTrigger className="w-[180px] bg-background">
                <SelectValue placeholder="Select Skill" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Skills</SelectItem>
                <SelectItem value="no_skill">No Skill</SelectItem>
                <SelectItem value="posit_skill">Posit Skill</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Difficulty</label>
            <Tabs value={difficultyFilter} onValueChange={setDifficultyFilter} className="w-auto">
              <TabsList className="bg-background border">
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="easy">Easy</TabsTrigger>
                <TabsTrigger value="medium">Medium</TabsTrigger>
                <TabsTrigger value="hard">Hard</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Package</label>
            <Select value={packageFilter} onValueChange={setPackageFilter}>
              <SelectTrigger className="w-[180px] bg-background">
                <SelectValue placeholder="All Packages" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Packages</SelectItem>
                {data.metadata.packages.map((pkg) => (
                  <SelectItem key={pkg} value={pkg}>
                    {pkg}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="rounded-md border overflow-hidden bg-background">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="w-[300px]">Model</TableHead>
                <TableHead>Provider</TableHead>
                {(skillFilter === "all" || skillFilter === "no_skill") && (
                  <TableHead className="text-right">No Skill</TableHead>
                )}
                {(skillFilter === "all" || skillFilter === "posit_skill") && (
                  <TableHead className="text-right">Posit Skill</TableHead>
                )}
                <TableHead className="text-right">Delta</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredModels.map((model, index) => {
                const noSkillRate = getPassRate(model, "no_skill");
                const positSkillRate = getPassRate(model, "posit_skill");
                const delta = calculateDelta(model);

                return (
                  <TableRow 
                    key={model.name}
                    className={`cursor-pointer hover:bg-muted/50 transition-colors ${selectedModelIndex === index ? 'bg-primary/5' : ''}`}
                    onClick={() => setSelectedModelIndex(index)}
                  >
                    <TableCell className="font-medium">
                      {model.display_name}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-normal">{model.provider}</Badge>
                    </TableCell>
                    {(skillFilter === "all" || skillFilter === "no_skill") && (
                      <TableCell className="text-right font-mono">
                        {noSkillRate !== null ? formatPercent(noSkillRate) : "N/A"}
                      </TableCell>
                    )}
                    {(skillFilter === "all" || skillFilter === "posit_skill") && (
                      <TableCell className="text-right font-mono">
                        <div className="flex items-center justify-end gap-1">
                          {positSkillRate !== null ? formatPercent(positSkillRate) : "N/A"}
                          {positSkillRate === 1 && <span className="text-green-500 text-xs">ok</span>}
                        </div>
                      </TableCell>
                    )}
                    <TableCell className="text-right font-mono">
                      {renderDelta(delta)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <DifficultyChart 
            models={filteredModels} 
            skill={skillFilter === "all" ? "posit_skill" : skillFilter} 
          />
          <PackageChart 
            model={selectedModel} 
            skill={skillFilter} 
          />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default function Home() {
  // Validate data at runtime with explicit error handling
  const validationResult = useMemo(() => {
    return validateVisualizerDataV1(benchmarkDataRaw);
  }, []);

  // Show error state if validation fails
  if (!validationResult.ok) {
    const errorMessage = [
      validationResult.error,
      ...(validationResult.details || []),
    ].join("\n");
    return <DataError error={errorMessage} />;
  }

  return <DashboardContent data={validationResult.data} />;
}
