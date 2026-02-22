import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { formatPercent } from "@/lib/utils";
import benchmarkData from "@/data/benchmark-results.json";
import { BenchmarkData } from "@/lib/types";

export default function Home() {
  const data = benchmarkData as BenchmarkData;

  return (
    <div className="container mx-auto py-10 px-4">
      <header className="mb-10">
        <h1 className="text-4xl font-bold tracking-tight mb-2">trainR Benchmark</h1>
        <p className="text-muted-foreground">
          Evaluating AI models on R package testing and development tasks.
        </p>
      </header>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[300px]">Model</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead className="text-right">No Skill (Pass Rate)</TableHead>
              <TableHead className="text-right">Posit Skill (Pass Rate)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.models.map((model) => (
              <TableRow key={model.name}>
                <TableCell className="font-medium">
                  {model.display_name}
                </TableCell>
                <TableCell>
                  <Badge variant="outline">{model.provider}</Badge>
                </TableCell>
                <TableCell className="text-right font-mono">
                  {model.results.no_skill 
                    ? formatPercent(model.results.no_skill.overall.pass_rate)
                    : "N/A"}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {model.results.posit_skill 
                    ? formatPercent(model.results.posit_skill.overall.pass_rate)
                    : "N/A"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <footer className="mt-10 text-sm text-muted-foreground">
        <p>Last updated: {new Date(data.metadata.last_updated).toLocaleString()}</p>
        <p>Total tasks in benchmark: {data.metadata.total_tasks}</p>
      </footer>
    </div>
  );
}
