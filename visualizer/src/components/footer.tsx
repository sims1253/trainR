import benchmarkData from "@/data/benchmark-results.json"
import { BenchmarkData } from "@/lib/types"

export function Footer() {
  const data = benchmarkData as BenchmarkData
  const lastUpdated = new Date(data.metadata.last_updated).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })
  
  const totalTasks = data.metadata.total_tasks
  const modelCount = data.models.length

  return (
    <footer className="border-t bg-muted/30 mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-muted-foreground">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span>Last updated: {lastUpdated}</span>
            <span className="hidden md:inline">•</span>
            <span>{totalTasks} tasks</span>
            <span className="hidden md:inline">•</span>
            <span>{modelCount} models</span>
          </div>
          <div className="flex items-center gap-2">
            <span>Inspired by <a href="https://github.com/OpenCode-AI/skatebench" target="_blank" rel="noreferrer" className="underline hover:text-foreground">skatebench</a></span>
            <span>•</span>
            <span>Built with Next.js & shadcn/ui</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
