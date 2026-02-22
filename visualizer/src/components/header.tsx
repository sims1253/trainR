"use client"

import { ThemeToggle } from "@/components/theme-toggle"
import { GithubIcon, LayoutDashboardIcon } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Header() {
  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-primary/10 p-2 rounded-lg">
            <LayoutDashboardIcon className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">📊 trainR Benchmark</h1>
            <p className="text-sm text-muted-foreground">
              AI Model Performance on R Package Testing Tasks
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Button variant="outline" size="icon" asChild>
            <a 
              href="https://github.com/m0hawk/trainR" 
              target="_blank" 
              rel="noreferrer"
              title="GitHub Repository"
            >
              <GithubIcon className="h-[1.2rem] w-[1.2rem]" />
              <span className="sr-only">GitHub</span>
            </a>
          </Button>
        </div>
      </div>
    </header>
  )
}
