"use client";

import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type {
  SupportProfileRefV1,
  ToolProfileRefV1,
  TaskSplitV1,
} from "@/lib/types";
import { FilterIcon, XIcon } from "lucide-react";

export interface DimensionFilters {
  supportProfile: string;
  toolProfile: string;
  taskSplit: string;
  difficulty: string;
  packageFilter: string;
}

interface DimensionFilterProps {
  filters: DimensionFilters;
  onFiltersChange: (filters: DimensionFilters) => void;
  supportProfiles?: SupportProfileRefV1[];
  toolProfiles?: ToolProfileRefV1[];
  taskSplits?: TaskSplitV1[];
  packages?: string[];
  difficultyLevels?: string[];
}

/**
 * Get display name for support mode
 */
function getSupportModeLabel(mode: string): string {
  const labels: Record<string, string> = {
    none: "No Support",
    system_only: "System Prompt Only",
    agents_only: "Agents Only",
    system_plus_agents: "System + Agents",
    single_skill: "Single Skill",
    collection_forced: "Collection (Forced)",
    collection_selective: "Collection (Selective)",
  };
  return labels[mode] || mode;
}

/**
 * Dimension Filter Component
 *
 * Provides filters for support profiles, tool profiles, task splits,
 * difficulty, and packages to enable isolating dimensions during analysis.
 */
export function DimensionFilter({
  filters,
  onFiltersChange,
  supportProfiles = [],
  toolProfiles = [],
  taskSplits = [],
  packages = [],
  difficultyLevels = ["easy", "medium", "hard"],
}: DimensionFilterProps) {
  const updateFilter = (key: keyof DimensionFilters, value: string) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  const resetFilters = () => {
    onFiltersChange({
      supportProfile: "all",
      toolProfile: "all",
      taskSplit: "all",
      difficulty: "all",
      packageFilter: "all",
    });
  };

  const activeFilterCount = Object.values(filters).filter((v) => v !== "all").length;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <FilterIcon className="w-4 h-4" />
            Dimension Filters
          </CardTitle>
          {activeFilterCount > 0 && (
            <button
              onClick={resetFilters}
              className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
            >
              <XIcon className="w-3 h-3" />
              Clear ({activeFilterCount})
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Support Profile Filter */}
          {supportProfiles.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Support Profile</label>
              <Select
                value={filters.supportProfile}
                onValueChange={(v) => updateFilter("supportProfile", v)}
              >
                <SelectTrigger className="bg-background">
                  <SelectValue placeholder="All Profiles" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Support Profiles</SelectItem>
                  {supportProfiles.map((profile) => (
                    <SelectItem key={profile.profile_id} value={profile.profile_id}>
                      {profile.name || profile.profile_id}
                      <span className="text-muted-foreground ml-2">
                        ({getSupportModeLabel(profile.mode)})
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Tool Profile Filter */}
          {toolProfiles.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Tool Profile</label>
              <Select
                value={filters.toolProfile}
                onValueChange={(v) => updateFilter("toolProfile", v)}
              >
                <SelectTrigger className="bg-background">
                  <SelectValue placeholder="All Profiles" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Tool Profiles</SelectItem>
                  {toolProfiles.map((profile) => (
                    <SelectItem key={profile.tool_id} value={profile.tool_id}>
                      {profile.name || profile.tool_id}
                      {profile.variant && (
                        <Badge variant="secondary" className="ml-2 text-xs">
                          {profile.variant}
                        </Badge>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Task Split Filter */}
          {taskSplits.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Task Split</label>
              <Tabs
                value={filters.taskSplit}
                onValueChange={(v) => updateFilter("taskSplit", v)}
                className="w-auto"
              >
                <TabsList className="bg-background border">
                  <TabsTrigger value="all">All</TabsTrigger>
                  {taskSplits.map((split) => (
                    <TabsTrigger key={split} value={split}>
                      {split.charAt(0).toUpperCase() + split.slice(1)}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
            </div>
          )}

          {/* Difficulty Filter */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Difficulty</label>
            <Tabs
              value={filters.difficulty}
              onValueChange={(v) => updateFilter("difficulty", v)}
              className="w-auto"
            >
              <TabsList className="bg-background border">
                <TabsTrigger value="all">All</TabsTrigger>
                {difficultyLevels.map((level) => (
                  <TabsTrigger key={level} value={level}>
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>

          {/* Package Filter */}
          {packages.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Package</label>
              <Select
                value={filters.packageFilter}
                onValueChange={(v) => updateFilter("packageFilter", v)}
              >
                <SelectTrigger className="bg-background">
                  <SelectValue placeholder="All Packages" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Packages</SelectItem>
                  {packages.map((pkg) => (
                    <SelectItem key={pkg} value={pkg}>
                      {pkg}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {/* Active Filter Tags */}
        {activeFilterCount > 0 && (
          <div className="flex flex-wrap gap-2 pt-2 border-t">
            <span className="text-sm text-muted-foreground">Active:</span>
            {filters.supportProfile !== "all" && (
              <Badge variant="secondary" className="font-normal">
                Support: {filters.supportProfile}
              </Badge>
            )}
            {filters.toolProfile !== "all" && (
              <Badge variant="secondary" className="font-normal">
                Tool: {filters.toolProfile}
              </Badge>
            )}
            {filters.taskSplit !== "all" && (
              <Badge variant="secondary" className="font-normal">
                Split: {filters.taskSplit}
              </Badge>
            )}
            {filters.difficulty !== "all" && (
              <Badge variant="secondary" className="font-normal">
                Difficulty: {filters.difficulty}
              </Badge>
            )}
            {filters.packageFilter !== "all" && (
              <Badge variant="secondary" className="font-normal">
                Package: {filters.packageFilter}
              </Badge>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
