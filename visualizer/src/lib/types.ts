/**
 * TypeScript types for the Benchmark Visualizer
 *
 * These types are aligned with the V1 schema defined in schema.ts.
 * Import from this file in components for type safety.
 */

// Re-export schema types for backward compatibility
export type {
  VisualizerDataV1,
  ModelResultV1,
  ModelSkillResultsV1,
  SkillResultV1,
  OverallResultV1,
  DifficultyBreakdownV1,
  PackageBreakdownV1,
  MetadataV1,
  SkillTypeV1,
  ValidationResult,
  SupportModeV1,
  ToolVersionV1,
  SupportProfileRefV1,
  ToolProfileRefV1,
  TaskSplitV1,
  PairedDeltaV1,
} from "./schema";

// Re-export validation functions
export {
  validateVisualizerDataV1,
  guardVisualizerDataV1,
  isValidVisualizerDataV1,
  VISUALIZER_DATA_VERSION,
} from "./schema";

// Import types for local aliases
import type {
  VisualizerDataV1,
  ModelResultV1,
  OverallResultV1,
  DifficultyBreakdownV1,
  PackageBreakdownV1,
  MetadataV1,
  SkillResultV1,
} from "./schema";

// ============================================================================
// Convenience Aliases (for backward compatibility)
// ============================================================================

/**
 * Main benchmark data structure.
 * @deprecated Use VisualizerDataV1 for explicit versioning
 */
export type BenchmarkData = VisualizerDataV1;

/**
 * Model result entry.
 * @deprecated Use ModelResultV1 for explicit versioning
 */
export type ModelResult = ModelResultV1;

/**
 * Overall result statistics.
 * @deprecated Use OverallResultV1 for explicit versioning
 */
export type OverallResult = OverallResultV1;

/**
 * Difficulty breakdown mapping.
 * @deprecated Use DifficultyBreakdownV1 for explicit versioning
 */
export type DifficultyBreakdown = DifficultyBreakdownV1;

/**
 * Package-level breakdown.
 * @deprecated Use PackageBreakdownV1 for explicit versioning
 */
export type PackageBreakdown = PackageBreakdownV1;

/**
 * Benchmark metadata.
 * @deprecated Use MetadataV1 for explicit versioning
 */
export type BenchmarkMetadata = MetadataV1;

/**
 * Skill result structure.
 * @deprecated Use SkillResultV1 for explicit versioning
 */
export type SkillResult = SkillResultV1;
