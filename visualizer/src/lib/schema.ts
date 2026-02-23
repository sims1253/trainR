/**
 * Visualizer Data Contract - Version 1
 *
 * This file defines the stable frontend-facing schema for the benchmark visualizer.
 * The schema is versioned to prevent drift while the backend evolves.
 *
 * CONTRACT VERSION: 1
 */

// ============================================================================
// Schema Version
// ============================================================================

export const VISUALIZER_DATA_VERSION = 1 as const;
export type VisualizerDataVersion = typeof VISUALIZER_DATA_VERSION;

// ============================================================================
// Core Data Types
// ============================================================================

// ============================================================================
// Support/Tool Dimension Types
// ============================================================================

/**
 * Support modes for agent/tool support configurations
 */
export type SupportModeV1 =
  | "none"
  | "system_only"
  | "agents_only"
  | "system_plus_agents"
  | "single_skill"
  | "collection_forced"
  | "collection_selective";

/**
 * Tool version identifiers
 */
export type ToolVersionV1 = "v1" | "patch_v1" | "patch_v2" | "patch_v3" | "custom" | string;

/**
 * Support profile identifier
 */
export interface SupportProfileRefV1 {
  profile_id: string;
  mode: SupportModeV1;
  name?: string;
}

/**
 * Tool profile identifier
 */
export interface ToolProfileRefV1 {
  tool_id: string;
  version: ToolVersionV1;
  name?: string;
  variant?: string;
}

/**
 * Overall result statistics for a skill evaluation
 */
export interface OverallResultV1 {
  pass_rate: number;
  total: number;
  passed: number;
  failed: number;
}

/**
 * Difficulty breakdown for pass rates
 */
export interface DifficultyBreakdownV1 {
  easy: number;
  medium: number;
  hard: number;
}

/**
 * Package-level pass rate mapping
 */
export type PackageBreakdownV1 = Record<string, number>;

/**
 * Task split types for filtering
 */
export type TaskSplitV1 = "train" | "dev" | "test";

/**
 * Paired delta result for A/B comparison
 */
export interface PairedDeltaV1 {
  /** Identifier for the A profile */
  profile_a: string;
  /** Identifier for the B profile */
  profile_b: string;
  /** Delta in pass rate (B - A) */
  delta_pass_rate: number;
  /** Delta in cost (B - A), optional */
  delta_cost?: number;
  /** Delta in latency (B - A), optional */
  delta_latency_ms?: number;
  /** Number of paired samples */
  sample_count: number;
  /** Statistical significance (p-value), optional */
  p_value?: number;
  /** Model this delta applies to */
  model_name?: string;
}

/**
 * Skill-level results containing overall, difficulty, and package breakdowns
 */
export interface SkillResultV1 {
  overall: OverallResultV1;
  by_difficulty: DifficultyBreakdownV1;
  by_package: PackageBreakdownV1;
}

/**
 * Skill type identifier
 */
export type SkillTypeV1 = "no_skill" | "posit_skill";

/**
 * All skill results for a model
 */
export interface ModelSkillResultsV1 {
  no_skill: SkillResultV1;
  posit_skill: SkillResultV1;
}

/**
 * Individual model result entry
 */
export interface ModelResultV1 {
  name: string;
  display_name: string;
  provider: string;
  results: ModelSkillResultsV1;
  /** Support profile this result was generated with (optional) */
  support_profile?: SupportProfileRefV1;
  /** Tool profile this result was generated with (optional) */
  tool_profile?: ToolProfileRefV1;
}

/**
 * Metadata about the benchmark run
 */
export interface MetadataV1 {
  last_updated: string;
  total_tasks: number;
  packages: string[];
  runs_included: number;
  /** Available support profiles in the data */
  support_profiles?: SupportProfileRefV1[];
  /** Available tool profiles in the data */
  tool_profiles?: ToolProfileRefV1[];
  /** Available task splits */
  task_splits?: TaskSplitV1[];
  /** Difficulty levels available */
  difficulty_levels?: string[];
  /** Paired delta results for A/B comparisons */
  paired_deltas?: PairedDeltaV1[];
}

/**
 * Complete visualizer data structure - Version 1
 *
 * This is the canonical schema for all data consumed by the visualizer.
 * Any changes to this schema require a version bump.
 */
export interface VisualizerDataV1 {
  visualizer_data_version: VisualizerDataVersion;
  models: ModelResultV1[];
  metadata: MetadataV1;
}

// ============================================================================
// Validation Types
// ============================================================================

/**
 * Result of validation - either success with data or error with message
 */
export type ValidationResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string; details?: string[] };

// ============================================================================
// Validation Helpers
// ============================================================================

/**
 * Type guard to check if a value is a valid number
 */
function isValidNumber(value: unknown): value is number {
  return typeof value === "number" && !isNaN(value);
}

/**
 * Type guard to check if a value is a valid string
 */
function isValidString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Type guard to check if a value is a valid object (not null, not array)
 */
function isValidObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validate OverallResultV1 structure
 */
function validateOverallResult(data: unknown, path: string): ValidationResult<OverallResultV1> {
  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const requiredFields: (keyof OverallResultV1)[] = ["pass_rate", "total", "passed", "failed"];
  const errors: string[] = [];

  for (const field of requiredFields) {
    if (!isValidNumber(data[field])) {
      errors.push(`${path}.${field} must be a valid number (got: ${typeof data[field]})`);
    }
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid overall result at ${path}`, details: errors };
  }

  // At this point we know all fields are valid numbers
  const passRate = data.pass_rate as number;
  const passed = data.passed as number;
  const failed = data.failed as number;
  const total = data.total as number;

  // Additional validation: pass_rate should be between 0 and 1
  if (passRate < 0 || passRate > 1) {
    errors.push(`${path}.pass_rate must be between 0 and 1 (got: ${passRate})`);
  }

  // Additional validation: counts should match
  if (passed + failed !== total) {
    errors.push(
      `${path} has inconsistent counts: passed (${passed}) + failed (${failed}) != total (${total})`
    );
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid overall result at ${path}`, details: errors };
  }

  return { ok: true, data: data as unknown as OverallResultV1 };
}

/**
 * Validate DifficultyBreakdownV1 structure
 */
function validateDifficultyBreakdown(data: unknown, path: string): ValidationResult<DifficultyBreakdownV1> {
  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const requiredLevels: (keyof DifficultyBreakdownV1)[] = ["easy", "medium", "hard"];
  const errors: string[] = [];

  for (const level of requiredLevels) {
    if (!isValidNumber(data[level])) {
      errors.push(`${path}.${level} must be a valid number (got: ${typeof data[level]})`);
    } else if (data[level] < 0 || data[level] > 1) {
      errors.push(`${path}.${level} must be between 0 and 1 (got: ${data[level]})`);
    }
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid difficulty breakdown at ${path}`, details: errors };
  }

  return { ok: true, data: data as unknown as DifficultyBreakdownV1 };
}

/**
 * Validate PackageBreakdownV1 structure
 */
function validatePackageBreakdown(data: unknown, path: string): ValidationResult<PackageBreakdownV1> {
  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const errors: string[] = [];

  for (const [packageName, rate] of Object.entries(data)) {
    if (!isValidString(packageName)) {
      errors.push(`${path} has invalid package name`);
    }
    if (!isValidNumber(rate)) {
      errors.push(`${path}.${packageName} must be a valid number (got: ${typeof rate})`);
    } else if (rate < 0 || rate > 1) {
      errors.push(`${path}.${packageName} must be between 0 and 1 (got: ${rate})`);
    }
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid package breakdown at ${path}`, details: errors };
  }

  return { ok: true, data: data as unknown as PackageBreakdownV1 };
}

/**
 * Validate SkillResultV1 structure
 */
function validateSkillResult(data: unknown, path: string): ValidationResult<SkillResultV1> {
  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const overallResult = validateOverallResult(data.overall, `${path}.overall`);
  if (!overallResult.ok) return overallResult;

  const difficultyResult = validateDifficultyBreakdown(data.by_difficulty, `${path}.by_difficulty`);
  if (!difficultyResult.ok) return difficultyResult;

  const packageResult = validatePackageBreakdown(data.by_package, `${path}.by_package`);
  if (!packageResult.ok) return packageResult;

  return {
    ok: true,
    data: {
      overall: overallResult.data,
      by_difficulty: difficultyResult.data,
      by_package: packageResult.data,
    },
  };
}

/**
 * Validate ModelSkillResultsV1 structure
 */
function validateModelSkillResults(data: unknown, path: string): ValidationResult<ModelSkillResultsV1> {
  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const requiredSkills: SkillTypeV1[] = ["no_skill", "posit_skill"];
  const errors: string[] = [];
  const results: Partial<ModelSkillResultsV1> = {};

  for (const skill of requiredSkills) {
    const skillResult = validateSkillResult(data[skill], `${path}.${skill}`);
    if (!skillResult.ok) {
      errors.push(...(skillResult.details || [skillResult.error]));
    } else {
      results[skill] = skillResult.data;
    }
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid skill results at ${path}`, details: errors };
  }

  return { ok: true, data: results as ModelSkillResultsV1 };
}

/**
 * Validate ModelResultV1 structure
 */
function validateModelResult(data: unknown, index: number): ValidationResult<ModelResultV1> {
  const path = `models[${index}]`;

  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const errors: string[] = [];

  // Validate required string fields
  if (!isValidString(data.name)) {
    errors.push(`${path}.name must be a non-empty string (got: ${typeof data.name})`);
  }
  if (!isValidString(data.display_name)) {
    errors.push(`${path}.display_name must be a non-empty string (got: ${typeof data.display_name})`);
  }
  if (!isValidString(data.provider)) {
    errors.push(`${path}.provider must be a non-empty string (got: ${typeof data.provider})`);
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid model at ${path}`, details: errors };
  }

  const resultsValidation = validateModelSkillResults(data.results, `${path}.results`);
  if (!resultsValidation.ok) return resultsValidation;

  // Optional profile fields - just pass through if present
  const result: ModelResultV1 = {
    name: data.name as string,
    display_name: data.display_name as string,
    provider: data.provider as string,
    results: resultsValidation.data,
  };

  // Add optional profile fields if present
  if (data.support_profile && isValidObject(data.support_profile)) {
    result.support_profile = data.support_profile as unknown as SupportProfileRefV1;
  }
  if (data.tool_profile && isValidObject(data.tool_profile)) {
    result.tool_profile = data.tool_profile as unknown as ToolProfileRefV1;
  }

  return { ok: true, data: result };
}

/**
 * Validate MetadataV1 structure
 */
function validateMetadata(data: unknown): ValidationResult<MetadataV1> {
  const path = "metadata";

  if (!isValidObject(data)) {
    return { ok: false, error: `${path} must be an object`, details: [`Received: ${typeof data}`] };
  }

  const errors: string[] = [];

  // Validate last_updated (ISO date string)
  if (!isValidString(data.last_updated)) {
    errors.push(`${path}.last_updated must be a string (got: ${typeof data.last_updated})`);
  } else {
    const date = new Date(data.last_updated);
    if (isNaN(date.getTime())) {
      errors.push(`${path}.last_updated must be a valid ISO date string`);
    }
  }

  // Validate total_tasks
  if (!isValidNumber(data.total_tasks) || data.total_tasks < 0 || !Number.isInteger(data.total_tasks)) {
    errors.push(`${path}.total_tasks must be a non-negative integer (got: ${data.total_tasks})`);
  }

  // Validate packages array
  if (!Array.isArray(data.packages)) {
    errors.push(`${path}.packages must be an array (got: ${typeof data.packages})`);
  } else {
    for (let i = 0; i < data.packages.length; i++) {
      if (!isValidString(data.packages[i])) {
        errors.push(`${path}.packages[${i}] must be a non-empty string`);
      }
    }
  }

  // Validate runs_included
  if (!isValidNumber(data.runs_included) || data.runs_included < 0 || !Number.isInteger(data.runs_included)) {
    errors.push(`${path}.runs_included must be a non-negative integer (got: ${data.runs_included})`);
  }

  if (errors.length > 0) {
    return { ok: false, error: `Invalid metadata`, details: errors };
  }

  return { ok: true, data: data as unknown as MetadataV1 };
}

/**
 * Validate the complete VisualizerDataV1 structure
 *
 * This is the main entry point for runtime validation of incoming data.
 * It performs strict validation and returns either the validated data
 * or an actionable error message.
 *
 * @param data - Raw data to validate (typically from JSON.parse)
 * @returns ValidationResult with either validated data or error details
 *
 * @example
 * ```typescript
 * const rawData = JSON.parse(jsonString);
 * const result = validateVisualizerDataV1(rawData);
 *
 * if (result.ok) {
 *   // Use result.data safely
 * } else {
 *   console.error(result.error);
 *   console.error(result.details);
 * }
 * ```
 */
export function validateVisualizerDataV1(data: unknown): ValidationResult<VisualizerDataV1> {
  if (!isValidObject(data)) {
    return {
      ok: false,
      error: "Data must be a valid object",
      details: [`Received: ${typeof data}. Ensure the JSON file exists and contains valid JSON.`],
    };
  }

  // Check version field
  if (data.visualizer_data_version !== VISUALIZER_DATA_VERSION) {
    return {
      ok: false,
      error: `Invalid or missing visualizer_data_version`,
      details: [
        `Expected version: ${VISUALIZER_DATA_VERSION}`,
        `Received: ${JSON.stringify(data.visualizer_data_version)}`,
        "The data schema may have changed. Please regenerate the benchmark data.",
      ],
    };
  }

  // Validate models array
  if (!Array.isArray(data.models)) {
    return {
      ok: false,
      error: "models must be an array",
      details: [`Received: ${typeof data.models}`],
    };
  }

  if (data.models.length === 0) {
    return {
      ok: false,
      error: "models array cannot be empty",
      details: ["At least one model result is required."],
    };
  }

  const validatedModels: ModelResultV1[] = [];
  for (let i = 0; i < data.models.length; i++) {
    const modelResult = validateModelResult(data.models[i], i);
    if (!modelResult.ok) return modelResult;
    validatedModels.push(modelResult.data);
  }

  // Validate metadata
  const metadataResult = validateMetadata(data.metadata);
  if (!metadataResult.ok) return metadataResult;

  // Cross-validate: packages in metadata should cover all packages in model results
  const metadataPackages = new Set(metadataResult.data.packages);
  for (const model of validatedModels) {
    for (const skill of ["no_skill", "posit_skill"] as const) {
      for (const pkg of Object.keys(model.results[skill].by_package)) {
        if (!metadataPackages.has(pkg)) {
          console.warn(
            `Warning: Package "${pkg}" found in model "${model.name}" results but not in metadata.packages`
          );
        }
      }
    }
  }

  return {
    ok: true,
    data: {
      visualizer_data_version: VISUALIZER_DATA_VERSION,
      models: validatedModels,
      metadata: metadataResult.data,
    },
  };
}

/**
 * Strict runtime guard that throws on invalid data.
 * Use this when you want to fail fast with an actionable error message.
 *
 * @param data - Raw data to validate
 * @returns The validated VisualizerDataV1
 * @throws Error with detailed message about what went wrong
 *
 * @example
 * ```typescript
 * try {
 *   const data = guardVisualizerDataV1(JSON.parse(jsonString));
 *   // Use data safely
 * } catch (error) {
 *   // Show error to user
 * }
 * ```
 */
export function guardVisualizerDataV1(data: unknown): VisualizerDataV1 {
  const result = validateVisualizerDataV1(data);

  if (!result.ok) {
    const message = [
      `Visualizer Data Validation Error: ${result.error}`,
      "",
      ...(result.details || []),
      "",
      "To fix this issue:",
      "1. Run: uv run python scripts/aggregate_results.py",
      "2. Ensure the output JSON is valid",
      "3. Check that all required fields are present",
    ].join("\n");

    throw new Error(message);
  }

  return result.data;
}

/**
 * Check if unknown data is valid VisualizerDataV1 without throwing.
 * Returns a tuple of [isValid, dataOrError].
 *
 * @param data - Raw data to check
 * @returns Tuple of [boolean, data | error message]
 */
export function isValidVisualizerDataV1(data: unknown): [boolean, VisualizerDataV1 | string] {
  const result = validateVisualizerDataV1(data);
  if (result.ok) {
    return [true, result.data];
  }
  return [false, `${result.error}: ${result.details?.join("; ")}`];
}
