# trainR Benchmark Visualizer

A web-based dashboard for exploring and comparing AI model performance on R package testing tasks.

Live at: [rbench.scholzmx.com](https://rbench.scholzmx.com)

Inspired by [skatebench](https://github.com/T3-Content/skatebench)

## Features

- **Leaderboard**: Compare models across No Skill / Posit Skill with delta column
- **Filters**: By skill type, difficulty level (Easy/Medium/Hard), and R package
- **Charts**: 
  - Performance by Difficulty (grouped bar chart)
  - Performance by Package (horizontal bar chart)
- **Dark Mode**: System-aware with manual toggle
- **Responsive**: Works on desktop and mobile

## Tech Stack

- Next.js 16 + React 19
- Tailwind CSS 4
- shadcn/ui components
- Recharts for visualizations
- next-themes for dark mode

## Local Development

```bash
cd visualizer
bun dev
```

Open [localhost:3000](http://localhost:3000)

## Data Management

The visualizer consumes aggregated benchmark data from `src/data/benchmark-results.json`.

### Regenerate Data

After running benchmarks, regenerate the data:

```bash
# From project root
uv run python scripts/aggregate_results.py
```

This script:
- Parses all `results/baselines/**/*.json` files
- Maps skill names: `no_skill` → No Skill, `testing-r-packages-orig` → Posit Skill
- Calculates per-difficulty and per-package breakdowns
- Outputs to `visualizer/src/data/benchmark-results.json`

### Data Format

```json
{
  "visualizer_data_version": 1,
  "models": [
    {
      "name": "gpt-oss-120b:free",
      "display_name": "OpenAI GPT-OSS-120B",
      "provider": "openrouter",
      "results": {
        "no_skill": {
          "overall": {"pass_rate": 0.333, "total": 12, "passed": 4, "failed": 8},
          "by_difficulty": {"easy": 0.8, "medium": 0.3, "hard": 0.1},
          "by_package": {"cli": 0.33, "dplyr": 0.4}
        },
        "posit_skill": { ... }
      }
    }
  ],
  "metadata": {
    "last_updated": "2026-02-22T14:00:00Z",
    "total_tasks": 138,
    "packages": ["cli", "dplyr", ...],
    "runs_included": 10
  }
}
```

## Data Contract

The visualizer enforces a versioned data contract to prevent drift between the backend aggregation script and the frontend. The schema is defined in `src/lib/schema.ts`.

### Contract Version

Current version: **1**

The `visualizer_data_version` field must be present and match the expected version. This ensures backward compatibility when the data format evolves.

### Validation

Data is validated at runtime using `validateVisualizerDataV1()` which:

1. Checks required fields and types
2. Validates numeric ranges (pass rates must be 0-1)
3. Verifies consistency (passed + failed = total)
4. Ensures both `no_skill` and `posit_skill` results exist

### Error Handling

If validation fails, the UI displays an explicit error state with:

- The validation error message
- Detailed information about what failed
- Instructions to regenerate the data

### Schema Types

| Type | Description |
|------|-------------|
| `VisualizerDataV1` | Root data structure with version, models, metadata |
| `ModelResultV1` | Individual model entry with results |
| `SkillResultV1` | Per-skill results with overall, difficulty, package breakdowns |
| `MetadataV1` | Benchmark metadata (timestamp, packages, run counts) |

### Updating the Schema

If you need to change the data format:

1. Increment `VISUALIZER_DATA_VERSION` in `schema.ts`
2. Create new types (e.g., `VisualizerDataV2`)
3. Add a migration function for backward compatibility
4. Update `aggregate_results.py` to output the new version
5. Document breaking changes here

### Validation Functions

```typescript
// Returns result object with ok/error
const result = validateVisualizerDataV1(data);
if (result.ok) {
  // Use result.data
} else {
  // Handle result.error and result.details
}

// Throws on invalid data (for fail-fast scenarios)
const data = guardVisualizerDataV1(rawData);

// Returns tuple [isValid, dataOrError]
const [valid, dataOrError] = isValidVisualizerDataV1(data);
```

## Deployment

### Vercel Setup (One-time)

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import GitHub repo: `m0hawk/trainR`
3. **Root Directory**: Set to `visualizer` (important!)
4. Framework: Next.js (auto-detected)
5. Click **Deploy**

### Custom Domain

1. Project Settings → Domains
2. Add `rbench.scholzmx.com`
3. Add DNS record at your registrar:
   - Type: CNAME
   - Name: `rbench`
   - Value: `cname.vercel-dns.com`

### Update Workflow

```bash
# 1. Run benchmarks
make baseline-all  # or individual benchmark commands

# 2. Regenerate data
uv run python scripts/aggregate_results.py

# 3. Commit and push (Vercel auto-deploys)
git add visualizer/src/data/benchmark-results.json
git commit -m "Update benchmark data"
git push
```

## Project Structure

```
visualizer/
├── src/
│   ├── app/
│   │   ├── layout.tsx        # Root layout with ThemeProvider
│   │   ├── page.tsx          # Main dashboard
│   │   ├── globals.css       # Tailwind + CSS variables
│   │   └── icon.svg          # R hexagon favicon
│   ├── components/
│   │   ├── header.tsx        # Title, nav, theme toggle
│   │   ├── footer.tsx        # Metadata footer
│   │   ├── theme-provider.tsx
│   │   ├── theme-toggle.tsx
│   │   ├── charts/
│   │   │   ├── difficulty-chart.tsx
│   │   │   └── package-chart.tsx
│   │   └── ui/               # shadcn components
│   ├── lib/
│   │   ├── schema.ts         # Versioned data contract and validation
│   │   ├── types.ts          # TypeScript interfaces (re-exports from schema)
│   │   └── utils.ts          # Helpers (formatPercent, etc.)
│   └── data/
│       └── benchmark-results.json
├── package.json
├── vercel.json
└── README.md
```

## Adding New Models

New models are automatically picked up by the aggregation script if their results are in `results/baselines/` with the naming pattern:
- `eval_{model}_{skill}_{timestamp}.json` (top-level)
- `{model}_{skill}_{timestamp}.json` (in subdirectories)

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Easy | Green | #22c55e |
| Medium | Yellow | #eab308 |
| Hard | Red | #ef4444 |
| No Skill | Blue | #3b82f6 |
| Posit Skill | Purple | #8b5cf6 |
