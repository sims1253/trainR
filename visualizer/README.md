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
│   │   ├── types.ts          # TypeScript interfaces
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
