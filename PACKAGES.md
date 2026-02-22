# R Packages Used for Task Generation

This document lists all R packages used to generate testing tasks, along with the rationale for each.

## Overview

- **Total Tasks**: 125
- **Total Packages**: 17
- **Split Distribution**: 75 train / 25 dev / 25 held_out
- **Difficulty Distribution**: 50 hard (40%) / 50 medium (40%) / 25 easy (20%)

### Task Distribution Rationale

| Package | Tasks | Category | Rationale |
|---------|-------|----------|-----------|
| cli | 30 | Infrastructure | **Legacy initial dataset**. Original MVP focused on cli for rapid iteration. |
| testthat | 10 | Testing | Meta-testing: testing the testing framework itself. Rich test patterns. |
| dplyr | 10 | Data manipulation | High complexity, many function families (verbs, joins, grouping) |
| ggplot2 | 10 | Visualization | Complex layered API with many components |
| posterior | 5 | Bayesian | Mathematical operations (draws, diagnostics) - harder test targets |
| bayesplot | 5 | Bayesian | MCMC visualization - math + graphics combination |
| officer | 5 | Documents | Office document generation - distinct domain |
| flextable | 5 | Documents | Table formatting - distinct domain |
| farver | 5 | Colors | Color space manipulation - mathematical transformations |
| Others | 5 each | Various | Focused coverage of key functions |

**Key insight**: 80% of tasks are medium/hard difficulty, providing good optimization signal.

## Package Selection Criteria

Packages were selected based on:
1. **Download count** (rpkg.net top 50)
2. **Dependency count** (packages depended on by many others)
3. **GitHub stars** (community interest)
4. **Code quality** (well-tested, good documentation)
5. **Task diversity** (variety of functions to test)

---

## Tier 1: Core Tidyverse (High Priority)

### cli (30 tasks)
- **Downloads**: 2.5M/month
- **Dependents**: 15,607
- **Rationale**: Command-line interface helpers with diverse function patterns (color output, progress bars, lists, trees). Already had existing tasks.
- **Categories**: Output formatting, ANSI colors, progress indicators, semantic CLI elements
- **GitHub**: https://github.com/r-lib/cli

### testthat (10 tasks)
- **Downloads**: 722K/month
- **Dependents**: 462
- **Rationale**: The testing framework itself! Meta-testing provides unique challenges. 945 test patterns extracted.
- **Categories**: Expectations, mocking, snapshots, test organization, fixtures
- **GitHub**: https://github.com/r-lib/testthat

### dplyr (10 tasks)
- **Downloads**: 1.7M/month
- **Dependents**: 8,726
- **Rationale**: Most widely used data manipulation package. Complex API with verbs like `mutate()`, `filter()`, `summarize()`, joins, and grouping.
- **Categories**: Data transformation, grouping, joins, window functions
- **GitHub**: https://github.com/tidyverse/dplyr

### ggplot2 (10 tasks)
- **Downloads**: 3.1M/month
- **Dependents**: 8,231
- **Rationale**: Most starred R package. Complex layered graphics API with many edge cases around aesthetics, scales, and themes.
- **Categories**: Visualization, layer composition, scales, themes, facets
- **GitHub**: https://github.com/tidyverse/ggplot2

### tidyr (5 tasks)
- **Downloads**: 1.3M/month
- **Dependents**: 6,211
- **Rationale**: Data tidying operations - pivoting, nesting, separation. Good for testing data structure transformations.
- **Categories**: Pivoting, nesting, separation, missing value handling
- **GitHub**: https://github.com/tidyverse/tidyr

### stringr (5 tasks)
- **Downloads**: 1.3M/month
- **Dependents**: 8,900
- **Rationale**: String manipulation with consistent API. Clear function contracts make it good for testing patterns.
- **Categories**: Pattern matching, substitution, extraction, padding
- **GitHub**: https://github.com/tidyverse/stringr

### purrr (5 tasks)
- **Downloads**: 1.4M/month
- **Dependents**: 7,673
- **Rationale**: Functional programming tools - map, reduce, walk patterns. Higher-order functions are good test targets.
- **Categories**: Mapping, iteration, function composition, error handling
- **GitHub**: https://github.com/tidyverse/purrr

---

## Tier 2: Infrastructure (Medium Priority)

### rlang (5 tasks)
- **Downloads**: 2.9M/month
- **Dependents**: 15,969
- **Rationale**: Core tidyverse metaprogramming. Quosures, tidy evaluation, and error handling patterns.
- **Categories**: Quosures, tidy evaluation, error catching, env manipulation
- **GitHub**: https://github.com/r-lib/rlang

### vctrs (5 tasks)
- **Downloads**: 2.9M/month
- **Dependents**: 14,142
- **Rationale**: Vector helpers for custom types. Type coercion and casting patterns are well-defined.
- **Categories**: Type coercion, casting, vector operations, S3 dispatch
- **GitHub**: https://github.com/r-lib/vctrs

### tibble (5 tasks)
- **Downloads**: 2.7M/month
- **Dependents**: 9,949
- **Rationale**: Modern data frames with better printing and subsetting. Good for testing data frame operations.
- **Categories**: Construction, subsetting, printing, type preservation
- **GitHub**: https://github.com/tidyverse/tibble

### withr (5 tasks)
- **Downloads**: 2.1M/month
- **Dependents**: 12,926
- **Rationale**: Temporarily modified global state. Cleanup patterns are important for robust testing.
- **Categories**: Temporarily settings, deferred cleanup, options, connections
- **GitHub**: https://github.com/r-lib/withr

### glue (5 tasks)
- **Downloads**: 2.2M/month
- **Dependents**: 14,886
- **Rationale**: String interpolation. Simple API but useful for testing template patterns.
- **Categories**: Interpolation, escaping, custom delimiters
- **GitHub**: https://github.com/tidyverse/glue

---

## Tier 3: Bayesian Analysis

### posterior (5 tasks)
- **Downloads**: 1.5M/month
- **Dependents**: 89
- **Rationale**: Posterior distribution manipulation and draws processing. Mathematical operations involving log-probabilities and MCMC diagnostics provide harder test targets.
- **Categories**: Draws manipulation, diagnostics, summary statistics, convergence checking
- **GitHub**: https://github.com/stan-dev/posterior

### bayesplot (5 tasks)
- **Downloads**: 912K/month
- **Dependents**: 156
- **Rationale**: Visualization of MCMC diagnostics. Combines mathematical operations with graphics, providing unique testing challenges.
- **Categories**: Trace plots, density plots, diagnostic visualizations, posterior predictive checks
- **GitHub**: https://github.com/stan-dev/bayesplot

---

## Tier 4: Document Generation

### officer (5 tasks)
- **Downloads**: 421K/month
- **Dependents**: 312
- **Rationale**: Office document generation (Word, PowerPoint). Distinct domain with XML manipulation and document structure concerns.
- **Categories**: Document creation, formatting, tables, images, cross-references
- **GitHub**: https://github.com/davidgohel/officer

### flextable (5 tasks)
- **Downloads**: 289K/month
- **Dependents**: 198
- **Rationale**: Table formatting for documents and presentations. Complex API for styling, merging, and conditional formatting.
- **Categories**: Table styling, cell formatting, merging, borders, conditional formatting
- **GitHub**: https://github.com/davidgohel/flextable

---

## Tier 5: Color Manipulation

### farver (5 tasks)
- **Downloads**: 5.2M/month
- **Dependents**: 234
- **Rationale**: Color space manipulation and conversion. Mathematical transformations between color spaces (RGB, HSL, LAB, etc.) provide distinct testing challenges.
- **Categories**: Color conversion, color comparison, color mixing, palette operations
- **GitHub**: https://github.com/thomasp85/farver

---

## Future Expansion Candidates

Based on rpkg.net and R-universe data, these packages are candidates for future task generation:

### High Priority
| Package | Downloads | Rationale |
|---------|-----------|-----------|
| readr | 1.0M | Data import edge cases |
| lubridate | 835K | Date/time handling |
| httr | 928K | HTTP request patterns |
| jsonlite | 1.5M | JSON parsing |
| data.table | 1.2M | High-performance data operations |

### Medium Priority
| Package | Downloads | Rationale |
|---------|-----------|-----------|
| checkmate | 1.3M | Argument validation patterns |
| R6 | 2.3M | OOP reference semantics |
| knitr | 1.3M | Report generation |
| shiny | 652K | Web application framework |

### Specialized
| Package | Downloads | Rationale |
|---------|-----------|-----------|
| sf | 437K | Spatial data |
| lubridate | 835K | Date handling |
| reticulate | 278K | Python interop |
| igraph | 531K | Network analysis |
| RSQLite | 209K | Database operations |

### Stan Ecosystem (Specialized)
| Package | Downloads | Rationale |
|---------|-----------|-----------|
| cmdstanr | 928K | Stan interface, MCMC diagnostics |
| brms | 564K | Bayesian regression formulas, multilevel models |
| loo | 1.5M | Leave-one-out cross-validation, model comparison |
| rstan | 1.9M | Core Stan interface, compilation, sampling |
| rstanarm | N/A | Applied regression modeling |
| shinystan | N/A | Interactive model exploration |

**Special value**: These packages involve mathematical operations (log-probabilities, gradient computations, MCMC diagnostics) that provide distinct testing challenges compared to data manipulation packages.

---

## Data Sources

- **CRAN Downloads**: https://www.rpkg.net/top_downloaded_r_packages_from_cran_rstudio_r_package_by_age.php
- **R-universe**: https://r-universe.dev/packages
- **Top Packages**: https://r-packages.io/top-packages

---

## Generation Methodology

Tasks are generated using `scripts/generate_tasks.py` which:

1. Clones package source from GitHub
2. Analyzes function signatures and documentation
3. Extracts test patterns from existing tests
4. Generates instruction/context/reference_test triplets
5. Splits tasks into train/dev/held_out (60/20/20)

See SWE-bench methodology for similar approach: https://github.com/swe-bench/SWE-bench/blob/main/docs/guides/datasets.md
