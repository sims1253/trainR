# Kaggle Benchmark Tasks

Benchmark tasks mined from Kaggle competitions with measurable outcomes.

## Overview

- **64 tasks** from **13 competitions**
- **Focus**: Problem description + evaluation metric (precise grading)
- **Optional**: Reference solutions for LLM-judge fallback

## Task Structure

```json
{
  "task_id": "kaggle_titanic_exploring-survival-on-the-titanic",
  "source": {
    "competition": "titanic",
    "author": "mrisdal",
    "votes": 10972,
    "url": "https://www.kaggle.com/code/..."
  },
  "problem_statement": {
    "title": "Titanic - Machine Learning from Disaster",
    "description": "Predict survival on the Titanic...",
    "evaluation_metric": "Categorization Accuracy",
    "problem_type": "binary_classification",
    "instruction": "Implement a complete ML pipeline...",
    "context": "The Titanic competition is..."
  },
  "grading": {
    "metric": "Categorization Accuracy",
    "problem_type": "classification",
    "competition_slug": "titanic",
    "data_available": true,
    "grading_method": "metric"
  },
  "reference_solution": {
    "code": "# R code...",
    "format": "r_script",
    "key_techniques": ["tidyverse", "randomForest"]
  },
  "metadata": {...}
}
```

## Grading

### Primary: Metric-based (precise)

```
Task → Worker implements solution → Generate predictions → Score against ground truth
```

Requirements:
- Competition data (download via `kaggle competitions download <slug>`)
- Evaluation script (per competition)
- Ground truth labels (in competition data)

### Fallback: LLM-judge (fuzzy)

```
Task → Worker implements solution → LLM compares to reference → Quality score
```

Use when:
- Can't access competition data
- Task is exploratory (EDA, visualization)
- No single correct answer

Requires `reference_solution` field.

## Statistics

| Problem Type | Count |
|--------------|-------|
| Classification | 44 |
| Regression | 15 |
| Ranking | 4 |

| Metric | Count |
|--------|-------|
| AUC-ROC | 29 |
| Normalized Gini | 9 |
| RMSLE | 8 |
| Accuracy | 6 |

## Mining Script

```bash
# Mine from specific competition
uv run python scripts/mine_kaggle.py --competition titanic --max-kernels 10

# Mine multiple competitions
uv run python scripts/mine_kaggle.py --competitions "titanic,house-prices" --max-kernels 20

# Bulk collection
uv run python scripts/mine_kaggle.py --max-kernels 50 --skip-quality-check
```
