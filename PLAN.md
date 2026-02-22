# Posit-gskill MVP Plan

> Evolutionary optimization of Claude Skills for R package development using GEPA

---

## 1. Project Overview

### Goal
Automatically improve Posit's R-related Claude Skills using GEPA (Gradient-free Evolutionary Prompt_Algorithm) by:
1. Generating synthetic tasks from well-tested R packages
2. Running Claude Code CLI to solve these tasks
3. Reflecting on failures to evolve the skill prompts

### MVP Scope
| Aspect | Target (v1.0) | Revised Target (v1.1) |
|--------|---------------|----------------------|
| **Skill** | `testing-r-packages` (testthat 3+ patterns) | Same |
| **Source Packages** | r-lib/cli, r-lib/withr | cli only (expand if signal stable) |
| **Task Count** | 50-100 synthetic tasks | 10-20 curated tasks |
| **Generations** | 50 | 5 |
| **Population Size** | 20 | 6-8 |
| **Data Splits** | Train (60%) / Dev (20%) / Held-out (20%) | Same (boundary-aware) |
| **Agent** | Claude Code CLI | Same |
| **Optimizer** | GEPA | Same |

### Why This Matters
- Manual skill authoring is slow and inconsistent
- R package testing has nuanced patterns (fixtures, snapshots, mocking)
- GEPA provides data-driven prompt improvement
- Success here unlocks optimization of all Posit skills

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POSIT-GSKILL PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   Source Packages    │
                    │  (cli, withr, ...)   │
                    └──────────┬───────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TASK GENERATOR                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│  │ AST Parser  │→ │  Template   │→ │  LLM Refinement +       │   │
│  │ (R code)    │  │  Generator  │  │  Quality Gate           │   │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Task Dataset       │
                    │  ┌────┐┌────┐┌────┐  │
                    │  │Trai││Dev ││Test│  │
                    │  │60% ││20% ││20% │  │
                    │  └────┘└────┘└────┘  │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    GEPA LOOP    │  │                 │  │                 │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │                 │  │                 │
│ │  Skill      │ │  │                 │  │                 │
│ │  Candidate  │ │  │                 │  │                 │
│ └──────┬──────┘ │  │                 │  │                 │
│        │        │  │                 │  │                 │
│        ▼        │  │                 │  │                 │
│ ┌─────────────┐ │  │                 │  │                 │
│ │   Skill     │ │  │                 │  │                 │
│ │   Adapter   │ │  │                 │  │                 │
│ └──────┬──────┘ │  │                 │  │                 │
│        │        │  │                 │  │                 │
│        ▼        │  │                 │  │                 │
│ ┌─────────────────────────────────────────────────────────┐│
│ │              EVALUATION SANDBOX                         ││
│ │  ┌───────────────┐    ┌───────────────────────────┐    ││
│ │  │   Docker      │    │   Claude Code CLI         │    ││
│ │  │   Container   │───▶│   + R Environment         │    ││
│ │  │   (R + deps)  │    │   + Skill Prompt          │    ││
│ │  └───────────────┘    └───────────────────────────┘    ││
│ └─────────────────────────────────────────────────────────┘│
│        │        │  │                 │  │                 │
│        ▼        │  │                 │  │                 │
│ ┌─────────────┐ │  │                 │  │                 │
│ │  Reflection │ │  │                 │  │                 │
│ │  & Scoring  │ │  │                 │  │                 │
│ └──────┬──────┘ │  │                 │  │                 │
│        │        │  │                 │  │                 │
│        ▼        │  │                 │  │                 │
│ ┌─────────────┐ │  │                 │  │                 │
│ │  GEPA       │ │  │                 │  │                 │
│ │  Mutate     │ │  │                 │  │                 │
│ └─────────────┘ │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                    EVOLVED SKILL                              │
│         (Optimized testing-r-packages.md)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Task Generator (R-Specific)

**Purpose**: Generate high-quality testing tasks from R package source code.

**Approach**: Hybrid AST + Template + LLM refinement

```python
# task_generator.py

from dataclasses import dataclass
from typing import Optional
import subprocess
import json

@dataclass
class TestingTask:
    """A synthetic testing task for R packages."""
    task_id: str
    source_package: str
    source_file: str
    function_name: str           # NEW: For boundary-aware splitting
    difficulty: str  # easy, medium, hard
    
    # Task specification
    context: str           # Code context (function to test)
    instruction: str       # What test to write
    reference_test: str    # Ground truth test from package
    
    # Metadata
    test_type: str         # unit, snapshot, integration
    patterns: list[str]    # testthat patterns used
    dependencies: list[str]
    
    # Split assignment
    split: str             # train, dev, held_out


class RTaskGenerator:
    """Generate testing tasks from R packages."""
    
    def __init__(self, packages: list[str]):
        self.packages = packages
        self.templates = self._load_templates()
    
    def extract_test_patterns(self, package_path: str) -> list[dict]:
        """Parse existing tests to identify patterns."""
        # Use R's parsed_parse or tree-sitter for AST
        cmd = f"""
        library(parsed)
        tests <- list.files("{package_path}/tests/testthat", 
                           pattern = "^test-.*\\.R$", full.names = TRUE)
        patterns <- lapply(tests, function(f) {{
          # Extract: describe/it blocks, expect_* calls, fixtures
          list(
            file = f,
            describe_blocks = count_describe_blocks(f),
            expectations = extract_expectations(f),
            fixtures = detect_fixtures(f)
          )
        }})
        jsonlite::toJSON(patterns, auto_unbox = TRUE)
        """
        result = subprocess.run(
            ["Rscript", "-e", cmd], 
            capture_output=True, text=True
        )
        return json.loads(result.stdout)
    
    def generate_from_template(self, pattern: dict) -> TestingTask:
        """Create task from extracted pattern."""
        template = self.templates[pattern['type']]
        return TestingTask(
            task_id=self._generate_id(),
            source_package=pattern['package'],
            # ... fill from template + pattern
        )
    
    def llm_refine(self, task: TestingTask) -> TestingTask:
        """Use LLM to improve task clarity and quality."""
        prompt = f"""
        Refine this R testing task for clarity and completeness.
        
        Original instruction: {task.instruction}
        Context: {task.context}
        
        Return JSON with improved 'instruction' and 'hints' fields.
        """
        # Call Claude API
        refined = call_claude(prompt)
        task.instruction = refined['instruction']
        return task
    
    def quality_gate(self, task: TestingTask) -> bool:
        """Validate task meets quality standards."""
        checks = [
            len(task.instruction) >= 20,
            len(task.reference_test) >= 10,
            task.test_type in ['unit', 'snapshot', 'integration'],
            task.difficulty in ['easy', 'medium', 'hard'],
            self._verify_reference_runs(task),  # Run the reference test
        ]
        return all(checks)
```

**Template Categories**:

| Template Type | Description | Example |
|---------------|-------------|---------|
| `expect_function` | Test a pure function | Test `str_trim()` with various inputs |
| `expect_snapshot` | Snapshot testing | Test `cli_alert()` output format |
| `with_fixture` | Test with setup/teardown | Test file operations with temp dirs |
| `mock_bindings` | Mock external dependencies | Mock HTTP calls in API tests |
| `describe_it` | BDD-style nested tests | Organize related test cases |

### 3.2 Evaluation Sandbox (Docker + R)

**Purpose**: Isolated environment to run Claude Code CLI and evaluate test solutions.

```dockerfile
# Dockerfile.evaluation
# Pin base image digest for reproducibility
FROM rocker/tidyverse:4.3@sha256:abc123...  # TODO: Replace with actual digest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI with version pinning and checksum verification
# Replace curl | sh with version-pinned install
ARG CLAUDE_CLI_VERSION=1.0.0
RUN curl -fsSL "https://github.com/anthropics/claude-code/releases/download/v${CLAUDE_CLI_VERSION}/claude-code-linux-x64.tar.gz" -o /tmp/claude.tar.gz && \
    echo "expected_sha256_hash  /tmp/claude.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/claude.tar.gz -C /usr/local/bin && \
    rm /tmp/claude.tar.gz

# Install R testing infrastructure with renv for reproducibility
# Use renv lockfile for version-pinned packages
COPY renv.lock /tmp/renv.lock
RUN Rscript -e "install.packages('renv'); renv::restore(lockfile = '/tmp/renv.lock')"

# Pre-install target packages for faster evaluation (pinned versions)
RUN Rscript -e "pak::pkg_install(c('cli@3.6.0', 'withr@3.0.0'))"

WORKDIR /workspace
```

```python
# evaluation_sandbox.py

import docker
import tempfile
import json
from pathlib import Path

class EvaluationSandbox:
    """Docker-based evaluation environment."""
    
    def __init__(self, image: str = "posit-gskill-eval:latest"):
        self.client = docker.from_env()
        self.image = image
    
    def run_evaluation(
        self, 
        task: TestingTask, 
        skill_prompt: str,
        timeout: int = 300
    ) -> EvaluationResult:
        """Run Claude Code CLI to solve a testing task."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare workspace
            workspace = Path(tmpdir)
            self._setup_workspace(workspace, task, skill_prompt)
            
            # Run container
            container = self.client.containers.run(
                self.image,
                volumes={str(workspace): {'bind': '/workspace', 'mode': 'rw'}},
                command=self._build_command(task),
                timeout=timeout,
                detach=False,
                stdout=True,
                stderr=True
            )
            
            # Collect results
            return self._parse_results(workspace, task, container)
    
    def _setup_workspace(self, workspace: Path, task: TestingTask, skill: str):
        """Prepare the workspace with task files."""
        # Write skill prompt
        (workspace / "SKILL.md").write_text(skill)
        
        # Write task specification
        (workspace / "TASK.json").write_text(json.dumps({
            "task_id": task.task_id,
            "instruction": task.instruction,
            "context": task.context,
            "output_path": f"tests/testthat/test-{task.task_id}.R"
        }))
        
        # Setup minimal R package structure
        (workspace / "R").mkdir()
        (workspace / "tests" / "testthat").mkdir(parents=True)
        
        # Write the function to test
        (workspace / "R" / "target.R").write_text(task.context)
    
    def _build_command(self, task: TestingTask) -> list[str]:
        """Build Claude Code CLI command."""
        return [
            "claude", "code",
            "--skill", "/workspace/SKILL.md",
            "--task", "/workspace/TASK.json",
            "--non-interactive",
            "--output-format", "json"
        ]
    
    def _parse_results(self, workspace: Path, task: TestingTask, output: bytes) -> EvaluationResult:
        """Parse evaluation results."""
        # Check if test file was created
        test_file = workspace / "tests" / "testthat" / f"test-{task.task_id}.R"
        
        # Run the generated test
        test_result = self._run_generated_test(test_file)
        
        return EvaluationResult(
            task_id=task.task_id,
            success=test_result['passed'],
            generated_code=test_file.read_text() if test_file.exists() else None,
            test_output=test_result['output'],
            error_message=test_result.get('error'),
            execution_time=test_result['time']
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a task."""
    task_id: str
    success: bool
    generated_code: Optional[str]
    test_output: str
    error_message: Optional[str]
    execution_time: float
```

### 3.3 Skill Adapter (GEPA Interface)

**Purpose**: Bridge between GEPA optimizer and Claude Code skill format.

```python
# skill_adapter.py

from typing import Protocol
import yaml

class SkillAdapter:
    """Adapt skill format for GEPA optimization."""
    
    def __init__(self, base_skill_path: str):
        self.base_skill = self._load_skill(base_skill_path)
        self.components = self._extract_components(self.base_skill)
    
    def _load_skill(self, path: str) -> dict:
        """Load skill from markdown with YAML frontmatter."""
        content = Path(path).read_text()
        # Parse SKILL.md format
        return parse_skill_markdown(content)
    
    def _extract_components(self, skill: dict) -> dict:
        """Extract mutable components for optimization."""
        return {
            'preamble': skill.get('preamble', ''),
            'instructions': skill.get('instructions', []),
            'examples': skill.get('examples', []),
            'anti_patterns': skill.get('anti_patterns', []),
        }
    
    def to_prompt(self, components: dict) -> str:
        """Convert components back to full skill prompt."""
        return f"""
# R Package Testing Skill

{components['preamble']}

## Instructions
{self._format_list(components['instructions'])}

## Examples
{self._format_examples(components['examples'])}

## Anti-Patterns to Avoid
{self._format_list(components['anti_patterns'])}
"""
    
    def to_gepa_individual(self, components: dict) -> list[str]:
        """Convert to GEPA's genome representation."""
        # GEPA works with lists of text chunks
        return [
            components['preamble'],
            *components['instructions'],
            *[ex['code'] for ex in components['examples']],
        ]
    
    def from_gepa_individual(self, genome: list[str]) -> dict:
        """Convert GEPA genome back to skill components."""
        # Map genome positions back to components
        return {
            'preamble': genome[0],
            'instructions': genome[1:-len(self.components['examples'])],
            'examples': self._reconstruct_examples(genome[-len(self.components['examples']):]),
            'anti_patterns': self.components['anti_patterns'],  # Keep static for MVP
        }


# GEPA Interface
class GEPAInterface:
    """Interface for GEPA optimizer."""
    
    def evaluate_population(
        self, 
        population: list[list[str]], 
        tasks: list[TestingTask],
        sandbox: EvaluationSandbox
    ) -> list[float]:
        """Evaluate each skill candidate on the task set."""
        scores = []
        
        for genome in population:
            skill = self.adapter.from_gepa_individual(genome)
            prompt = self.adapter.to_prompt(skill)
            
            task_scores = []
            for task in tasks:
                result = sandbox.run_evaluation(task, prompt)
                task_scores.append(1.0 if result.success else 0.0)
            
            scores.append(sum(task_scores) / len(task_scores))
        
        return scores
```

### 3.4 Optimization Loop

**Purpose**: Orchestrate the GEPA optimization process.

```python
# optimization_loop.py

from gepa import GEPA, GEPAConfig
from typing import Callable

class SkillOptimizer:
    """Main optimization orchestrator."""
    
    def __init__(
        self,
        adapter: SkillAdapter,
        sandbox: EvaluationSandbox,
        tasks: dict[str, list[TestingTask]],  # train, dev, held_out
        config: OptimizationConfig
    ):
        self.adapter = adapter
        self.sandbox = sandbox
        self.tasks = tasks
        self.config = config
        
        self.history: list[OptimizationStep] = []
        self.best_skill: Optional[str] = None
        self.best_score: float = 0.0
    
    def run(self) -> OptimizationResult:
        """Run the full optimization loop."""
        
        # Initialize GEPA
        gepa = GEPA(GEPAConfig(
            population_size=self.config.population_size,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            elite_size=self.config.elite_size,
        ))
        
        # Initialize population with base skill variations
        base_genome = self.adapter.to_gepa_individual(self.adapter.components)
        population = self._initialize_population(base_genome)
        
        for generation in range(self.config.max_generations):
            print(f"Generation {generation + 1}/{self.config.max_generations}")
            
            # Evaluate on training set
            train_scores = self._evaluate_population(population, 'train')
            
            # Selection & evolution
            population = gepa.evolve(population, train_scores)
            
            # Evaluate on dev set for best candidates
            elite_indices = sorted(range(len(train_scores)), 
                                   key=lambda i: train_scores[i], 
                                   reverse=True)[:self.config.elite_size]
            
            dev_scores = self._evaluate_population(
                [population[i] for i in elite_indices], 
                'dev'
            )
            
            # Track best - use max(dev_scores) not dev_scores[0]
            best_dev_score = max(dev_scores)
            best_idx = elite_indices[dev_scores.index(best_dev_score)]
            if best_dev_score > self.best_score:
                self.best_score = best_dev_score
                self.best_skill = self.adapter.to_prompt(
                    self.adapter.from_gepa_individual(population[best_idx])
                )
            
            # Record history
            self.history.append(OptimizationStep(
                generation=generation,
                train_scores=train_scores,
                dev_scores=dev_scores,
                best_skill=self.best_skill,
            ))
            
            # Early stopping check
            if self._check_convergence():
                break
        
        # Final evaluation on held-out set
        final_score = self._evaluate_skill(self.best_skill, 'held_out')
        
        return OptimizationResult(
            best_skill=self.best_skill,
            final_score=final_score,
            history=self.history,
        )
    
    def _evaluate_population(
        self, 
        population: list, 
        split: str
    ) -> list[float]:
        """Evaluate all candidates on a task split."""
        tasks = self.tasks[split]
        scores = []
        
        for genome in population:
            skill = self.adapter.from_gepa_individual(genome)
            prompt = self.adapter.to_prompt(skill)
            score = self._evaluate_skill(prompt, split)
            scores.append(score)
        
        return scores
    
    def _evaluate_skill(self, skill: str, split: str) -> float:
        """Evaluate a single skill on a split."""
        tasks = self.tasks[split]
        successes = 0
        
        for task in tasks:
            result = self.sandbox.run_evaluation(task, skill)
            if result.success:
                successes += 1
        
        return successes / len(tasks)
```

---

## 4. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                           │
└─────────────────────────────────────────────────────────────────────┘

1. TASK GENERATION FLOW
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ R pkg   │────▶│ AST     │────▶│Template │────▶│  LLM    │
   │ source  │     │ Parser  │     │ Fill    │     │ Refine  │
   └─────────┘     └─────────┘     └─────────┘     └────┬────┘
                                                        │
   ┌─────────┐     ┌─────────┐     ┌─────────┐          │
   │ Task    │◀────│ Quality │◀────│ Valid   │◀─────────┘
   │ Dataset │     │ Gate    │     │ Check   │
   └─────────┘     └─────────┘     └─────────┘

2. EVALUATION FLOW
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Task    │────▶│ Docker  │────▶│ Claude  │────▶│ Test    │
   │ + Skill │     │ Sandbox │     │ Code    │     │ Runner  │
   └─────────┘     └─────────┘     └─────────┘     └────┬────┘
                                                        │
                        ┌─────────┐     ┌─────────┐      │
                        │ Result  │◀────│ Score   │◀─────┘
                        │ Record  │     │ Compute │
                        └─────────┘     └─────────┘

3. OPTIMIZATION FLOW
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Initial │────▶│ GEPA    │────▶│ Mutate/ │────▶│Evaluate │
   │ Skill   │     │ Init    │     │ Cross   │     │ Batch   │
   └─────────┘     └─────────┘     └─────────┘     └────┬────┘
                                                        │
   ┌─────────┐     ┌─────────┐     ┌─────────┐          │
   │ Best    │◀────│ Select  │◀────│ Rank    │◀─────────┘
   │ Skill   │     │ Elite   │     │ Scores  │
   └─────────┘     └─────────┘     └─────────┘
        │
        └──────────────▶ Repeat until convergence
```

---

## 5. Task Specification Contract

Each task in the dataset follows this JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["task_id", "instruction", "context", "reference", "metadata"],
  "properties": {
    "task_id": {
      "type": "string",
      "pattern": "^task-[a-z0-9]{8}$",
      "description": "Unique identifier"
    },
    "instruction": {
      "type": "string",
      "minLength": 20,
      "description": "What test to write"
    },
    "context": {
      "type": "object",
      "properties": {
        "source_file": {"type": "string"},
        "function_name": {"type": "string"},
        "function_code": {"type": "string"},
        "package_dependencies": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },
    "reference": {
      "type": "object",
      "properties": {
        "test_code": {"type": "string"},
        "test_file": {"type": "string"},
        "patterns_used": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": [
              "expect_equal", "expect_snapshot", "expect_error",
              "describe_it", "with_fixture", "local_mocked_bindings",
              "test_that", "skip_if", "setup_teardown"
            ]
          }
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "difficulty": {
          "type": "string",
          "enum": ["easy", "medium", "hard"]
        },
        "source_package": {"type": "string"},
        "split": {
          "type": "string",
          "enum": ["train", "dev", "held_out"]
        },
        "created_at": {"type": "string", "format": "date-time"},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "constraints": {
      "type": "object",
      "properties": {
        "time_limit_seconds": {"type": "integer", "default": 300},
        "forbidden_patterns": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Regex patterns that should NOT appear in solution"
        },
        "required_patterns": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Regex patterns that MUST appear in solution"
        }
      }
    }
  }
}
```

**Example Task**:

```json
{
  "task_id": "task-a3f2c1d8",
  "instruction": "Write a testthat test for the `str_trim()` function that handles NA values gracefully. The function should return NA when given NA input, and trim whitespace from character vectors.",
  "context": {
    "source_file": "R/str-trim.R",
    "function_name": "str_trim",
    "function_code": "str_trim <- function(string, side = c('both', 'left', 'right')) {\n  side <- match.arg(side)\n  if (is.na(string)) return(NA_character_)\n  # ... implementation\n}",
    "package_dependencies": ["stringr"]
  },
  "reference": {
    "test_code": "test_that('str_trim handles NA values', {\n  expect_equal(str_trim(NA), NA_character_)\n  expect_equal(str_trim(c('  hi', NA, 'bye  ')), c('hi', NA, 'bye'))\n})",
    "test_file": "tests/testthat/test-str-trim.R",
    "patterns_used": ["test_that", "expect_equal"]
  },
  "metadata": {
    "difficulty": "easy",
    "source_package": "stringr",
    "split": "train",
    "created_at": "2026-02-20T00:00:00Z",
    "quality_score": 0.95
  },
  "constraints": {
    "time_limit_seconds": 120,
    "required_patterns": ["test_that\\(", "expect_"],
    "forbidden_patterns": ["skip\\("]
  }
}
```

---

## 6. Reflection Record Schema

After each evaluation, we record structured reflection for GEPA to use:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["evaluation_id", "task_id", "outcome", "failure_analysis"],
  "properties": {
    "evaluation_id": {
      "type": "string",
      "description": "Unique evaluation run identifier"
    },
    "task_id": {
      "type": "string",
      "description": "Which task was evaluated"
    },
    "skill_version": {
      "type": "string",
      "description": "Git hash or version of skill used"
    },
    "outcome": {
      "type": "object",
      "properties": {
        "success": {"type": "boolean"},
        "score": {"type": "number"},
        "execution_time_seconds": {"type": "number"}
      }
    },
    "generated_solution": {
      "type": "object",
      "properties": {
        "code": {"type": "string"},
        "diff_from_reference": {"type": "string"}
      }
    },
    "failure_analysis": {
      "type": "object",
      "description": "Populated when outcome.success is false",
      "properties": {
        "failure_category": {
          "type": "string",
          "enum": [
            "SYNTAX_ERROR",
            "TEST_FAILURE", 
            "TIMEOUT",
            "MISSING_IMPORT",
            "WRONG_ASSERTION",
            "INCOMPLETE_SOLUTION",
            "OVERLY_COMPLEX",
            "WRONG_FIXTURE_USAGE",
            "SNAPSHOT_MISMATCH"
          ]
        },
        "error_message": {"type": "string"},
        "error_location": {
          "type": "object",
          "properties": {
            "line": {"type": "integer"},
            "column": {"type": "integer"},
            "context": {"type": "string"}
          }
        },
        "root_cause_hypothesis": {
          "type": "string",
          "description": "LLM-generated hypothesis about why the skill failed"
        },
        "skill_improvement_suggestion": {
          "type": "string",
          "description": "Specific suggestion for skill improvement"
        }
      }
    },
    "test_output": {
      "type": "object",
      "properties": {
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "testthat_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "test": {"type": "string"},
              "result": {"type": "string", "enum": ["PASS", "FAIL", "SKIP", "ERROR"]},
              "message": {"type": "string"}
            }
          }
        }
      }
    }
  }
}
```

**Failure Taxonomy**:

| Category | Description | GEPA Signal |
|----------|-------------|-------------|
| `SYNTAX_ERROR` | Generated code has R syntax errors | Strengthen code examples |
| `TEST_FAILURE` | Tests run but assertions fail | Improve assertion guidance |
| `TIMEOUT` | Execution exceeded time limit | Simplify approach guidance |
| `MISSING_IMPORT` | Required library not loaded | Emphasize library() calls |
| `WRONG_ASSERTION` | Used wrong expect_* function | Clarify expectation selection |
| `INCOMPLETE_SOLUTION` | Missing test cases | Encourage completeness |
| `OVERLY_COMPLEX` | Solution too convoluted | Guide toward simplicity |
| `WRONG_FIXTURE_USAGE` | Misused test fixtures | Improve fixture examples |
| `SNAPSHOT_MISMATCH` | Snapshot test failed | Improve snapshot guidance |

---

## 7. Quality Gates & Guards

### 7.1 Task Quality Gates

```python
# quality_gates.py

class TaskQualityGate:
    """Validate task quality before inclusion in dataset."""
    
    def validate(self, task: TestingTask) -> tuple[bool, list[str]]:
        """Run all quality checks, return (passed, errors)."""
        errors = []
        
        # Structural checks
        if len(task.instruction) < 20:
            errors.append("Instruction too short (min 20 chars)")
        
        if not task.context or len(task.context) < 10:
            errors.append("Context missing or too short")
        
        if not task.reference_test:
            errors.append("Reference test required")
        
        # Semantic checks
        if not self._instruction_is_actionable(task.instruction):
            errors.append("Instruction must be actionable (start with verb)")
        
        if not self._context_is_valid_r(task.context):
            errors.append("Context must be valid R code")
        
        # Test validity
        if not self._reference_test_passes(task):
            errors.append("Reference test must pass")
        
        # Difficulty calibration
        expected_patterns = {
            'easy': (1, 2),      # 1-2 patterns for easy
            'medium': (3, 4),    # 3-4 patterns for medium
            'hard': (5, 10)      # 5+ patterns for hard
        }
        actual_patterns = len(task.patterns)
        min_patterns, max_patterns = expected_patterns.get(task.difficulty, (1, 10))
        if actual_patterns < min_patterns:
        if actual_patterns < min_patterns or actual_patterns > max_patterns:
            errors.append(f"Pattern count {actual_patterns} outside expected range [{min_patterns}, {max_patterns}] for {task.difficulty} difficulty")
        
        return len(errors) == 0, errors
    
    def _instruction_is_actionable(self, instruction: str) -> bool:
        action_verbs = ['write', 'create', 'implement', 'add', 'modify', 'refactor']
        first_word = instruction.lower().split()[0]
        return first_word in action_verbs
    
    def _reference_test_passes(self, task: TestingTask) -> bool:
        """Actually run the reference test in isolation."""
        # ... implementation
        pass
```

### 7.2 Regression Guards

```python
# regression_guards.py

class RegressionGuard:
    """Prevent regression during skill evolution."""
    
    def __init__(self, baseline_skill: str, canonical_tasks: list[TestingTask]):
        self.baseline = baseline_skill
        self.canonical_tasks = canonical_tasks
        self.baseline_scores = self._compute_baseline_scores()
    
    def check(self, candidate_skill: str) -> tuple[bool, str]:
        """Check if candidate maintains baseline performance."""
        candidate_scores = self._evaluate_on_canonical(candidate_skill)
        
        # Allow 5% regression on individual tasks
        for task_id, (base_score, cand_score) in self._zip_scores(
            self.baseline_scores, candidate_scores
        ).items():
            if cand_score < base_score * 0.95:
                return False, f"Regression on {task_id}: {cand_score} < {base_score * 0.95}"
        
        # No more than 10% tasks with any regression
        regressed = sum(1 for b, c in zip(self.baseline_scores, candidate_scores) 
                        if c < b)
        if regressed > len(self.canonical_tasks) * 0.1:
            return False, f"Too many regressed tasks: {regressed}"
        
        return True, "Passed regression checks"
```

### 7.3 Mutation Constraints

```python
# mutation_constraints.py

class MutationConstraints:
    """Constrain how GEPA can mutate skills."""
    
    PRESERVED_SECTIONS = [
        "## Required Packages",
        "## Environment Setup",
    ]
    
    MAX_MUTATION_RATIO = 0.3  # No more than 30% of content changed
    
    FORBIDDEN_ADDITIONS = [
        "skip_if_not_installed",  # Don't add lazy skips
        "@not_tested",            # Don't add exclusion markers
    ]
    
    def validate_mutation(
        self, 
        original: str, 
        mutated: str
    ) -> tuple[bool, str]:
        """Validate a mutation is acceptable."""
        
        # Check preserved sections
        for section in self.PRESERVED_SECTIONS:
            if section in original and section not in mutated:
                return False, f"Cannot remove preserved section: {section}"
        
        # Check mutation ratio
        diff_ratio = self._compute_diff_ratio(original, mutated)
        if diff_ratio > self.MAX_MUTATION_RATIO:
            return False, f"Mutation too aggressive: {diff_ratio:.1%} changed"
        
        # Check forbidden additions
        for forbidden in self.FORBIDDEN_ADDITIONS:
            if forbidden not in original and forbidden in mutated:
                return False, f"Cannot add forbidden pattern: {forbidden}"
        
        return True, "Valid mutation"
```

---

## 8. MVP Implementation Phases

### Phase 0: Setup & Infrastructure (Week 1)

**Goals**: Establish development environment and basic tooling

**Tasks**:
- [ ] Initialize project repository structure
- [ ] Set up Python environment with GEPA dependencies
- [ ] Set up R environment with testthat 3+
- [ ] Clone target packages (cli, withr) locally
- [ ] Build Docker evaluation image
- [ ] Verify Claude Code CLI works in Docker

**Deliverables**:
```
posit-gskill/
├── pyproject.toml          # Python dependencies
├── requirements.R          # R dependencies
├── Dockerfile.evaluation   # Evaluation sandbox
├── Makefile                # Common commands
└── .env.example            # Configuration template
```

**Verification**:
```bash
# Docker builds successfully
docker build -t posit-gskill-eval:latest -f Dockerfile.evaluation .

# Claude Code CLI accessible in container
docker run posit-gskill-eval claude --version

# R packages install correctly
docker run posit-gskill-eval Rscript -e "library(testthat); library(cli); library(withr)"
```

---

### Phase 1: Task Generation MVP (Weeks 2-3)

**Goals**: Generate 50-100 high-quality testing tasks

**Tasks**:
- [ ] Implement R AST parser for test file analysis
- [ ] Create task templates for each test pattern type
- [ ] Build semi-automated generation pipeline
- [ ] Implement quality gates
- [ ] Generate initial task pool
- [ ] Manual review and refinement
- [ ] Split into train/dev/held_out sets

**Deliverables**:
```
posit-gskill/
├── task_generator/
│   ├── __init__.py
│   ├── ast_parser.py       # R code analysis
│   ├── templates.py        # Task templates
│   ├── llm_refine.py       # LLM refinement
│   └── quality_gate.py     # Validation
├── tasks/
│   ├── train/
│   │   ├── task-001.json
│   │   └── ...
│   ├── dev/
│   │   └── ...
│   └── held_out/
│       └── ...
└── scripts/
    └── generate_tasks.py   # Main generation script
```

**Target Distribution**:

| Source Package | Easy | Medium | Hard | Total |
|----------------|------|--------|------|-------|
| cli | 15 | 12 | 8 | 35 |
| withr | 12 | 10 | 8 | 30 |
| **Total** | 27 | 22 | 16 | **65** |

**Verification**:
```bash
# Generate tasks
uv run python scripts/generate_tasks.py --packages cli,withr --output tasks/

# Validate all tasks pass quality gates
uv run python scripts/validate_tasks.py --tasks tasks/

# Expected output:
# ✓ 65 tasks generated
# ✓ 60/65 pass quality gate (92%)
# ✓ Train: 39, Dev: 13, Held-out: 13
```

---

### Phase 2: Evaluation Sandbox (Weeks 3-4)

**Goals**: Working evaluation pipeline with Claude Code CLI

**Tasks**:
- [ ] Implement sandbox container orchestration
- [ ] Build task workspace setup logic
- [ ] Integrate Claude Code CLI execution
- [ ] Implement test runner and result parsing
- [ ] Build reflection generator
- [ ] Add timeout and resource limits

**Deliverables**:
```
posit-gskill/
├── evaluation/
│   ├── __init__.py
│   ├── sandbox.py          # Docker orchestration
│   ├── workspace.py        # Task setup
│   ├── runner.py           # Claude Code CLI runner
│   ├── test_executor.py    # Run generated tests
│   └── reflector.py        # Generate reflection records
├── docker/
│   ├── Dockerfile.evaluation
│   └── entrypoint.sh
└── scripts/
    └── run_evaluation.py   # Single task evaluation
```

**Verification**:
```bash
# Test single task evaluation
uv run python scripts/run_evaluation.py \
    --task tasks/train/task-001.json \
    --skill skills/testing-r-packages.md \
    --verbose

# Expected output:
# Setting up workspace...
# Running Claude Code CLI...
# Generated test file: test-001.R
# Running testthat...
# ✓ 3/3 tests passed
# Score: 1.0
```

---

### Phase 3: GEPA Integration (Weeks 5-6)

**Goals**: Connect all components into optimization loop

**Tasks**:
- [ ] Implement skill adapter for GEPA format
- [ ] Configure GEPA hyperparameters
- [ ] Build batch evaluation pipeline
- [ ] Implement optimization loop
- [ ] Add checkpointing and recovery
- [ ] Build logging and monitoring

**Deliverables**:
```
posit-gskill/
├── optimization/
│   ├── __init__.py
│   ├── adapter.py          # Skill ↔ GEPA conversion
│   ├── evaluator.py        # Batch evaluation
│   ├── optimizer.py        # Main loop
│   └── checkpoint.py       # State persistence
├── configs/
│   └── gepa_config.yaml    # GEPA settings
└── scripts/
    └── run_optimization.py # Main entry point
```

**GEPA Configuration**:
```yaml
# configs/gepa_config.yaml
gepa:
  population_size: 20
  max_generations: 50
  mutation_rate: 0.15
  crossover_rate: 0.7
  elite_size: 4

evaluation:
  batch_size: 10
  timeout_per_task: 300
  parallel_workers: 4

constraints:
  max_mutation_ratio: 0.3
  preserve_sections:
    - "## Required Packages"
    - "## Environment Setup"
```

**Verification**:
```bash
# Run short optimization test
uv run python scripts/run_optimization.py \
    --config configs/gepa_config.yaml \
    --tasks tasks/ \
    --generations 5 \
    --test

# Expected output:
# Generation 1: Best score = 0.45 (train)
# Generation 2: Best score = 0.52 (train)
# ...
# Generation 5: Best score = 0.61 (train)
# Dev score: 0.58
# Checkpoint saved to: checkpoints/run-001/
```

---

### Phase 4: Optimization Runs (Weeks 7-8)

**Goals**: Execute full optimization and analyze results

**Tasks**:
- [ ] Run baseline evaluation with original skill
- [ ] Execute full optimization (50 generations)
- [ ] Run ablation studies
- [ ] Evaluate on held-out set
- [ ] Analyze failure modes
- [ ] Document findings

**Deliverables**:
```
posit-gskill/
├── results/
│   ├── baseline/
│   │   └── evaluation_report.json
│   ├── optimized/
│   │   ├── best_skill.md
│   │   └── evaluation_report.json
│   └── comparison/
│       └── improvement_analysis.md
├── analysis/
│   ├── failure_modes.md
│   ├── skill_diff.md
│   └── recommendations.md
└── skills/
    ├── testing-r-packages-orig.md   # Baseline
    └── testing-r-packages-opt.md    # Optimized
```

**Verification**:
```bash
# Baseline evaluation
uv run python scripts/evaluate_skill.py \
    --skill skills/testing-r-packages-orig.md \
    --tasks tasks/held_out/ \
    --output results/baseline/

# Optimized evaluation  
uv run python scripts/evaluate_skill.py \
    --skill results/optimized/best_skill.md \
    --tasks tasks/held_out/ \
    --output results/optimized/

# Compare
uv run python scripts/compare_results.py \
    --baseline results/baseline/ \
    --optimized results/optimized/
```

---

## Phase 4 Progress (Updated 2026-02-22)

### Completed Tasks

1. **DockerPiRunner Migration** ✅
   - Updated `evaluate_batch.py` to use `DockerPiRunner` instead of `DockerTestRunner`
   - Updated `evaluation/sandbox.py` to use `DockerPiRunner` and `DockerPiRunnerConfig`
   - Updated `optimization/adapter.py` to use `DockerPiRunnerConfig`
   - Added `--max-tasks` CLI argument for testing
   - Changed API key check from `Z_AI_API_KEY` to `OPENROUTER_API_KEY`

2. **Baseline Configuration** ✅
   - Created 8 baseline config files for 4 free models:
     - `openrouter/stepfun/step-3.5-flash:free`
     - `openrouter/openai/gpt-oss-120b:free`
     - `openrouter/nvidia/nemotron-3-nano-30b-a3b:free`
     - `opencode/minimax-m2.5-free`
   - Each model has no-skill and skill config variants
   - Configs use `train` split (all 18 tasks)
   - Set to 1 worker to avoid OOM issues

3. **Makefile Updates** ✅
   - Added `baseline-{model}-no-skill` targets for all 4 models
   - Added `baseline-{model}-skill` targets for all 4 models
   - Added `baseline-all-free` target to run all 8 baselines
   - Added `compare-free-models` target for comparison table

4. **Optimization Test** ✅
   - Verified GEPA optimization works with DockerPiRunner
   - Quick test (3 metric calls) completed successfully
   - Base program score: 66.67% on 6 validation tasks

### In Progress

1. **Baseline Runs** 🔄
   - Started all 8 baselines running in background (script: `scripts/run_all_baselines.sh`)
   - Estimated time: ~3 min/task × 18 tasks × 8 baselines = ~7 hours total
   - Logs: `logs/baselines/{model}-{skill}.log`
   - Results: `results/baselines/eval_*.json`

### Decisions Made

1. **Free Models**: Using OpenRouter free tier models for baselines due to cost constraints
   - Trade-off: Slower inference vs cost savings
   - May need to upgrade to paid models for faster iteration if signal is promising

2. **Worker Count**: Reduced to 1 worker per baseline run
   - Reason: Multiple Docker containers cause OOM issues
   - Each container runs the `pi` CLI with model inference

3. **Task Split**: Using `train` split for all tasks
   - Current dataset only has 18 tasks in `train` split
   - No dev/held_out splits currently defined
   - **DECISION NEEDED**: Define proper split strategy before optimization

### Next Steps

1. Wait for baseline runs to complete
2. Run GEPA optimization with free models (`make optimize-fresh OPT_MAX_CALLS=30`)
3. Compare optimized skill vs baselines across all 4 models
4. If signal is promising:
   - Define proper train/dev/held_out splits
   - Consider paid models for faster iteration
   - Expand task set from other packages

### Phase 4 Results (Completed 2026-02-22)

#### Baseline Comparison (18 tasks)

| Model | No-Skill | With Skill | Delta |
|-------|----------|------------|-------|
| StepFun Step-3.5-Flash | 50.0% | 50.0% | 0pp |
| OpenAI GPT-OSS-120B | 33.3% | 83.3% | **+50.0pp** |
| NVIDIA Nemotron-3-Nano | 83.3% | 72.2% | -11.1pp |
| Minimax M2.5 | 44.4% | **100.0%** | **+55.6pp** |

#### Key Findings

1. **Minimax M2.5** achieves 100% pass rate with skill guidance
   - Best performing model with the hand-crafted skill
   - 55.6pp improvement from no-skill baseline
   - **RECOMMENDATION**: Use Minimax M2.5 for future optimization

2. **OpenAI GPT-OSS-120B** shows strong skill benefit (+50pp)
   - No-skill baseline is weakest (33.3%)
   - With skill: 83.3% (strong improvement)
   - Good candidate for optimization

3. **NVIDIA Nemotron-3-Nano** performs worse with skill (-11.1pp)
   - High no-skill baseline (83.3%)
   - Skill may be confusing this model
   - Consider different skill format for this model

4. **StepFun Step-3.5-Flash** shows no skill benefit
   - Baseline and skill performance identical (~50%)
   - Model may not follow instructions well

#### GEPA Optimization Results

- **Best Score**: 33.3% (on OpenAI GPT-OSS-120B validation)
- **Total Metric Calls**: 39
- **Result**: Optimization did NOT improve over hand-crafted skill
  - Hand-crafted skill: 83.3% on OpenAI
  - Optimized skill: 33.3% on OpenAI
- **Analysis**: 
  - Free model rate limiting may have affected optimization quality
  - Need more optimization budget with better models
  - Original skill is already well-tuned

#### Conclusions

1. **Hand-crafted skill works well** for Minimax and OpenAI models
2. **GEPA optimization needs refinement**:
   - Use paid models for reflection (not free tier)
   - Increase optimization budget (more than 30 calls)
   - Consider multi-model optimization
3. **Model selection matters**:
   - Different models respond differently to skill guidance
   - Minimax M2.5 is best candidate for production use

#### Files Generated

- Baseline configs: `configs/baseline_*_{stepfun,openai,nvidia,minimax}.yaml`
- Baseline results: `results/baselines/eval_*/`
- Optimization run: `results/optimization/free_model_run/run_20260222_033958/`
- Comparison script: `scripts/compare_results.py`
- Runner script: `scripts/run_all_baselines.sh`

---

## Phase 5: Dataset Expansion (2026-02-22)

### Goals

1. **Expand task diversity** - Add tasks from multiple R packages
2. **Improve task difficulty** - Create more challenging testing scenarios
3. **Define proper splits** - Establish train/dev/held_out split strategy
4. **Document methodology** - Create reproducible task generation pipeline

### Target Packages for Task Generation

Based on CRAN download statistics and package quality:

#### Tier 1: High Priority (Widely Used, Well-Documented)
| Package | Category | Rationale |
|---------|----------|-----------|
| dplyr | Data manipulation | Core tidyverse, diverse functions |
| ggplot2 | Visualization | Complex API, many edge cases |
| stringr | String operations | Clear function contracts |
| tidyr | Data tidying | Multiple function families |
| readr | Data import | File handling edge cases |
| purrr | Functional programming | Higher-order functions |

#### Tier 2: Medium Priority (Infrastructure)
| Package | Category | Rationale |
|---------|----------|-----------|
| vctrs | Custom types | Type coercion patterns |
| rlang | Metaprogramming | Quosures, tidy eval |
| withr | State management | Cleanup patterns |
| glue | String interpolation | Simple but useful |

#### Tier 3: Domain-Specific
| Package | Category | Rationale |
|---------|----------|-----------|
| httr | HTTP requests | API mocking complexity |
| checkmate | Argument validation | Assertion patterns |
| R6 | OOP | Reference semantics |

### Task Generation Methodology

#### Option A: Synthetic Bug Generation
1. Take well-tested function from target package
2. Introduce controlled bug (wrong operator, missing edge case)
3. Create instruction asking to write test that catches bug
4. Validate with existing tests

#### Option B: Documentation Mining
1. Extract function examples from roxygen2 docs
2. Create tasks from examples that demonstrate edge cases
3. Use doc examples as reference tests

#### Option C: GitHub Issue Mining (Future)
1. Scrape bug reports from package GitHub repos
2. Extract minimal reproducible examples
3. Create tasks from real-world issues

### Split Strategy

**DECISION NEEDED**: Define proper split for optimization

Current state: 18 tasks, all in `train` split

Proposed strategy:
- **Train (60%)**: 10-12 tasks for skill optimization
- **Dev (20%)**: 3-4 tasks for early stopping/hyperparameter tuning
- **Held-out (20%)**: 3-4 tasks for final evaluation

### Implementation Steps

1. [ ] Create task generation script for new packages
2. [ ] Generate 20+ tasks from dplyr package
3. [ ] Generate 20+ tasks from stringr package
4. [ ] Define and implement split strategy
5. [ ] Re-run baselines with expanded dataset
6. [ ] Document task generation methodology

### Estimated Effort

- Task generation script: 2-3 hours
- Generate tasks from 2 packages: 1-2 hours
- Re-run baselines: 8-10 hours (with free models)
- Documentation: 1 hour

**Total**: ~15 hours

---

## 8.1 MVP v1.1 (Revised Scope)

> **Architect Review Feedback Integration** - This section addresses the revised MVP parameters and new components identified during architectural review.

### Revised MVP Parameters

After initial review, the MVP scope has been adjusted to be more achievable and focused:

| Parameter | Original | Revised | Rationale |
|-----------|----------|---------|-----------|
| **Task Count** | 50-65 | 10-20 | Curated high-quality tasks only |
| **Generations** | 50 | 5 | Faster iteration, validate signal first |
| **Population Size** | 20 | 6-8 | Smaller, more focused search |
| **Source Packages** | cli, withr | cli only | Single package to establish baseline |
| **Expansion Strategy** | Multi-package | Expand only if signal is stable | Validate approach before scaling |

### New Components

#### 8.1.1 Experiment Orchestrator/Tracker

**Purpose**: Track all experiment runs with full reproducibility and artifact management.

```python
# experiment/orchestrator.py

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json
from datetime import datetime
from pathlib import Path

@dataclass
class RunMetadata:
    """Metadata for tracking experiment runs."""
    run_id: str
    seed: int
    skill_version_hash: str
    config_hash: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Cost tracking
    total_tokens: int = 0
    api_calls: int = 0
    estimated_cost_usd: float = 0.0
    
    # Timing
    total_wall_time_seconds: float = 0.0
    eval_time_seconds: float = 0.0
    optimization_time_seconds: float = 0.0


@dataclass  
class ArtifactPaths:
    """Paths to experiment artifacts."""
    checkpoint_dir: Path
    logs_dir: Path
    metrics_file: Path
    skill_snapshots_dir: Path
    evaluation_results_dir: Path


class ExperimentOrchestrator:
    """Orchestrate and track optimization experiments."""
    
    def __init__(self, base_dir: Path = Path("experiments")):
        self.base_dir = base_dir
        self.current_run: Optional[RunMetadata] = None
        self.artifacts: Optional[ArtifactPaths] = None
    
    def start_run(
        self, 
        config: dict, 
        seed: int,
        skill_content: str
    ) -> str:
        """Initialize a new experiment run."""
        run_id = self._generate_run_id(seed)
        
        # Create run directory structure
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup artifact paths
        self.artifacts = ArtifactPaths(
            checkpoint_dir=run_dir / "checkpoints",
            logs_dir=run_dir / "logs",
            metrics_file=run_dir / "metrics.jsonl",
            skill_snapshots_dir=run_dir / "skill_snapshots",
            evaluation_results_dir=run_dir / "evaluations"
        )
        
        # Create directories
        for path in [self.artifacts.checkpoint_dir, self.artifacts.logs_dir,
                     self.artifacts.skill_snapshots_dir, self.artifacts.evaluation_results_dir]:
            path.mkdir(exist_ok=True)
        
        # Compute hashes for version tracking
        skill_hash = hashlib.sha256(skill_content.encode()).hexdigest()[:12]
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        
        self.current_run = RunMetadata(
            run_id=run_id,
            seed=seed,
            skill_version_hash=skill_hash,
            config_hash=config_hash,
            start_time=datetime.now()
        )
        
        # Save run config
        self._save_config(run_dir / "config.json", config, seed, skill_hash, config_hash)
        
        return run_id
    
    def log_metrics(self, metrics: dict) -> None:
        """Append metrics to the metrics file."""
        if self.artifacts is None:
            raise RuntimeError("No active run")
        
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.current_run.run_id,
            **metrics
        }
        
        with open(self.artifacts.metrics_file, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")
    
    def save_skill_snapshot(self, skill_content: str, generation: int, score: float) -> Path:
        """Save a skill snapshot for later analysis."""
        if self.artifacts is None:
            raise RuntimeError("No active run")
        
        snapshot_path = (
            self.artifacts.skill_snapshots_dir / 
            f"gen_{generation:03d}_score_{score:.4f}.md"
        )
        snapshot_path.write_text(skill_content)
        return snapshot_path
    
    def end_run(self, final_metrics: dict) -> None:
        """Finalize the current run."""
        if self.current_run is None:
            return
        
        self.current_run.end_time = datetime.now()
        self.current_run.total_wall_time_seconds = (
            self.current_run.end_time - self.current_run.start_time
        ).total_seconds()
        
        # Update with final metrics
        self.current_run.total_tokens = final_metrics.get("total_tokens", 0)
        self.current_run.api_calls = final_metrics.get("api_calls", 0)
        self.current_run.estimated_cost_usd = final_metrics.get("estimated_cost_usd", 0.0)
        
        # Save run summary
        self._save_run_summary()
    
    def _generate_run_id(self, seed: int) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}_seed{seed}"
    
    def _save_config(self, path: Path, config: dict, seed: int, skill_hash: str, config_hash: str) -> None:
        """Save configuration for reproducibility."""
        full_config = {
            "config": config,
            "seed": seed,
            "skill_version_hash": skill_hash,
            "config_hash": config_hash,
            "created_at": datetime.now().isoformat()
        }
        path.write_text(json.dumps(full_config, indent=2))
    
    def _save_run_summary(self) -> None:
        """Save final run summary."""
        if self.current_run is None or self.artifacts is None:
            return
        
        summary_path = self.base_dir / self.current_run.run_id / "run_summary.json"
        summary_path.write_text(json.dumps({
            "run_id": self.current_run.run_id,
            "seed": self.current_run.seed,
            "skill_version_hash": self.current_run.skill_version_hash,
            "config_hash": self.current_run.config_hash,
            "start_time": self.current_run.start_time.isoformat(),
            "end_time": self.current_run.end_time.isoformat() if self.current_run.end_time else None,
            "total_wall_time_seconds": self.current_run.total_wall_time_seconds,
            "total_tokens": self.current_run.total_tokens,
            "api_calls": self.current_run.api_calls,
            "estimated_cost_usd": self.current_run.estimated_cost_usd,
        }, indent=2))
```

#### 8.1.2 Enhanced Scoring System

**Purpose**: Multi-dimensional scoring beyond binary pass/fail.

```python
# evaluation/enhanced_scoring.py

from dataclasses import dataclass
from typing import Optional
import re
import subprocess
from pathlib import Path

@dataclass
class EnhancedScore:
    """Multi-dimensional evaluation score."""
    # Primary metrics
    test_pass_rate: float  # 0.0 - 1.0
    
    # Style compliance
    style_score: float  # 0.0 - 1.0
    style_violations: list[str]
    
    # Pattern compliance
    pattern_score: float  # 0.0 - 1.0
    required_patterns_found: list[str]
    required_patterns_missing: list[str]
    
    # Compilation/runtime checks
    compiles: bool
    runtime_errors: list[str]
    
    # Weighted composite
    composite_score: float
    
    # Raw outputs for debugging
    raw_test_output: str
    raw_style_output: str


class EnhancedScorer:
    """Score solutions on multiple dimensions."""
    
    # Weight configuration
    WEIGHTS = {
        'test_pass': 0.50,
        'style': 0.20,
        'pattern': 0.20,
        'compile': 0.10,
    }
    
    # Required patterns for testthat 3+ 
    REQUIRED_PATTERNS = {
        'describe_it': r'describe\s*\(\s*["\']',
        'test_that': r'test_that\s*\(',
        'expect_': r'expect_[a-z_]+\s*\(',
    }
    
    # Style constraints (lintr rules)
    STYLE_RULES = [
        'line_length_linter',
        'assignment_linter', 
        'infix_spaces_linter',
        'object_length_linter',
    ]
    
    def __init__(self, required_patterns: Optional[dict] = None):
        self.required_patterns = required_patterns or self.REQUIRED_PATTERNS
    
    def score(self, 
              generated_code: str, 
              test_output: str,
              workspace: Path) -> EnhancedScore:
        """Compute enhanced multi-dimensional score."""
        
        # 1. Test pass rate
        test_pass_rate = self._parse_test_pass_rate(test_output)
        
        # 2. Style compliance
        style_result = self._check_style(generated_code, workspace)
        
        # 3. Pattern compliance
        pattern_result = self._check_patterns(generated_code)
        
        # 4. Compilation check (syntax errors)
        compile_result = self._check_compilation(generated_code, workspace)
        
        # 5. Compute composite score
        composite = self._compute_composite(
            test_pass_rate=test_pass_rate,
            style_score=style_result['score'],
            pattern_score=pattern_result['score'],
            compiles=compile_result['success']
        )
        
        return EnhancedScore(
            test_pass_rate=test_pass_rate,
            style_score=style_result['score'],
            style_violations=style_result['violations'],
            pattern_score=pattern_result['score'],
            required_patterns_found=pattern_result['found'],
            required_patterns_missing=pattern_result['missing'],
            compiles=compile_result['success'],
            runtime_errors=compile_result['errors'],
            composite_score=composite,
            raw_test_output=test_output,
            raw_style_output=style_result['raw_output'],
        )
    
    def _parse_test_pass_rate(self, output: str) -> float:
        """Parse testthat output for pass rate."""
        # Match patterns like "5 passed, 2 failed"
        match = re.search(r'(\d+)\s+passed[,\s]+(\d+)\s+failed', output)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            total = passed + failed
            return passed / total if total > 0 else 0.0
        return 0.0
    
    def _check_style(self, code: str, workspace: Path) -> dict:
        """Run lintr on generated code."""
        # Write code to temp file and run lintr
        result = subprocess.run(
            ["Rscript", "-e", f"lintr::lint('{workspace}/test.R')"],
            capture_output=True, text=True
        )
        
        violations = []
        if result.stdout:
            violations = result.stdout.strip().split('\n')
        
        # Score based on violation count
        score = max(0.0, 1.0 - (len(violations) * 0.1))
        
        return {
            'score': score,
            'violations': violations,
            'raw_output': result.stdout
        }
    
    def _check_patterns(self, code: str) -> dict:
        """Check for required testthat patterns."""
        found = []
        missing = []
        
        for pattern_name, pattern_regex in self.required_patterns.items():
            if re.search(pattern_regex, code):
                found.append(pattern_name)
            else:
                missing.append(pattern_name)
        
        total = len(self.required_patterns)
        score = len(found) / total if total > 0 else 0.0
        
        return {
            'score': score,
            'found': found,
            'missing': missing
        }
    
    def _check_compilation(self, code: str, workspace: Path) -> dict:
        """Check R syntax validity."""
        result = subprocess.run(
            ["Rscript", "-e", f"parse(text = readLines('{workspace}/test.R'))"],
            capture_output=True, text=True
        )
        
        errors = []
        success = result.returncode == 0
        if not success:
            errors.append(result.stderr)
        
        return {
            'success': success,
            'errors': errors
        }
    
    def _compute_composite(self, 
                          test_pass_rate: float,
                          style_score: float,
                          pattern_score: float,
                          compiles: bool) -> float:
        """Compute weighted composite score."""
        compile_score = 1.0 if compiles else 0.0
        
        return (
            self.WEIGHTS['test_pass'] * test_pass_rate +
            self.WEIGHTS['style'] * style_score +
            self.WEIGHTS['pattern'] * pattern_score +
            self.WEIGHTS['compile'] * compile_score
        )
```

#### 8.1.3 Non-determinism Handling

**Purpose**: Handle stochastic LLM outputs with repeated runs and robust aggregation.

```python
# evaluation/non_determinism.py

from dataclasses import dataclass
from typing import Optional
import statistics
from collections import defaultdict

@dataclass
class RobustEvaluationResult:
    """Result with non-determinism handling."""
    task_id: str
    mean_score: float
    std_score: float
    median_score: float
    all_scores: list[float]
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    sample_size: int


class NonDeterminismHandler:
    """Handle LLM non-determinism through repetition and bootstrapping."""
    
    def __init__(
        self, 
        num_repeats: int = 2,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        self.num_repeats = num_repeats
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
    
    def evaluate_with_repeats(
        self,
        task,
        skill_prompt: str,
        evaluator,  # EvaluationSandbox
        num_repeats: Optional[int] = None
    ) -> RobustEvaluationResult:
        """Evaluate a task multiple times to handle non-determinism."""
        repeats = num_repeats or self.num_repeats
        
        scores = []
        for i in range(repeats):
            result = evaluator.run_evaluation(task, skill_prompt)
            scores.append(result.composite_score if hasattr(result, 'composite_score') else (1.0 if result.success else 0.0))
        
        return self._compute_robust_stats(task.task_id, scores)
    
    def evaluate_elite_robust(
        self,
        elite_candidates: list,  # List of skill genomes
        tasks: list,
        evaluator,
        adapter
    ) -> dict[str, RobustEvaluationResult]:
        """Robustly evaluate elite candidates with bootstrapped means."""
        
        # First pass: evaluate all candidates on all tasks
        candidate_task_scores: dict[str, list[float]] = defaultdict(list)
        
        for genome in elite_candidates:
            skill = adapter.from_gepa_individual(genome)
            prompt = adapter.to_prompt(skill)
            candidate_id = hash(tuple(genome))  # Simple ID
            
            for task in tasks:
                result = self.evaluate_with_repeats(task, prompt, evaluator)
                candidate_task_scores[candidate_id].append(result.mean_score)
        
        # Bootstrap aggregate scores
        robust_results = {}
        for candidate_id, scores in candidate_task_scores.items():
            robust_results[candidate_id] = self._compute_robust_stats(
                candidate_id, 
                scores
            )
        
        return robust_results
    
    def select_best_elite(
        self,
        robust_results: dict[str, RobustEvaluationResult]
    ) -> tuple[str, float]:
        """Select best candidate using bootstrapped mean."""
        best_id = None
        best_score = -1
        
        for candidate_id, result in robust_results.items():
            # Use lower bound of CI for conservative selection
            conservative_score = result.bootstrap_ci_low
            
            if conservative_score > best_score:
                best_score = conservative_score
                best_id = candidate_id
        
        return best_id, best_score
    
    def _compute_robust_stats(self, identifier: str, scores: list[float]) -> RobustEvaluationResult:
        """Compute robust statistics with bootstrap confidence interval."""
        if not scores:
            return RobustEvaluationResult(
                task_id=identifier,
                mean_score=0.0,
                std_score=0.0,
                median_score=0.0,
                all_scores=[],
                bootstrap_ci_low=0.0,
                bootstrap_ci_high=0.0,
                sample_size=0
            )
        
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        median_score = statistics.median(scores)
        
        # Bootstrap confidence interval
        ci_low, ci_high = self._bootstrap_ci(scores)
        
        return RobustEvaluationResult(
            task_id=identifier,
            mean_score=mean_score,
            std_score=std_score,
            median_score=median_score,
            all_scores=scores,
            bootstrap_ci_low=ci_low,
            bootstrap_ci_high=ci_high,
            sample_size=len(scores)
        )
    
    def _bootstrap_ci(self, data: list[float]) -> tuple[float, float]:
        """Compute bootstrap confidence interval."""
        import random
        
        if len(data) < 2:
            return (min(data) if data else 0.0, max(data) if data else 0.0)
        
        bootstrap_means = []
        n = len(data)
        
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            sample = [random.choice(data) for _ in range(n)]
            bootstrap_means.append(statistics.mean(sample))
        
        bootstrap_means.sort()
        
        # Compute percentile-based CI
        alpha = 1 - self.confidence_level
        lower_idx = int(self.bootstrap_samples * alpha / 2)
        upper_idx = int(self.bootstrap_samples * (1 - alpha / 2)) - 1
        
        return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
```

### Reproducibility Improvements

#### Docker Base Image Pinning

```dockerfile
# In Dockerfile.evaluation
# Pin to specific digest for bit-for-bit reproducibility
FROM rocker/tidyverse:4.3.3@sha256:abcd1234...

# Verify image digest at runtime
RUN echo "Expected SHA256: abcd1234..." && \
    docker inspect --format='{{.Id}}' | grep -q "abcd1234"
```

#### R Package Lockfile

```r
# renv.lock - Generated by renv::snapshot()
{
  "R": {
    "Version": "4.3.3",
    "Repository": "CRAN"
  },
  "Packages": {
    "testthat": {
      "Package": "testthat",
      "Version": "3.2.1",
      "Source": "Repository",
      "Repository": "RSPM/CRAN"
    },
    "cli": {
      "Package": "cli",
      "Version": "3.6.2",
      "Source": "Repository"
    },
    "withr": {
      "Package": "withr",
      "Version": "3.0.0",
      "Source": "Repository"
    }
  }
}
```

#### Claude CLI Version Pinning

```bash
# scripts/install_claude_cli.sh
#!/bin/bash
set -euo pipefail

CLAUDE_VERSION="${CLAUDE_VERSION:-1.0.0}"
CLAUDE_SHA256="${CLAUDE_SHA256:-expected_sha256_hash}"

# Download specific version
curl -fsSL \
  "https://github.com/anthropics/claude-code/releases/download/v${CLAUDE_VERSION}/claude-code-linux-x64.tar.gz" \
  -o /tmp/claude.tar.gz

# Verify checksum
echo "${CLAUDE_SHA256}  /tmp/claude.tar.gz" | sha256sum -c -

# Install
tar -xzf /tmp/claude.tar.gz -C /usr/local/bin
rm /tmp/claude.tar.gz

# Verify installation
claude --version
echo "Claude CLI v${CLAUDE_VERSION} installed successfully"
```

### Task Data Hygiene

#### Deduplication

```python
# task_generator/deduplication.py

import difflib
from typing import Iterable

class TaskDeduplicator:
    """Remove near-identical tasks from dataset."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, tasks: list[TestingTask]) -> list[TestingTask]:
        """Remove tasks that are too similar to existing ones."""
        unique_tasks = []
        
        for task in tasks:
            if not self._is_near_duplicate(task, unique_tasks):
                unique_tasks.append(task)
        
        return unique_tasks
    
    def _is_near_duplicate(self, candidate: TestingTask, existing: list[TestingTask]) -> bool:
        """Check if candidate is too similar to any existing task."""
        for task in existing:
            if self._compute_similarity(candidate, task) > self.similarity_threshold:
                return True
        return False
    
    def _compute_similarity(self, task1: TestingTask, task2: TestingTask) -> float:
        """Compute similarity score between two tasks."""
        # Compare instructions
        inst_sim = difflib.SequenceMatcher(
            None, 
            task1.instruction.lower(), 
            task2.instruction.lower()
        ).ratio()
        
        # Compare context (function code)
        ctx_sim = difflib.SequenceMatcher(
            None,
            task1.context.lower(),
            task2.context.lower()
        ).ratio()
        
        # Weighted average
        return 0.6 * inst_sim + 0.4 * ctx_sim
```

#### Split by Function/Package Boundary

```python
# task_generator/split_strategy.py

from collections import defaultdict

class BoundaryAwareSplitter:
    """Split tasks by function/package boundary, not randomly."""
    
    def __init__(self, train_ratio: float = 0.6, dev_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
    
    def split(
        self, 
        tasks: list[TestingTask]
    ) -> dict[str, list[TestingTask]]:
        """Split tasks ensuring no function appears in multiple splits."""
        
        # Group tasks by function
        by_function: dict[str, list[TestingTask]] = defaultdict(list)
        for task in tasks:
            by_function[task.function_name].append(task)
        
        # Shuffle function names (not individual tasks)
        function_names = list(by_function.keys())
        import random
        random.shuffle(function_names)
        
        # Split by function
        n_functions = len(function_names)
        train_end = int(n_functions * self.train_ratio)
        dev_end = train_end + int(n_functions * self.dev_ratio)
        
        return {
            'train': self._collect_tasks(function_names[:train_end], by_function),
            'dev': self._collect_tasks(function_names[train_end:dev_end], by_function),
            'held_out': self._collect_tasks(function_names[dev_end:], by_function)
        }
    
    def _collect_tasks(
        self, 
        function_names: list[str], 
        by_function: dict
    ) -> list[TestingTask]:
        """Collect all tasks for given function names."""
        tasks = []
        for fn in function_names:
            tasks.extend(by_function[fn])
        return tasks
```

#### Contamination Checks

```python
# task_generator/contamination_check.py

class ContaminationChecker:
    """Check for contamination between splits."""
    
    def check_contamination(
        self, 
        train_tasks: list[TestingTask],
        held_out_tasks: list[TestingTask]
    ) -> list[dict]:
        """Check if held-out tasks are contaminated by train tasks."""
        issues = []
        
        train_contexts = {t.task_id: t.context for t in train_tasks}
        
        for held_task in held_out_tasks:
            # Check for exact function matches
            for train_id, train_ctx in train_contexts.items():
                if held_task.context == train_ctx:
                    issues.append({
                        'type': 'exact_match',
                        'held_out_task': held_task.task_id,
                        'train_task': train_id
                    })
                
                # Check for high similarity
                similarity = self._quick_similarity(held_task.context, train_ctx)
                if similarity > 0.9:
                    issues.append({
                        'type': 'high_similarity',
                        'held_out_task': held_task.task_id,
                        'train_task': train_id,
                        'similarity': similarity
                    })
        
        return issues
    
    def _quick_similarity(self, text1: str, text2: str) -> float:
        """Fast similarity check using hash of n-grams."""
        # Use min-hash or similar for large datasets
        ngrams1 = set(self._ngrams(text1, 3))
        ngrams2 = set(self._ngrams(text2, 3))
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _ngrams(self, text: str, n: int) -> Iterable[str]:
        """Generate character n-grams."""
        for i in range(len(text) - n + 1):
            yield text[i:i+n]
```

### Package Fixture Template

```r
# fixtures/package_scaffold/DESCRIPTION
Package: testtarget
Title: Test Target Package
Version: 0.0.0.9000
Description: Minimal package for testing skill outputs.
Depends: R (>= 4.0)
Imports: 
Suggests: testthat (>= 3.0.0)
Config/testthat/edition: 3

# fixtures/package_scaffold/NAMESPACE
exportPattern("^[[:alpha:]]")

# fixtures/package_scaffold/tests/testthat.R
library(testthat)
library(testtarget)

test_check("testtarget")

# fixtures/package_scaffold/tests/testthat/_snaps/.gitkeep
# Empty file for snapshot tests

# fixtures/package_scaffold/R/helper-test.R
# Helper functions for tests (if needed)
```

```python
# fixtures/scaffold_generator.py

from pathlib import Path
import shutil

class PackageScaffoldGenerator:
    """Generate proper R package scaffolds for evaluation."""
    
    TEMPLATE_DIR = Path("fixtures/package_scaffold")
    
    def generate(self, dest: Path, task: TestingTask) -> None:
        """Generate package scaffold for a task."""
        # Copy base template
        shutil.copytree(self.TEMPLATE_DIR, dest)
        
        # Write function to test
        r_dir = dest / "R"
        r_dir.mkdir(exist_ok=True)
        (r_dir / "target.R").write_text(task.context)
        
        # Update DESCRIPTION with task-specific info
        desc_path = dest / "DESCRIPTION"
        desc = desc_path.read_text()
        desc = desc.replace("testtarget", f"task_{task.task_id}")
        desc_path.write_text(desc)
        
        # Create testthat directory structure
        test_dir = dest / "tests" / "testthat"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create _snaps directory for snapshot tests
        snaps_dir = test_dir / "_snaps"
        snaps_dir.mkdir(exist_ok=True)
        (snaps_dir / ".gitkeep").touch()
```

### Revised Success Metrics (MVP v1.1)

| Metric | Baseline | Target | Notes |
|--------|----------|--------|-------|
| **Tasks passing** | 6/20 (30%) | 12/20 (60%) | +30% improvement |
| **Dev set accuracy** | 35% | 55% | +20% improvement |
| **Style compliance** | 40% | 70% | Enhanced scoring |
| **Pattern compliance** | 50% | 80% | testthat 3+ patterns |
| **Compute budget** | N/A | < $50 | Per optimization run |
| **Wall time** | N/A | < 4 hours | Per optimization run |

### Revised GEPA Configuration

```yaml
# configs/gepa_config_v1.1.yaml
gepa:
  population_size: 6        # Reduced from 20
  max_generations: 5        # Reduced from 50
  mutation_rate: 0.2        # Slightly increased for smaller population
  crossover_rate: 0.5       # Reduced 
  elite_size: 2             # Top 2 for reproduction

evaluation:
  batch_size: 5             # Evaluate 5 tasks at a time
  timeout_per_task: 300
  parallel_workers: 2       # Reduced parallelism
  num_repeats: 2            # For non-determinism handling

scoring:
  weights:
    test_pass: 0.50
    style: 0.20
    pattern: 0.20
    compile: 0.10

constraints:
  max_mutation_ratio: 0.3
  preserve_sections:
    - "## Required Packages"
    - "## Environment Setup"

experiment:
  seed: 42
  track_artifacts: true
  log_level: "DEBUG"
```

---

## 9. Success Metrics

### Primary Metrics

| Metric | Baseline Target | Success Threshold |
|--------|----------------|-------------------|
| **Held-out accuracy** | 60% | ≥ 75% (+15% improvement) |
| **Dev set accuracy** | 62% | ≥ 78% |
| **Failure rate reduction** | 40% failures | ≤ 25% failures |

### Secondary Metrics

| Metric | Target |
|--------|--------|
| **Task quality rate** | ≥ 90% pass quality gates |
| **Evaluation stability** | < 5% variance across runs |
| **Optimization convergence** | Within 30 generations |
| **Generation time** | < 2 hours per generation |

### Quality Indicators

```python
# metrics.py

def compute_metrics(results: list[EvaluationResult]) -> dict:
    """Compute success metrics from evaluation results."""
    
    total = len(results)
    successes = sum(1 for r in results if r.success)
    
    # Failure breakdown
    failures_by_category = defaultdict(int)
    for r in results:
        if not r.success and r.failure_analysis:
            failures_by_category[r.failure_analysis.failure_category] += 1
    
    # Time metrics
    times = [r.execution_time for r in results]
    
    return {
        'accuracy': successes / total,
        'total_tasks': total,
        'successful_tasks': successes,
        'failure_breakdown': dict(failures_by_category),
        'avg_execution_time': sum(times) / len(times),
        'p95_execution_time': sorted(times)[int(len(times) * 0.95)],
    }
```

---

## 10. Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Claude Code CLI instability** | Medium | High | Add retry logic, fallback to API, extensive logging |
| **Docker performance issues** | Medium | Medium | Pre-warm containers, optimize image, resource limits |
| **Task quality too low** | Medium | High | Multi-stage validation, human review loop |
| **GEPA doesn't improve skill** | Low | High | Baseline ablation, try alternative mutation strategies |
| **R package version conflicts** | Medium | Medium | Pin versions, use renv for reproducibility |

### Process Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Scope creep** | High | Medium | Strict MVP definition, defer enhancements |
| **LLM API costs exceed budget** | Medium | Medium | Cache results, batch requests, monitor spend |
| **Held-out contamination** | Low | High | Strict data hygiene, separate storage |

### Contingency Plans

```python
# contingency.py

class ContingencyPlan:
    """Fallback strategies for common failure modes."""
    
    def handle_claude_cli_failure(self, task: TestingTask) -> str:
        """Fallback when Claude Code CLI fails."""
        # 1. Retry with exponential backoff
        # 2. Fall back to direct Anthropic API
        # 3. Use cached response if available
        # 4. Mark task as skipped and continue
        pass
    
    def handle_docker_timeout(self, container_id: str) -> None:
        """Handle container timeout or hang."""
        # 1. Kill container
        # 2. Log state for debugging
        # 3. Restart with fresh container
        # 4. Reduce timeout for subsequent runs
        pass
    
    def handle_poor_task_quality(self, tasks: list[TestingTask]) -> list[TestingTask]:
        """Handle when too many tasks fail quality gates."""
        # 1. Identify common failure patterns
        # 2. Adjust generation parameters
        # 3. Request human review
        # 4. Reduce task count if necessary
        pass
```

---

## 11. File Structure

```
posit-gskill/
├── README.md
├── PLAN.md                          # This document
├── pyproject.toml                   # Python project config (uv)
├── uv.lock                          # Python dependency lockfile
├── requirements.R                   # R dependencies
├── renv.lock                        # R package lockfile (reproducibility)
├── Makefile                         # Common commands
├── .env.example                     # Environment template
│
├── task_generator/                  # Phase 1
│   ├── __init__.py
│   ├── ast_parser.py
│   ├── templates.py
│   ├── llm_refine.py
│   ├── quality_gate.py
│   ├── generator.py
│   ├── deduplication.py             # NEW: Near-duplicate detection
│   ├── split_strategy.py            # NEW: Boundary-aware splitting
│   └── contamination_check.py       # NEW: Cross-split contamination checks
│
├── evaluation/                      # Phase 2
│   ├── __init__.py
│   ├── sandbox.py
│   ├── workspace.py
│   ├── runner.py
│   ├── test_executor.py
│   ├── reflector.py
│   ├── enhanced_scoring.py          # NEW: Multi-dimensional scoring
│   └── non_determinism.py           # NEW: Repeated runs & bootstrapping
│
├── optimization/                    # Phase 3
│   ├── __init__.py
│   ├── adapter.py
│   ├── evaluator.py
│   ├── optimizer.py
│   ├── checkpoint.py
│   └── constraints.py
│
├── experiment/                      # NEW: Experiment tracking
│   ├── __init__.py
│   ├── orchestrator.py              # Run IDs, seeds, version hashing
│   ├── artifact_manager.py          # Cost/time logging, artifact storage
│   └── run_summary.py               # Run summary generation
│
├── fixtures/                        # NEW: Package scaffolds
│   ├── package_scaffold/
│   │   ├── DESCRIPTION
│   │   ├── NAMESPACE
│   │   ├── R/
│   │   └── tests/
│   │       ├── testthat.R
│   │       └── testthat/
│   │           └── _snaps/
│   └── scaffold_generator.py
│
├── docker/
│   ├── Dockerfile.evaluation
│   ├── Dockerfile.development
│   └── entrypoint.sh
│
├── configs/
│   ├── gepa_config.yaml
│   ├── gepa_config_v1.1.yaml        # NEW: Revised MVP config
│   ├── evaluation_config.yaml
│   └── logging_config.yaml
│
├── tasks/                           # Generated tasks
│   ├── train/
│   ├── dev/
│   └── held_out/
│
├── skills/                          # Skill definitions
│   ├── testing-r-packages-orig.md
│   └── testing-r-packages-opt.md
│
├── experiments/                     # NEW: Experiment artifacts
│   └── run_YYYYMMDD_HHMMSS_seed*/
│       ├── config.json
│       ├── run_summary.json
│       ├── metrics.jsonl
│       ├── checkpoints/
│       ├── logs/
│       ├── skill_snapshots/
│       └── evaluations/
│
├── results/                         # Optimization results
│   ├── baseline/
│   ├── optimized/
│   └── comparison/
│
├── analysis/                        # Analysis outputs
│   ├── failure_modes.md
│   ├── skill_diff.md
│   └── recommendations.md
│
├── scripts/                         # Utility scripts
│   ├── generate_tasks.py
│   ├── validate_tasks.py
│   ├── run_evaluation.py
│   ├── run_optimization.py
│   ├── compare_results.py
│   └── install_claude_cli.sh        # NEW: Version-pinned install
│
├── tests/                           # Unit tests
│   ├── test_generator/
│   ├── test_evaluation/
│   ├── test_optimization/
│   └── test_experiment/             # NEW: Experiment tracking tests
│
└── notebooks/                       # Analysis notebooks
    ├── task_analysis.ipynb
    └── optimization_analysis.ipynb
```

---

## Appendix A: Makefile Commands

```makefile
# Makefile

.PHONY: setup docker tasks evaluate optimize test clean

setup:
	uv sync
	Rscript requirements.R

docker:
	docker build -t posit-gskill-eval:latest -f docker/Dockerfile.evaluation .

tasks:
	uv run python scripts/generate_tasks.py --packages cli,withr --output tasks/

validate:
	uv run python scripts/validate_tasks.py --tasks tasks/

evaluate:
	uv run python scripts/run_evaluation.py --task $(TASK) --skill $(SKILL)

optimize:
	uv run python scripts/run_optimization.py --config configs/gepa_config.yaml

test:
	uv run pytest tests/ -v

clean:
	rm -rf tasks/*.json
	rm -rf results/*
	rm -rf __pycache__ .pytest_cache
```

---

## Appendix B: Environment Variables

```bash
# .env.example

# Anthropic API
ANTHROPIC_API_KEY=your_key_here

# Claude Code CLI
CLAUDE_CODE_PATH=/usr/local/bin/claude

# Docker
DOCKER_IMAGE=posit-gskill-eval:latest

# Paths
TASK_DIR=./tasks
SKILL_DIR=./skills
RESULTS_DIR=./results

# GEPA Settings
GEPA_POPULATION_SIZE=20
GEPA_MAX_GENERATIONS=50

# Evaluation
EVAL_TIMEOUT=300
EVAL_PARALLEL_WORKERS=4
```

---

## Appendix C: Quick Start

```bash
# 1. Setup
git clone https://github.com/your-org/posit-gskill.git
cd posit-gskill
make setup

# 2. Build Docker image
make docker

# 3. Generate tasks
make tasks

# 4. Run optimization
make optimize

# 5. View results
open results/comparison/improvement_analysis.md
```

---

## 9. Phase 4 Progress & Learnings (2026-02-21)

### Current Status

**Completed:**
- ✅ Parallel batch evaluation infrastructure (`evaluate_batch.py`)
- ✅ Multi-model optimization support (`MultiModelSkillEvaluator`)
- ✅ YAML configuration system for evaluation runs
- ✅ Pi SDK runner prototype (`PiRunner`, `DockerPiRunner`)
- ✅ Baseline comparison framework

**In Progress:**
- 🔄 Docker + Pi integration (env vars not passing correctly)
- 🔄 GEPA optimization with reflection model

**Not Started:**
- ⏳ Full optimization runs
- ⏳ Held-out evaluation
- ⏳ Failure mode analysis

### Key Learnings

#### 1. Current Skill Hurts Performance

| Condition | Pass Rate | Model |
|-----------|-----------|-------|
| No-skill baseline | 58.3% | glm-4.5 |
| With current skill | 33.3% | glm-4.5 |
| **Delta** | **-25.0pp** | |

The hand-authored `testing-r-packages-orig.md` skill actively harms performance on glm-4.5. This is valuable signal for GEPA - there's clear room for improvement.

#### 2. cc-mirror Stack Issues

The current Docker + cc-mirror + LiteLLM stack has multiple issues:
- **~70s overhead** per container for variant creation
- **Streaming errors** from LiteLLM parsing z.ai responses
- **Rate limits** with parallel execution
- **Prompt passing failures** ("Input must be provided")
- **Burns paid credits** (no free model support)

#### 3. Pi SDK as Alternative

Pi (`@mariozechner/pi-coding-agent`) offers a simpler alternative:
- **~2-5s startup** (no variant creation)
- **Agent Skills standard** (drop-in compatible with existing skills)
- **Free model support** via OpenRouter
- **Stable JSON output** mode
- **Built-in tools** (read, bash, edit, write)

**Status:** Pi runner code created, Docker integration pending env var fix.

#### 4. Multi-Model Optimization Strategy

For robust skill optimization, evaluate on multiple models simultaneously:
- **Models:** glm-4.5 (weak), glm-4.6 (medium), glm-4.7 (strong)
- **Aggregation:** Use `min()` to optimize worst-case performance
- **Goal:** Skill that generalizes across model capabilities

#### 5. Model Concurrency Limits

| Model | Concurrent Requests | Recommended Workers |
|-------|---------------------|---------------------|
| glm-4.5 | 10 | 5-8 |
| glm-4.6 | 10 | 5-8 |
| glm-4.7 | 5 | 3-4 |
| glm-5 | 3 | 2 |

### Architecture Decision

**Recommended path forward:** Replace cc-mirror with Docker + Pi:

```
┌─────────────────────────────────────────────────────────────┐
│               Target Architecture                           │
├─────────────────────────────────────────────────────────────┤
│  Python (GEPA)                                              │
│       ↓                                                     │
│  DockerPiRunner                                             │
│       ↓                                                     │
│  docker run --entrypoint "" posit-gskill-eval:latest        │
│       ↓                                                     │
│  pi --print --mode json --model <provider/model> "prompt"   │
│       ↓                                                     │
│  testthat verification                                      │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Sandbox isolation (Docker)
- Fast startup (no cc-mirror variant)
- Free models (OpenRouter)
- Simpler code (no LiteLLM)

### Blocking Issues

1. **Pi env vars in Docker** - API keys not reaching Pi inside container
2. **Reflection model** - Need to verify `zai-coding-plan/glm-5` works for GEPA reflection

### Next Steps

1. Fix Docker + Pi env var passing
2. Run baseline comparison with Pi runner
3. Run GEPA optimization with working stack
4. Compare optimized skill vs no-skill baseline
5. Expand task set if ceiling effect persists

---

*Document Version: 1.1.0*  
*Last Updated: 2026-02-20*  
*Status: Revised MVP Scope - Architect Review Feedback Integrated*

### Changelog
- **v1.1.0**: Added Section 8.1 (MVP v1.1 Revised Scope) with:
  - Revised MVP parameters (10-20 tasks, 5 generations, population 6-8, cli-only)
  - Experiment Orchestrator/Tracker for run management
  - Enhanced Scoring system (multi-dimensional)
  - Non-determinism handling (repeated runs, bootstrapping)
  - Reproducibility improvements (Docker digest pinning, renv lockfile, Claude CLI versioning)
  - Task data hygiene (deduplication, boundary-aware splitting, contamination checks)
  - Package fixture template
  - Bug fixes: `_parse_results()` task parameter, difficulty mapping syntax, optimizer tracking
  - Updated tooling: Python uses `uv`, JS/TS uses `bun`
- **v1.0.0**: Initial MVP planning document
