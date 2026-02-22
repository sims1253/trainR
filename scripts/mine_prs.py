#!/usr/bin/env python3
"""
GitHub PR Mining System for R Testing Tasks

This script mines merged PRs from target R packages to extract
high-quality testing tasks. It uses:
- gh CLI for GitHub API access
- LiteLLM/OpenAI for LLM-based task evaluation
- Pydantic for structured outputs
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import yaml
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_llm_config
from task_generator.mined_task import (
    Difficulty,
    MinedTaskSchema,
    PRAnalysisInput,
    TaskType,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


@dataclass
class PRDetails:
    """Container for all PR-related information"""

    number: int
    title: str
    body: str
    url: str
    merged_at: datetime | None
    files_changed: list[str]
    code_diff: str
    test_diff: str | None
    linked_issue: dict[str, Any] | None
    repo_name: str
    # Commit SHIs for SWE-bench format
    base_commit: str | None = None
    head_commit: str | None = None
    merged_commit: str | None = None
    # Files with diffs for patch separation
    files_with_diffs: list[dict[str, Any]] | None = None

    def to_analysis_input(self) -> PRAnalysisInput:
        """Convert to LLM input format"""
        issue = self.linked_issue or {}
        return PRAnalysisInput(
            pr_title=self.title,
            pr_body=self.body or "",
            issue_title=issue.get("title", ""),
            issue_body=issue.get("body", ""),
            files_changed=self.files_changed,
            code_diff=self.code_diff[:50000],  # Truncate to avoid token limits
            test_diff=self.test_diff[:20000] if self.test_diff else None,
        )


def separate_patches(files: list[dict]) -> tuple[str, str]:
    """
    Separate file changes into patch (code) and test_patch (tests).
    Follows SWE-bench convention: files with 'test' in path are test files.

    Args:
        files: List of file dicts with 'path' and 'diff' keys

    Returns:
        Tuple of (patch, test_patch) as unified diff strings
    """
    test_path_patterns = ["test", "tests", "testing", "spec", "e2e"]

    code_changes = []
    test_changes = []

    for file in files:
        path = file.get("path", "")
        diff = file.get("diff", "")

        if not diff:
            continue

        # Check if this is a test file
        path_lower = path.lower()
        is_test = any(pattern in path_lower for pattern in test_path_patterns)

        if is_test:
            test_changes.append(diff)
        else:
            code_changes.append(diff)

    patch = "\n".join(code_changes) if code_changes else ""
    test_patch = "\n".join(test_changes) if test_changes else ""

    return patch, test_patch


class GitHubPRMiner:
    """Mine GitHub PRs using gh CLI"""

    # Patterns to identify test files in R packages
    TEST_FILE_PATTERNS: ClassVar[list[str]] = [
        "tests/",
        "test-",
        "_test.",
        "test/",
    ]

    # Patterns to find linked issues in PR body
    ISSUE_PATTERNS: ClassVar[list[str]] = [
        r"(?:fixes|closes|resolves|fix|close|resolve)\s*#(\d+)",
        r"(?:fixes|closes|resolves|fix|close|resolve)\s+https://github\.com/[^/]+/[^/]+/issues/(\d+)",
        r"issue\s*#?(\d+)",
    ]

    def __init__(self):
        """Check gh is authenticated"""
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("gh CLI not authenticated. Run 'gh auth login'")

    def get_merged_prs(self, repo: str, since_days: int = 30, max_prs: int = 50) -> list[dict]:
        """
        Get merged PRs from repo using gh CLI.

        Args:
            repo: Repository in format "owner/repo"
            since_days: Number of days to look back
            max_prs: Maximum number of PRs to return

        Returns:
            List of PR dicts with basic info
        """
        cmd = [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--limit",
            str(max_prs),
            "--json",
            "number,title,body,mergedAt,files,additions,deletions,url",
            "--search",
            f"merged:>{self._since_date(since_days)}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error fetching PRs from {repo}: {result.stderr}")
            return []
        return json.loads(result.stdout)

    def _since_date(self, days: int) -> str:
        """Calculate the since date string"""
        return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    def _extract_linked_issues(self, body: str) -> list[str]:
        """Extract issue numbers from PR body (e.g., 'Fixes #123')"""
        issues = []
        for pattern in self.ISSUE_PATTERNS:
            issues.extend(re.findall(pattern, body or "", re.IGNORECASE))
        return list(dict.fromkeys(issues))  # Preserve order, remove duplicates

    def _get_issue_details(self, repo: str, issue_number: str) -> dict:
        """Get issue details using gh CLI"""
        cmd = [
            "gh",
            "issue",
            "view",
            issue_number,
            "--repo",
            repo,
            "--json",
            "number,title,body,labels,url",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {}

    def has_test_changes(self, pr_data: dict) -> bool:
        """
        Check if a PR modifies any test files.

        Args:
            pr_data: PR dict from gh CLI

        Returns:
            True if PR modifies test files
        """
        for f in pr_data.get("files", []):
            path = f.get("path", "") if isinstance(f, dict) else str(f)
            path_lower = path.lower()
            if any(p in path_lower for p in self.TEST_FILE_PATTERNS):
                return True
        return False

    def extract_test_changes(self, pr_diff: str) -> str | None:
        """
        Extract test-related changes from the full diff.

        Args:
            pr_diff: Full PR diff string

        Returns:
            Test changes portion or None if no test changes
        """
        test_sections = []
        in_test_file = False
        current_diff = []

        for line in pr_diff.split("\n"):
            if line.startswith("diff --git"):
                # Save previous file if it was a test file
                if in_test_file and current_diff:
                    test_sections.append("\n".join(current_diff))

                # Start new file
                current_diff = [line]
                in_test_file = any(p in line.lower() for p in self.TEST_FILE_PATTERNS)
            else:
                current_diff.append(line)

        # Don't forget the last file
        if in_test_file and current_diff:
            test_sections.append("\n".join(current_diff))

        return "\n\n".join(test_sections) if test_sections else None

    def get_pr_details(self, repo: str, pr_data: dict) -> PRDetails:
        """
        Get full PR details including diff.

        Args:
            repo: Repository in format "owner/repo"
            pr_data: Basic PR dict from gh pr list

        Returns:
            PRDetails with all relevant information
        """
        pr_number = pr_data["number"]

        # Get full PR info including commit SHAs
        cmd = [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,title,body,baseRefName,headRefName,files,url,mergedAt,"
            "baseRefOid,headRefOid,mergeCommit",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gh pr view failed: {result.stderr}")

        full_pr_data = json.loads(result.stdout)

        # Get commit SHAs for SWE-bench format
        base_commit = full_pr_data.get("baseRefOid")
        head_commit = full_pr_data.get("headRefOid")
        merged_commit = full_pr_data.get("mergeCommit", {}).get("oid")

        # Get diff
        cmd = ["gh", "pr", "diff", str(pr_number), "--repo", repo]
        diff_result = subprocess.run(cmd, capture_output=True, text=True)
        diff_text = diff_result.stdout if diff_result.returncode == 0 else "Unable to fetch diff"

        # Get files changed and build files_with_diffs list
        files_data = full_pr_data.get("files", [])
        files_changed = [f.get("path", "") for f in files_data]

        # Parse individual file diffs from unified diff
        files_with_diffs = self._parse_file_diffs(diff_text, files_data)

        # Extract test changes
        test_diff = self.extract_test_changes(diff_text)

        # Get linked issues
        linked_issue = None
        issue_numbers = self._extract_linked_issues(full_pr_data.get("body", ""))
        if issue_numbers:
            issue_details = self._get_issue_details(repo, issue_numbers[0])
            if issue_details:
                linked_issue = {
                    "number": issue_details.get("number"),
                    "title": issue_details.get("title", ""),
                    "body": issue_details.get("body", ""),
                    "url": issue_details.get("url", ""),
                    "labels": [label.get("name", "") for label in issue_details.get("labels", [])],
                }

        # Parse merged_at
        merged_at_str = full_pr_data.get("mergedAt")
        if merged_at_str:
            merged_at = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
        else:
            merged_at = None

        return PRDetails(
            number=full_pr_data["number"],
            title=full_pr_data.get("title", ""),
            body=full_pr_data.get("body", "") or "",
            url=full_pr_data.get("url", ""),
            merged_at=merged_at,
            files_changed=files_changed,
            code_diff=diff_text,
            test_diff=test_diff,
            linked_issue=linked_issue,
            repo_name=repo,
            base_commit=base_commit,
            head_commit=head_commit,
            merged_commit=merged_commit,
            files_with_diffs=files_with_diffs,
        )

    def _parse_file_diffs(self, diff_text: str, files_data: list[dict]) -> list[dict]:
        """
        Parse unified diff into individual file diffs.

        Args:
            diff_text: Full unified diff string
            files_data: List of file metadata from GitHub API

        Returns:
            List of dicts with 'path' and 'diff' keys
        """
        files_with_diffs = []
        current_file = None
        current_diff = []

        for line in diff_text.split("\n"):
            if line.startswith("diff --git"):
                # Save previous file if exists
                if current_file and current_diff:
                    files_with_diffs.append(
                        {
                            "path": current_file,
                            "diff": "\n".join(current_diff),
                        }
                    )
                # Start new file - extract path from "diff --git a/path b/path"
                parts = line.split(" ")
                if len(parts) >= 4:
                    # Get path from b/path (the new file version)
                    current_file = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                current_diff = [line]
            else:
                if current_file:
                    current_diff.append(line)

        # Don't forget the last file
        if current_file and current_diff:
            files_with_diffs.append(
                {
                    "path": current_file,
                    "diff": "\n".join(current_diff),
                }
            )

        return files_with_diffs


class LLMTaskJudge:
    """Use LLM to evaluate PR quality and extract structured task data"""

    SYSTEM_PROMPT = """You are an expert R package developer and testing specialist.
Your task is to analyze GitHub pull requests and evaluate their suitability as
testing/evaluation tasks for training AI coding assistants.

Evaluate each PR on:
1. **Clarity**: Is the task clearly defined from the issue/PR description?
2. **Testability**: Are there clear test cases that verify the fix/feature?
3. **Self-contained**: Can the task be understood and solved without extensive context?
4. **Difficulty appropriateness**: Is it neither trivial nor impossibly complex?
5. **Realism**: Does it represent a realistic real-world coding task?

## Response Format (JSON Schema)

You MUST include these required fields in your JSON response:

**task_type** (required): One of "bug_fix", "feature_impl", "test_writing", "refactor", "documentation", "performance", "security", "dependency"

**difficulty** (required): One of "easy" (single file, <20 lines), "medium" (multiple files, 20-100 lines), "hard" (complex, >100 lines)

**quality_score** (required): Integer 1-10. 7+ is good. Consider: clarity, testability, self-contained, realistic.

**instruction** (required): Clear task description. Example: "Fix the bug in test_discovery() where nested directories are not being scanned. The function should recursively find all test files."

**is_good_task** (required): Boolean. True if this PR would make a good evaluation task, False if too ambiguous/large/unsuitable.

Optional fields:
- **context**: Relevant code snippets, function signatures
- **fail_to_pass**: Array of test descriptions that verify the fix
- **pass_to_pass**: Array of tests for regression checking
- **key_changes**: Array of hints about what changes are needed
- **potential_pitfalls**: Array of common mistakes to avoid
- **rejection_reason**: If is_good_task is false, explain why"""

    EVALUATION_PROMPT = """Analyze this GitHub PR and evaluate its quality as a testing task.

## PR Information
**Title**: {pr_title}
**Body**: {pr_body}

## Linked Issue
**Title**: {issue_title}
**Body**: {issue_body}

## Files Changed
{files_changed}

## Code Changes (diff)
```diff
{code_diff}
```

## Test Changes (diff)
```diff
{test_diff}
```

Provide a structured evaluation of this PR as a potential testing task.
"""

    def __init__(self, model: str | None = None, api_key: str | None = None, verbose: bool = False):
        """
        Initialize LLM judge.
        Uses unified config from configs/llm.yaml by default.
        """
        self.verbose = verbose
        config = get_llm_config()

        # Get model name (just the name, e.g., "glm-5-free")
        model_name = model or config.get_mining_model()
        self.model = model_name

        # Get full model config from llm.yaml
        try:
            model_cfg = config.get_model_config(model_name)
            provider = model_cfg.get("provider", "")
            self.provider = provider

            # Get LiteLLM-compatible model string
            self._litellm_model = config.get_litellm_model(model_name)

            # Get base_url if needed
            base_url = config.get_model_base_url(model_name)
            self._litellm_kwargs = {"base_url": base_url} if base_url else {}

            # Get API key from config
            self.api_key = api_key or config.get_model_api_key(model_name)

        except ValueError:
            # Model not in llm.yaml - treat as raw model ID with provider prefix
            # Fallback for models like "opencode/glm-5-free" passed directly
            self._litellm_model = model_name
            if model_name.startswith("opencode/") or model_name.startswith("zai/"):
                self._litellm_model = "openai/" + model_name.split("/", 1)[1]
                self.provider = model_name.split("/", 1)[0]
            elif model_name.startswith("openrouter/"):
                self.provider = "openrouter"
            else:
                self.provider = "litellm"

            # Try to get base_url from legacy method
            base_url = config.get_base_url(model_name)
            self._litellm_kwargs = {"base_url": base_url} if base_url else {}

            # Fallback API key detection for provider prefix format
            if api_key:
                self.api_key = api_key
            elif model_name.startswith("opencode/"):
                self.api_key = os.environ.get("OPENCODE_API_KEY")
            elif model_name.startswith("zai/"):
                self.api_key = os.environ.get("Z_AI_API_KEY")
            elif model_name.startswith("openrouter/"):
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
            else:
                self.api_key = config.get_api_key()

        if not self.api_key:
            print(f"Warning: No API key found for model '{model_name}'. Check llm.yaml config.")

    def _call_openai(
        self, messages: list[dict[str, Any]], response_format: type[BaseModel]
    ) -> BaseModel:
        """Call OpenAI API with structured output"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            response_format=response_format,
            temperature=0.3,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed response")
        return parsed

    def _call_litellm(
        self, messages: list[dict[str, Any]], response_format: type[BaseModel]
    ) -> BaseModel:
        """Call LiteLLM for model-agnostic access"""
        import json

        import litellm

        # Build schema instruction to include in prompt (some models ignore response_format.schema)
        schema_json = response_format.model_json_schema()
        schema_instruction = f"""
You MUST respond with valid JSON matching this schema:
{json.dumps(schema_json, indent=2)}

Required fields:
- task_type: one of "bug_fix", "feature_impl", "test_writing", "refactor", "documentation", "performance", "security", "dependency"
- difficulty: one of "easy", "medium", "hard"
- quality_score: integer 1-10 (7+ is good)
- instruction: string describing the task
- is_good_task: boolean"""

        # Append schema instruction to last user message
        enhanced_messages = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i == len(messages) - 1:
                # Last user message - append schema instruction
                enhanced_messages.append(
                    {"role": "user", "content": msg["content"] + schema_instruction}
                )
            else:
                enhanced_messages.append(msg)

        if self.verbose:
            print("\n" + "=" * 60)
            print("LLM INPUT:")
            print(f"Model: {self._litellm_model}")
            print(f"Base URL: {self._litellm_kwargs.get('base_url', 'default')}")
            print(f"API Key: {self.api_key[:15]}..." if self.api_key else "No API key")
            print("Messages:")
            for msg in enhanced_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "... (truncated)"
                print(f"  [{role}]: {content}")

        # Drop unsupported params for compatibility (e.g., gpt-5 doesn't support temperature!=1)
        litellm.drop_params = True

        response = litellm.completion(
            model=self._litellm_model,  # Use transformed model name for LiteLLM
            messages=enhanced_messages,  # Use enhanced messages with schema in prompt
            response_format={"type": "json_object"},  # Just JSON mode, no schema
            temperature=0.3,  # Will be dropped automatically for models that don't support it
            api_key=self.api_key,
            max_tokens=2000,  # Increased for reasoning models
            **self._litellm_kwargs,  # Passes api_base for custom endpoints
        )

        raw_content = response.choices[0].message.content

        # Handle reasoning models (e.g., glm-5-free) that put content in reasoning_content
        if not raw_content:
            # Try reasoning_content field (used by some reasoning models)
            raw_content = getattr(response.choices[0].message, "reasoning_content", None)
            # Also check provider_specific_fields
            if not raw_content:
                provider_fields = getattr(
                    response.choices[0].message, "provider_specific_fields", None
                )
                if provider_fields:
                    raw_content = provider_fields.get("reasoning_content")

        if not raw_content:
            raise ValueError("LLM returned empty content")

        if self.verbose:
            content_source = (
                "reasoning_content" if not response.choices[0].message.content else "content"
            )
            print(f"\nLLM OUTPUT (from {content_source}):")
            print(raw_content[:1000] if len(raw_content) > 1000 else raw_content)
            print("=" * 60 + "\n")

        return response_format.model_validate_json(raw_content)

    def evaluate_task_quality(self, pr_details: PRDetails) -> MinedTaskSchema:
        """
        Evaluate PR quality and extract structured task data.

        Args:
            pr_details: PRDetails object with all PR information

        Returns:
            MinedTaskSchema with evaluation and extracted task data
        """
        analysis_input = pr_details.to_analysis_input()

        prompt = self.EVALUATION_PROMPT.format(
            pr_title=analysis_input.pr_title,
            pr_body=analysis_input.pr_body[:2000],  # Truncate for token limits
            issue_title=analysis_input.issue_title,
            issue_body=analysis_input.issue_body[:2000] if analysis_input.issue_body else "",
            files_changed="\n".join(f"- {f}" for f in analysis_input.files_changed),
            code_diff=analysis_input.code_diff,
            test_diff=analysis_input.test_diff or "No test changes",
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            if self.provider == "openai":
                return cast("MinedTaskSchema", self._call_openai(messages, MinedTaskSchema))
            else:
                return cast("MinedTaskSchema", self._call_litellm(messages, MinedTaskSchema))
        except Exception as e:
            if self.verbose:
                print(f"\n[LLM ERROR] {type(e).__name__}: {e}")
            # Return a failed evaluation
            return MinedTaskSchema(
                task_type=TaskType.BUG_FIX,
                difficulty=Difficulty.MEDIUM,
                quality_score=1,
                instruction=f"Error evaluating PR: {e!s}",
                context="",
                fail_to_pass=[],
                pass_to_pass=[],
                key_changes=[],
                potential_pitfalls=[],
                is_good_task=False,
                rejection_reason=f"LLM evaluation failed: {e!s}",
            )


def create_task_from_pr(pr_details: PRDetails, llm_schema: MinedTaskSchema) -> dict:
    """
    Create final task JSON from PR details and LLM evaluation.
    Uses SWE-bench format with separated patch and test_patch.

    Args:
        pr_details: PRDetails object
        llm_schema: MinedTaskSchema from LLM evaluation

    Returns:
        Dict in SWE-bench compatible task format
    """
    task_id = f"{pr_details.repo_name.replace('/', '_')}_{pr_details.number}"
    _owner, _repo = pr_details.repo_name.split("/", 1)

    # Separate patches into code and test changes
    files_with_diffs = pr_details.files_with_diffs or []
    patch, test_patch = separate_patches(files_with_diffs)

    task = {
        "task_id": task_id,
        "source": {
            "repo": pr_details.repo_name,
            "pr_number": pr_details.number,
            "base_commit": pr_details.base_commit,
            "head_commit": pr_details.head_commit,
            "merged_commit": pr_details.merged_commit,
            "pr_url": pr_details.url,
            "merged_at": pr_details.merged_at.isoformat() if pr_details.merged_at else None,
            "issue_url": pr_details.linked_issue.get("url") if pr_details.linked_issue else None,
        },
        "problem_statement": {
            "title": pr_details.title,
            "body": pr_details.body,
            "issue_title": pr_details.linked_issue.get("title")
            if pr_details.linked_issue
            else None,
            "issue_body": pr_details.linked_issue.get("body") if pr_details.linked_issue else None,
        },
        "patch": patch,
        "test_patch": test_patch,
        "tests": {
            "fail_to_pass": llm_schema.fail_to_pass or [],
            "pass_to_pass": llm_schema.pass_to_pass or [],
        },
        "metadata": {
            "task_type": llm_schema.task_type.value,
            "difficulty": llm_schema.difficulty.value,
            "quality_score": llm_schema.quality_score,
            "mined_at": datetime.now(timezone.utc).isoformat(),
            "key_changes": llm_schema.key_changes,
            "potential_pitfalls": llm_schema.potential_pitfalls,
        },
        "instruction": llm_schema.instruction,
        "context": llm_schema.context,
    }

    return task


def load_repos_from_yaml(yaml_path: str) -> list[dict]:
    """Load repository list from YAML config file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # New flat format: repos is a single list
    if "repos" in config:
        repos = []
        for repo_entry in config["repos"]:
            repos.append(
                {
                    "owner": repo_entry["owner"],
                    "repo": repo_entry["repo"],
                    "name": f"{repo_entry['owner']}/{repo_entry['repo']}",
                    "domain": repo_entry.get("domain"),
                    "stars": repo_entry.get("stars"),
                    "notes": repo_entry.get("notes"),
                }
            )
        return repos

    # Fallback to old nested format for backwards compatibility
    repos = []
    repo_config = config.get("repositories", config)

    for category in [
        "task_generation",
        "high_priority",
        "medium_priority",
        "bayesian_stan",
        "specialized",
        "data_table",
        "visualization",
    ]:
        if category in repo_config:
            for repo_entry in repo_config[category]:
                repos.append(
                    {
                        "owner": repo_entry["owner"],
                        "repo": repo_entry["repo"],
                        "name": f"{repo_entry['owner']}/{repo_entry['repo']}",
                    }
                )

    return repos


def load_repos_from_file(file_path: str) -> list[dict]:
    """Load repository list from plain text file (one repo per line)."""
    repos = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "/" in line:
                owner, repo = line.split("/", 1)
                repos.append({"owner": owner, "repo": repo, "name": line})
    return repos


def get_mined_task_ids(output_dir: Path) -> tuple[set[str], dict[str, dict]]:
    """Get set of task IDs that have already been mined and rejected PRs.

    Returns:
        Tuple of (mined_task_ids, rejected_prs_info)
        - mined_task_ids: Set of successfully mined task IDs
        - rejected_prs_info: Dict mapping task_id to rejection details
    """
    task_ids = set()
    rejected_prs = {}

    if output_dir.exists():
        # Load successful tasks
        for task_file in output_dir.glob("*.json"):
            task_id = task_file.stem
            task_ids.add(task_id)

    # Load rejected PRs cache
    rejected_file = output_dir / "rejected_prs.json"
    if rejected_file.exists():
        rejected_prs = json.loads(rejected_file.read_text())

    return task_ids, rejected_prs


def save_rejected_pr(output_dir: Path, task_id: str, details: dict) -> None:
    """Save rejected PR details to cache.

    Args:
        output_dir: Output directory for tasks
        task_id: Task identifier (owner_repo_pr_number)
        details: Rejection details including quality_score, reason, etc.
    """
    rejected_file = output_dir / "rejected_prs.json"

    # Load existing rejections
    rejected_prs = json.loads(rejected_file.read_text()) if rejected_file.exists() else {}

    # Add/update this rejection
    rejected_prs[task_id] = {
        **details,
        "rejected_at": datetime.now().isoformat(),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_file.write_text(json.dumps(rejected_prs, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Mine GitHub PRs for R testing tasks")
    parser.add_argument(
        "--repo",
        help="Single repository to mine (format: owner/repo)",
    )
    parser.add_argument(
        "--repos-file",
        help="File with list of repos (one per line or YAML config)",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=30,
        help="Look back N days for merged PRs (default: 30)",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=7,
        help="Minimum LLM quality score (1-10, default: 7)",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=50,
        help="Max PRs to process per repo (default: 50)",
    )
    parser.add_argument(
        "--output",
        default="tasks/mined",
        help="Output directory for task JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--model",
        default=None,  # Now uses config file default
        help="LLM model for evaluation (default: from configs/llm.yaml)",
    )
    parser.add_argument(
        "--require-tests",
        action="store_true",
        default=True,
        help="Only process PRs that modify test files",
    )
    parser.add_argument(
        "--require-issue",
        action="store_true",
        default=True,
        help="Only process PRs with linked issues",
    )
    parser.add_argument(
        "--no-require-tests",
        action="store_true",
        help="Don't require test file changes",
    )
    parser.add_argument(
        "--no-require-issue",
        action="store_true",
        help="Don't require linked issue",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed debug info including LLM inputs/outputs",
    )

    args = parser.parse_args()

    # Collect repos to mine
    repos = []
    if args.repo:
        owner, repo = args.repo.split("/", 1)
        repos.append({"owner": owner, "repo": repo, "name": args.repo})
    if args.repos_file:
        if args.repos_file.endswith(".yaml") or args.repos_file.endswith(".yml"):
            repos.extend(load_repos_from_yaml(args.repos_file))
        else:
            repos.extend(load_repos_from_file(args.repos_file))

    if not repos:
        parser.error("No repos specified. Use --repo or --repos-file")

    # Remove duplicates
    seen = set()
    unique_repos = []
    for r in repos:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_repos.append(r)
    repos = unique_repos

    print(f"Mining {len(repos)} repositories...")
    if args.dry_run:
        print("DRY RUN MODE - No LLM calls, no files written\n")

    # Initialize components
    try:
        miner = GitHubPRMiner()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    judge = LLMTaskJudge(model=args.model, verbose=args.verbose)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get already-mined task IDs AND rejected PRs
    mined_task_ids, rejected_prs = get_mined_task_ids(output_dir)

    if mined_task_ids and not args.dry_run:
        print(f"Found {len(mined_task_ids)} previously mined tasks (will skip)")
    if rejected_prs and not args.dry_run:
        print(f"Found {len(rejected_prs)} previously rejected PRs (will skip)")

    # Statistics
    stats = {
        "repos_processed": 0,
        "prs_found": 0,
        "already_mined": 0,
        "previously_rejected": 0,
        "prs_filtered_no_tests": 0,
        "prs_filtered_no_issue": 0,
        "would_evaluate": 0,  # for dry run
        "tasks_created": 0,
        "tasks_rejected": 0,
    }

    # Mine each repo
    for repo_info in repos:
        repo_name = repo_info["name"]
        print(f"\n{'=' * 60}")
        print(f"Mining: {repo_name}")
        print("=" * 60)

        stats["repos_processed"] += 1

        # Get merged PRs
        prs = miner.get_merged_prs(repo_name, since_days=args.since_days, max_prs=args.max_prs)
        print(f"Found {len(prs)} merged PRs in last {args.since_days} days")
        stats["prs_found"] += len(prs)

        for pr in prs:
            print(f"\n  PR #{pr['number']}: {pr['title'][:60]}...")

            # Generate task ID for this PR
            owner, repo = repo_name.split("/", 1)
            task_id = f"{owner}_{repo}_{pr['number']}"

            # Skip if already mined
            if task_id in mined_task_ids:
                print(f"    → Skipping PR #{pr['number']} (already mined)")
                stats["already_mined"] += 1
                continue

            # Skip if previously rejected
            if task_id in rejected_prs:
                prev_rejection = rejected_prs[task_id]
                print(
                    f"    → Skipping PR #{pr['number']} (previously rejected: {prev_rejection.get('rejection_reason', 'unknown')})"
                )
                stats["previously_rejected"] += 1
                continue

            # Check for test changes
            has_tests = miner.has_test_changes(pr)
            if not args.no_require_tests and args.require_tests and not has_tests:
                if args.dry_run:
                    print("    → Would skip: No test file changes")
                else:
                    print("    → Skipping: No test file changes")
                stats["prs_filtered_no_tests"] += 1
                continue

            # Get PR details
            try:
                pr_details = miner.get_pr_details(repo_name, pr)
            except Exception as e:
                print(f"    → Error fetching details: {e}")
                continue

            # Check for linked issue
            if not args.no_require_issue and args.require_issue and not pr_details.linked_issue:
                if args.dry_run:
                    print("    → Would skip: No linked issue")
                else:
                    print("    → Skipping: No linked issue")
                stats["prs_filtered_no_issue"] += 1
                continue

            # Evaluate with LLM (skip in dry-run)
            if args.dry_run:
                print("    → Would evaluate with LLM")
                stats["would_evaluate"] += 1
                continue

            print("    → Evaluating with LLM...")
            llm_schema = judge.evaluate_task_quality(pr_details)

            # Check quality threshold and task suitability
            if not llm_schema.is_good_task or llm_schema.quality_score < args.min_quality:
                rejection_reason = (
                    llm_schema.rejection_reason
                    or f"Quality score {llm_schema.quality_score} < {args.min_quality}"
                )
                print(f"    → Rejected: {rejection_reason}")
                stats["tasks_rejected"] += 1

                # Save rejection details
                save_rejected_pr(
                    output_dir,
                    task_id,
                    {
                        "quality_score": llm_schema.quality_score,
                        "task_type": str(llm_schema.task_type.value)
                        if llm_schema.task_type
                        else None,
                        "difficulty": str(llm_schema.difficulty.value)
                        if llm_schema.difficulty
                        else None,
                        "rejection_reason": rejection_reason,
                        "pr_title": pr_details.title,
                        "pr_url": pr_details.url,
                    },
                )
                continue

            # Create task
            task = create_task_from_pr(pr_details, llm_schema)

            # Save task
            task_file = output_dir / f"{task['task_id']}.json"
            with open(task_file, "w") as f:
                json.dump(task, f, indent=2)

            print(
                f"    → ✓ Task created: {llm_schema.task_type.value} "
                f"(difficulty: {llm_schema.difficulty.value}, "
                f"quality: {llm_schema.quality_score}/10)"
            )
            stats["tasks_created"] += 1

    # Print summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("MINING COMPLETE")
    print("=" * 60)
    print(f"Repositories processed: {stats['repos_processed']}")
    print(f"Total PRs found: {stats['prs_found']}")
    if stats["already_mined"] > 0:
        print(f"PRs skipped (already mined): {stats['already_mined']}")
    if stats["previously_rejected"] > 0:
        print(f"PRs skipped (previously rejected): {stats['previously_rejected']}")
    if args.dry_run:
        print(f"PRs would be filtered (no tests): {stats['prs_filtered_no_tests']}")
        print(f"PRs would be filtered (no issue): {stats['prs_filtered_no_issue']}")
        print(f"PRs would be evaluated by LLM: {stats['would_evaluate']}")
        print("Tasks might be created: unknown (requires LLM evaluation)")
    else:
        print(f"PRs filtered (no tests): {stats['prs_filtered_no_tests']}")
        print(f"PRs filtered (no issue): {stats['prs_filtered_no_issue']}")
        print(f"Tasks created: {stats['tasks_created']}")
        print(f"Tasks rejected: {stats['tasks_rejected']}")
        print(f"\nOutput directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
