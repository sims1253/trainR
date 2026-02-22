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

        # Get full PR info
        cmd = [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,title,body,baseRefName,headRefName,files,url,mergedAt",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gh pr view failed: {result.stderr}")

        full_pr_data = json.loads(result.stdout)

        # Get diff
        cmd = ["gh", "pr", "diff", str(pr_number), "--repo", repo]
        diff_result = subprocess.run(cmd, capture_output=True, text=True)
        diff_text = diff_result.stdout if diff_result.returncode == 0 else "Unable to fetch diff"

        # Get files changed
        files_changed = [f.get("path", "") for f in full_pr_data.get("files", [])]

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
        )


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

Reject PRs that:
- Are primarily documentation changes
- Involve massive refactoring without clear test boundaries
- Require extensive domain-specific knowledge
- Have ambiguous or unclear requirements
- Don't modify any source code
- Are primarily configuration/deployment changes"""

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

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """
        Initialize LLM judge.

        Uses unified config from configs/llm.yaml by default.
        """
        config = get_llm_config()

        self.model = model or config.get_mining_model()
        self.api_key = api_key or config.get_api_key()

        # Determine provider based on model name
        if self.model.startswith("openrouter/"):
            self.provider = "litellm"
        elif "gpt" in self.model.lower() or "o1" in self.model.lower():
            self.provider = "openai"
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        elif "claude" in self.model.lower():
            self.provider = "anthropic"
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            self.provider = "litellm"

        if not self.api_key:
            print("Warning: No API key found for provider. Set appropriate env var.")

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
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": response_format.model_json_schema(),
            },
            temperature=0.3,
            api_key=self.api_key,
        )
        return response_format.model_validate_json(response.choices[0].message.content)

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
            # Return a failed evaluation
            return MinedTaskSchema(
                task_type=TaskType.BUG_FIX,
                difficulty=Difficulty.MEDIUM,
                quality_score=0,
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

    Args:
        pr_details: PRDetails object
        llm_schema: MinedTaskSchema from LLM evaluation

    Returns:
        Dict in standard task format
    """
    task_id = f"{pr_details.repo_name.replace('/', '_')}_{pr_details.number}"

    task = {
        "task_id": task_id,
        "source": {
            "type": "github_pr",
            "repo": pr_details.repo_name,
            "pr_number": pr_details.number,
            "pr_url": pr_details.url,
            "merged_at": pr_details.merged_at.isoformat() if pr_details.merged_at else None,
            "issue_number": pr_details.linked_issue.get("number")
            if pr_details.linked_issue
            else None,
        },
        "metadata": {
            "task_type": llm_schema.task_type.value,
            "difficulty": llm_schema.difficulty.value,
            "quality_score": llm_schema.quality_score,
            "mined_at": datetime.now(timezone.utc).isoformat(),
        },
        "task": {
            "instruction": llm_schema.instruction,
            "context": llm_schema.context,
        },
        "tests": {
            "fail_to_pass": llm_schema.fail_to_pass,
            "pass_to_pass": llm_schema.pass_to_pass,
        },
        "solution_hints": {
            "key_changes": llm_schema.key_changes,
            "potential_pitfalls": llm_schema.potential_pitfalls,
            "original_diff": pr_details.code_diff[:10000],  # Truncated
        },
        "files": {
            "changed": pr_details.files_changed,
            "test_changes": pr_details.test_diff[:5000] if pr_details.test_diff else None,
        },
    }

    return task


def load_repos_from_yaml(yaml_path: str) -> list[dict]:
    """Load repository list from YAML config file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    repos = []

    # Flatten all priority categories
    for category in ["high_priority", "medium_priority", "bayesian_stan", "specialized"]:
        if category in config:
            for repo_config in config[category]:
                repos.append(
                    {
                        "owner": repo_config["owner"],
                        "repo": repo_config["repo"],
                        "name": f"{repo_config['owner']}/{repo_config['repo']}",
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
        for r in repos:
            print(f"  - {r['name']}")
        print("\nDry run complete. No tasks collected.")
        return

    # Initialize components
    try:
        miner = GitHubPRMiner()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    judge = LLMTaskJudge(model=args.model)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "repos_processed": 0,
        "prs_found": 0,
        "prs_with_tests": 0,
        "prs_with_issues": 0,
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

            # Check for test changes
            has_tests = miner.has_test_changes(pr)
            if args.require_tests and not has_tests:
                print("    → Skipping: No test file changes")
                continue
            stats["prs_with_tests"] += 1

            # Get PR details
            try:
                pr_details = miner.get_pr_details(repo_name, pr)
            except Exception as e:
                print(f"    → Error fetching details: {e}")
                continue

            # Check for linked issue
            if args.require_issue and not pr_details.linked_issue:
                print("    → Skipping: No linked issue")
                continue
            stats["prs_with_issues"] += 1

            # Evaluate with LLM
            print("    → Evaluating with LLM...")
            llm_schema = judge.evaluate_task_quality(pr_details)

            # Check quality threshold
            if llm_schema.quality_score < args.min_quality:
                print(
                    f"    → Rejected: Quality score {llm_schema.quality_score} < {args.min_quality}"
                )
                stats["tasks_rejected"] += 1
                continue

            if not llm_schema.is_good_task:
                print(f"    → Rejected: {llm_schema.rejection_reason}")
                stats["tasks_rejected"] += 1
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
    print("MINING COMPLETE")
    print("=" * 60)
    print(f"Repositories processed: {stats['repos_processed']}")
    print(f"Total PRs found: {stats['prs_found']}")
    print(f"PRs with test changes: {stats['prs_with_tests']}")
    print(f"PRs with linked issues: {stats['prs_with_issues']}")
    print(f"Tasks created: {stats['tasks_created']}")
    print(f"Tasks rejected: {stats['tasks_rejected']}")
    print(f"\nOutput directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
