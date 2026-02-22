#!/usr/bin/env python3
"""
GitHub PR Mining System for R Testing Tasks

This script mines merged PRs from target R packages to extract
high-quality testing tasks. It uses:
- PyGithub for GitHub API access
- LiteLLM/OpenAI for LLM-based task evaluation
- Pydantic for structured outputs
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from github import Github, GithubException, PullRequest, Repository
from github.Issue import Issue
from pydantic import BaseModel, Field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_generator.mined_task import (
    Difficulty,
    MinedTaskSchema,
    PRAnalysisInput,
    TaskType,
)


@dataclass
class PRDetails:
    """Container for all PR-related information"""

    number: int
    title: str
    body: str
    url: str
    merged_at: datetime
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
    """Mine merged PRs from GitHub repositories for testing tasks"""

    # Patterns to identify test files in R packages
    TEST_FILE_PATTERNS = [
        r"tests/",
        r"test-",
        r"_test\.R$",
        r"test/.+\.R$",
    ]

    # Patterns to find linked issues in PR body
    ISSUE_PATTERNS = [
        r"(?:fixes|closes|resolves|fix|close|resolve)\s*#(\d+)",
        r"(?:fixes|closes|resolves|fix|close|resolve)\s+https://github\.com/[^/]+/[^/]+/issues/(\d+)",
        r"issue\s*#?(\d+)",
    ]

    def __init__(self, token: str | None = None):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token. Falls back to GITHUB_TOKEN env var.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN env var or pass token parameter."
            )
        self.github = Github(self.token)

    def get_merged_prs(
        self, repo_name: str, since_days: int = 30, max_prs: int = 50
    ) -> list[PullRequest.PullRequest]:
        """
        Fetch merged PRs from a repository within the specified time range.

        Args:
            repo_name: Repository in format "owner/repo"
            since_days: Number of days to look back
            max_prs: Maximum number of PRs to return

        Returns:
            List of merged PullRequest objects
        """
        repo = self.github.get_repo(repo_name)
        since = datetime.now(timezone.utc) - timedelta(days=since_days)

        merged_prs = []
        try:
            # Get closed PRs and filter for merged ones
            prs = repo.get_pulls(state="closed", sort="updated", direction="desc")

            for pr in prs:
                if len(merged_prs) >= max_prs:
                    break

                # Check if merged and within time range
                if pr.merged and pr.merged_at and pr.merged_at >= since:
                    merged_prs.append(pr)

        except GithubException as e:
            print(f"Error fetching PRs from {repo_name}: {e}")
            return []

        return merged_prs

    def has_test_changes(self, pr: PullRequest.PullRequest) -> bool:
        """
        Check if a PR modifies any test files.

        Args:
            pr: PullRequest object

        Returns:
            True if PR modifies test files
        """
        try:
            files = pr.get_files()
            for file in files:
                for pattern in self.TEST_FILE_PATTERNS:
                    if re.search(pattern, file.filename, re.IGNORECASE):
                        return True
        except GithubException:
            pass
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
        current_file = None
        in_test_file = False
        current_diff = []

        for line in pr_diff.split("\n"):
            if line.startswith("diff --git"):
                # Save previous file if it was a test file
                if in_test_file and current_diff:
                    test_sections.append("\n".join(current_diff))

                # Start new file
                current_file = line
                current_diff = [line]
                in_test_file = any(
                    re.search(p, line, re.IGNORECASE) for p in self.TEST_FILE_PATTERNS
                )
            else:
                current_diff.append(line)

        # Don't forget the last file
        if in_test_file and current_diff:
            test_sections.append("\n".join(current_diff))

        return "\n\n".join(test_sections) if test_sections else None

    def get_linked_issue(self, pr: PullRequest.PullRequest) -> dict[str, Any] | None:
        """
        Extract linked issue from PR body and fetch its details.

        Args:
            pr: PullRequest object

        Returns:
            Dict with issue details or None
        """
        body = pr.body or ""
        issue_numbers = []

        for pattern in self.ISSUE_PATTERNS:
            matches = re.findall(pattern, body, re.IGNORECASE)
            issue_numbers.extend(int(m) for m in matches)

        if not issue_numbers:
            return None

        # Try to fetch the first linked issue
        repo = pr.base.repo
        for issue_num in issue_numbers:
            try:
                issue = repo.get_issue(issue_num)
                return {
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "url": issue.html_url,
                    "labels": [label.name for label in issue.labels],
                }
            except GithubException:
                continue

        return None

    def get_pr_details(self, pr: PullRequest.PullRequest) -> PRDetails:
        """
        Get comprehensive details about a PR.

        Args:
            pr: PullRequest object

        Returns:
            PRDetails with all relevant information
        """
        # Get files changed
        files_changed = [f.filename for f in pr.get_files()]

        # Get diff
        try:
            diff = pr.get_diff()
            diff_text = diff if isinstance(diff, str) else diff.raw_data
        except GithubException:
            diff_text = "Unable to fetch diff"

        # Extract test changes
        test_diff = self.extract_test_changes(diff_text)

        # Get linked issue
        linked_issue = self.get_linked_issue(pr)

        return PRDetails(
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            url=pr.html_url,
            merged_at=pr.merged_at,
            files_changed=files_changed,
            code_diff=diff_text,
            test_diff=test_diff,
            linked_issue=linked_issue,
            repo_name=pr.base.repo.full_name,
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

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        """
        Initialize LLM judge.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")
            api_key: API key. Falls back to OPENAI_API_KEY or ANTHROPIC_API_KEY env vars.
        """
        self.model = model
        self.api_key = api_key

        # Determine which client to use based on model name
        if "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
            self.provider = "openai"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif "claude" in model.lower():
            self.provider = "anthropic"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        else:
            # Default to litellm for other models
            self.provider = "litellm"
            self.api_key = api_key

        if not self.api_key:
            print("Warning: No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY env var.")

    def _call_openai(self, messages: list[dict], response_format: type[BaseModel]) -> BaseModel:
        """Call OpenAI API with structured output"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
            temperature=0.3,
        )
        return response.choices[0].message.parsed

    def _call_litellm(self, messages: list[dict], response_format: type[BaseModel]) -> BaseModel:
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
                return self._call_openai(messages, MinedTaskSchema)
            else:
                return self._call_litellm(messages, MinedTaskSchema)
        except Exception as e:
            # Return a failed evaluation
            return MinedTaskSchema(
                task_type=TaskType.BUG_FIX,
                difficulty=Difficulty.MEDIUM,
                quality_score=0,
                instruction=f"Error evaluating PR: {str(e)}",
                context="",
                fail_to_pass=[],
                pass_to_pass=[],
                key_changes=[],
                potential_pitfalls=[],
                is_good_task=False,
                rejection_reason=f"LLM evaluation failed: {str(e)}",
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
            if line and not line.startswith("#"):
                if "/" in line:
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
        default="gpt-4o-mini",
        help="LLM model to use for evaluation (default: gpt-4o-mini)",
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
    except ValueError as e:
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
            print(f"\n  PR #{pr.number}: {pr.title[:60]}...")

            # Check for test changes
            has_tests = miner.has_test_changes(pr)
            if args.require_tests and not has_tests:
                print("    → Skipping: No test file changes")
                continue
            stats["prs_with_tests"] += 1

            # Get PR details
            try:
                pr_details = miner.get_pr_details(pr)
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
