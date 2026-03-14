#!/usr/bin/env python3
"""
Kaggle Benchmark Runner (APPS-style Evaluation)

Grading:
  1. EXECUTION (Pass/Fail): Does code run? Valid output?
  2. RAW SCORE: Actual CV accuracy
  3. NORMALIZED: 0 = random, 1 = top Kaggle
  4. CHEATING: Similarity to reference solution

Error Types (APPS-style):
  - compile_error: R syntax errors, parse failures
  - runtime_error: Execution crashes, timeouts, missing packages
  - wrong_answer: Runs but produces invalid output
  - success: Runs correctly with valid output
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_llm_config  # noqa: E402


def load_task(path):
    return json.load(open(path))


def get_baselines(task: dict) -> dict:
    task_id = task.get("task_id", "").lower()
    if "titanic" in task_id:
        return {"random": 0.50, "top_kaggle": 0.85}
    elif "house-price" in task_id:
        return {"random": 0.0, "top_kaggle": 0.15}
    else:
        return {"random": 0.50, "top_kaggle": 0.85}


def normalize_score(raw: float, baselines: dict) -> float:
    rand, top = baselines["random"], baselines["top_kaggle"]
    return (raw - rand) / (top - rand) if top > rand else 0.0


def generate_r_code(task: dict, model: str) -> str:
    config = get_llm_config()
    mc = config.get_model_config(model)
    api_key_env = mc.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise RuntimeError(f"Missing api_key_env for model {model}")
    api_key = os.environ.get(api_key_env)
    base_url_raw = mc.get("base_url")
    base_url = base_url_raw if isinstance(base_url_raw, str) else None
    model_id = mc.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"Missing model id for model {model}")
    client = OpenAI(base_url=base_url, api_key=api_key)

    prompt = """Write R code for Titanic survival. Return ONLY R code.

train.csv: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
test.csv: same but no Survived

Requirements:
1. Load data, handle missing values
2. Feature engineering
3. Train model
4. 5-fold CV, print: CV_ACCURACY: 0.XXXX
5. Write submission.csv with PassengerId,Survived"""

    for attempt in range(3):
        try:
            print(f"[LLM] Attempt {attempt + 1}...")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r"```[rR]?\s*\n?", "", content)
            content = re.sub(r"```\s*$", "", content).strip()
            if len(content) > 100:
                print(f"[LLM] Generated {len(content)} chars")
                return content
            time.sleep(2)
        except Exception as e:
            print(f"[LLM] Error: {e}")
            time.sleep(3)
    raise RuntimeError("Failed to generate code")


def check_cheating(code: str, ref: str) -> dict:
    """Check code similarity and detect potential cheating."""
    if not ref:
        return {"similarity": 0.0, "has_reference": False, "warning": None}

    # Normalize for comparison
    def normalize(c):
        lines = []
        for line in c.split("\n"):
            line = re.sub(r"#.*$", "", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    sim = SequenceMatcher(None, normalize(code), normalize(ref)).ratio()

    warning = None
    if sim > 0.5:
        warning = "HIGH - likely copied"
    elif sim > 0.3:
        warning = "MODERATE - review recommended"

    return {
        "similarity": round(sim, 3),
        "has_reference": True,
        "warning": warning,
    }


def check_dangerous_functions(code: str) -> list:
    """
    Check for potentially dangerous R function calls.
    Returns a list of warnings (not blocking, just informational).
    """
    warnings = []

    # Define dangerous patterns with descriptions
    dangerous_patterns = [
        (r"\bsystem\s*\(", "system() - executes shell commands"),
        (r"\bshell\s*\(", "shell() - executes shell commands"),
        (r"\bpipe\s*\(", "pipe() - creates pipe connections"),
        (r"\bfile\.remove\s*\(", "file.remove() - deletes files"),
        (r"\bunlink\s*\(", "unlink() - deletes files/directories"),
        (r"\bdownload\.file\s*\(", "download.file() - downloads files from internet"),
        (r"\bsetwd\s*\(", "setwd() - changes working directory (may interfere with paths)"),
    ]

    for pattern, description in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            # Find line number
            for i, line in enumerate(code.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    warnings.append(
                        {
                            "line": i,
                            "function": description.split(" - ")[0],
                            "description": description,
                            "severity": "warning",
                        }
                    )
                    break

    return warnings


def compare_csv(
    generated_path: Path,
    expected_path: Path,
    tolerance: float = 0.0001,
    ignore_row_order: bool = True,
) -> dict:
    """
    Compare two CSV files with float tolerance.

    Args:
        generated_path: Path to the generated CSV file
        expected_path: Path to the expected CSV file
        tolerance: Float comparison tolerance (default 0.0001)
        ignore_row_order: Whether to ignore row order for comparison

    Returns:
        dict with keys: match, header_match, row_count_match, float_match, details
    """
    result = {
        "match": False,
        "header_match": False,
        "row_count_match": False,
        "float_match": True,  # Default to True, set False if comparison fails
        "details": "",
    }

    # Check if files exist
    if not generated_path.exists():
        result["details"] = f"Generated file not found: {generated_path}"
        return result
    if not expected_path.exists():
        result["details"] = f"Expected file not found: {expected_path}"
        return result

    try:
        with (
            open(generated_path, newline="") as f1,
            open(expected_path, newline="") as f2,
        ):
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)

            rows1 = list(reader1)
            rows2 = list(reader2)

        if not rows1 or not rows2:
            result["details"] = "One or both files are empty"
            return result

        # Compare headers (case-insensitive)
        headers1 = [h.strip().lower() for h in rows1[0]]
        headers2 = [h.strip().lower() for h in rows2[0]]

        result["header_match"] = headers1 == headers2
        if not result["header_match"]:
            result["details"] = f"Header mismatch: got {headers1}, expected {headers2}"
            return result

        # Compare row counts (excluding header)
        data_rows1 = rows1[1:]
        data_rows2 = rows2[1:]
        result["row_count_match"] = len(data_rows1) == len(data_rows2)

        if not result["row_count_match"]:
            result["details"] = (
                f"Row count mismatch: got {len(data_rows1)}, expected {len(data_rows2)}"
            )
            # Continue comparison anyway for partial match info

        # Determine numeric columns
        numeric_cols = []
        for col_idx, _header in enumerate(headers2):
            is_numeric = True
            for row in data_rows2:
                if col_idx < len(row):
                    try:
                        float(row[col_idx])
                    except (ValueError, IndexError):
                        is_numeric = False
                        break
            if is_numeric:
                numeric_cols.append(col_idx)

        # Compare values
        if ignore_row_order:
            # Sort rows for comparison (convert to tuples for sorting)
            sorted_rows1 = sorted([tuple(row) for row in data_rows1])
            sorted_rows2 = sorted([tuple(row) for row in data_rows2])
        else:
            sorted_rows1 = [tuple(row) for row in data_rows1]
            sorted_rows2 = [tuple(row) for row in data_rows2]

        # Compare with tolerance for numeric columns
        mismatches = []
        min_rows = min(len(sorted_rows1), len(sorted_rows2))

        for row_idx in range(min_rows):
            row1 = sorted_rows1[row_idx]
            row2 = sorted_rows2[row_idx]

            for col_idx in range(min(len(row1), len(row2))):
                if col_idx in numeric_cols:
                    # Numeric comparison with tolerance
                    try:
                        val1 = float(row1[col_idx]) if col_idx < len(row1) else None
                        val2 = float(row2[col_idx]) if col_idx < len(row2) else None
                        if val1 is not None and val2 is not None and abs(val1 - val2) > tolerance:
                            result["float_match"] = False
                            mismatches.append(
                                f"Row {row_idx + 1}, Col {col_idx}: {val1} != {val2} (diff: {abs(val1 - val2)})"
                            )
                    except (ValueError, IndexError):
                        pass
                else:
                    # String comparison
                    val1 = row1[col_idx] if col_idx < len(row1) else ""
                    val2 = row2[col_idx] if col_idx < len(row2) else ""
                    if val1 != val2:
                        mismatches.append(f"Row {row_idx + 1}, Col {col_idx}: '{val1}' != '{val2}'")

        # Determine overall match
        result["match"] = (
            result["header_match"]
            and result["row_count_match"]
            and result["float_match"]
            and len(mismatches) == 0
        )

        if mismatches:
            result["details"] = f"Mismatches found: {'; '.join(mismatches[:5])}"
            if len(mismatches) > 5:
                result["details"] += f" ... and {len(mismatches) - 5} more"
        elif not result["row_count_match"]:
            result["details"] = "Row count differs but content matches where comparable"
        else:
            result["details"] = "CSV files match"

    except Exception as e:
        result["details"] = f"Error comparing CSVs: {e!s}"

    return result


def classify_error(stderr: str, returncode: int, stdout: str = "") -> str:
    """
    Classify error type based on stderr/output (APPS-style).

    Returns:
        "compile_error": R syntax errors, parse failures
        "runtime_error": Execution crashes, timeouts, missing packages
        "wrong_answer": Runs but produces invalid output
        "success": Runs correctly with valid output
    """
    combined = stderr + stdout

    # Compile errors: R syntax errors, parse failures
    compile_patterns = [
        r"Error:\s*unexpected",
        r"Error in parse",
        r"syntax error",
        r"unexpected symbol",
        r"unexpected string constant",
        r"unexpected numeric constant",
        r"unexpected end of input",
        r"unexpected \)",
        r"unexpected \'",
    ]

    for pattern in compile_patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            return "compile_error"

    # Runtime errors: Execution crashes, missing packages
    runtime_patterns = [
        r"Error:",
        r"fatal error",
        r"could not find function",
        r"there is no package called",
        r"package .* not found",
        r"object .* not found",
        r"subscript out of bounds",
        r"index out of bounds",
        r"memory allocation",
        r"stack overflow",
    ]

    if returncode != 0:
        for pattern in runtime_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return "runtime_error"
        # Generic runtime error if returncode is non-zero but no specific pattern
        if combined.strip():
            return "runtime_error"

    # If we got here with non-zero returncode but no stderr, still runtime error
    if returncode != 0:
        return "runtime_error"

    return "success"


def run_r_code(code: str, data_dir: Path, output_dir: Path, timeout: int = 300) -> dict:
    """
    Run R code with timeout and error classification.

    Args:
        code: R code to execute
        data_dir: Directory containing data files
        output_dir: Directory for output files
        timeout: Execution timeout in seconds (default 300)

    Returns:
        dict with execution results including error_type classification
    """
    (data_dir / "solution.R").write_text(code)

    # Use Popen for proper timeout handling
    proc = subprocess.Popen(
        ["Rscript", "solution.R"],
        cwd=str(data_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timeout_expired = False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()  # Clear buffers
        stdout, stderr = "", "Timeout expired"
        timeout_expired = True

    output = stdout + stderr

    # Classify error type
    if timeout_expired:
        error_type = "runtime_error"
    else:
        error_type = classify_error(stderr, proc.returncode, stdout)

    # Extract raw score
    raw_score = 0.0
    match = re.search(r"CV_ACCURACY:\s*([0-9.]+)", output)
    if match:
        raw_score = float(match.group(1))

    # Check submission file
    submission_path = data_dir / "submission.csv"
    submission_ok = False
    if submission_path.exists():
        shutil.copy(submission_path, output_dir / "submission.csv")
        try:
            with open(submission_path) as f:
                h = f.readline().lower()
            submission_ok = "passengerid" in h and "survived" in h
        except Exception:
            pass

    # If code ran but no valid output, classify as wrong_answer
    if error_type == "success" and (not submission_ok or raw_score == 0.0):
        error_type = "wrong_answer"

    return {
        "runs": proc.returncode == 0 and not timeout_expired,
        "output": output,
        "raw_score": raw_score,
        "submission_ok": submission_ok,
        "error_type": error_type,
        "timeout": timeout_expired,
        "returncode": proc.returncode,
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description="Kaggle Benchmark Runner with APPS-style evaluation")
    p.add_argument(
        "--task",
        default="tasks/kaggle/kaggle_titanic_exploring-survival-on-the-titanic.json",
        help="Path to task JSON file",
    )
    p.add_argument("--model", default="zai/glm-5", help="LLM model to use for code generation")
    p.add_argument(
        "--data-dir", default="data/kaggle/titanic", help="Directory containing data files"
    )
    p.add_argument(
        "--output-dir", default="output/benchmark_titanic", help="Directory for output files"
    )
    p.add_argument("--code", help="Provide R code directly (skip generation)")
    p.add_argument(
        "--timeout", type=int, default=300, help="Execution timeout in seconds (default: 300)"
    )
    p.add_argument(
        "--expected-csv",
        type=str,
        default=None,
        help="Path to expected submission.csv for comparison",
    )
    args = p.parse_args()

    print("=" * 60)
    print("KAGGLE BENCHMARK (APPS-style Evaluation)")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = load_task(args.task)
    baselines = get_baselines(task)
    ref = task.get("reference_solution", {}).get("code", "")

    print(f"\nTask: {task.get('task_id')}")
    print(f"Baselines: random={baselines['random']:.0%}, top_kaggle={baselines['top_kaggle']:.0%}")
    print(f"Timeout: {args.timeout}s")

    # Generate code
    if args.code:
        code = args.code
    else:
        print(f"\n[1] Generating with {args.model}...")
        code = generate_r_code(task, args.model)
    (output_dir / "generated_code.R").write_text(code)

    # Check for dangerous functions
    print("\n[1.5] Checking for dangerous functions...")
    dangerous_warnings = check_dangerous_functions(code)
    if dangerous_warnings:
        print(f"    Found {len(dangerous_warnings)} warning(s):")
        for w in dangerous_warnings:
            print(f"      Line {w['line']}: {w['description']}")
    else:
        print("    No dangerous functions detected")

    # Run
    print(f"\n[2] Running (timeout={args.timeout}s)...")
    run_result = run_r_code(code, data_dir, output_dir, timeout=args.timeout)

    # CSV comparison if expected file provided
    csv_comparison = None
    if args.expected_csv:
        expected_path = Path(args.expected_csv)
        generated_path = output_dir / "submission.csv"
        print("\n[2.5] Comparing CSVs...")
        csv_comparison = compare_csv(generated_path, expected_path)
        print(f"    Match: {csv_comparison['match']}")
        print(f"    Header match: {csv_comparison['header_match']}")
        print(f"    Row count match: {csv_comparison['row_count_match']}")
        print(f"    Float match: {csv_comparison['float_match']}")

    # Grade
    execution_pass = (
        run_result["runs"] and run_result["submission_ok"] and run_result["raw_score"] > 0
    )
    raw = run_result["raw_score"]
    normalized = normalize_score(raw, baselines) if execution_pass else 0.0
    error_type = run_result["error_type"]

    # Cheating check
    cheating = check_cheating(code, ref)

    # Output
    print("\n[3] GRADING")
    print(f"    Execution: {'PASS' if execution_pass else 'FAIL'}")
    print(f"    Error Type: {error_type}")
    print(f"    Raw Score:  {raw:.2%}")
    print(f"    Normalized: {normalized:.2f} (0=random, 1=top)")
    print(
        f"    Cheating:   {cheating['similarity']:.1%} similarity"
        + (f" - {cheating['warning']}" if cheating["warning"] else " (OK)")
    )

    overall = "PASS" if execution_pass and normalized >= 0.3 else "FAIL"
    print(f"\n    Overall: {overall}")

    if not run_result["runs"] or run_result["timeout"]:
        print(f"\nError ({error_type}):")
        print(run_result["output"][-500:] if run_result["output"] else "No output")

    # Save
    results = {
        "task_id": task.get("task_id"),
        "model": args.model,
        "baselines": baselines,
        "grading": {
            "execution": "PASS" if execution_pass else "FAIL",
            "error_type": error_type,
            "raw_score": round(raw, 4),
            "normalized_score": round(normalized, 3),
            "timeout_seconds": args.timeout,
            "overall": overall,
        },
        "execution_details": {
            "returncode": run_result["returncode"],
            "timeout_expired": run_result["timeout"],
            "submission_ok": run_result["submission_ok"],
        },
        "dangerous_functions": dangerous_warnings,
        "cheating": cheating,
    }

    if csv_comparison:
        results["csv_comparison"] = csv_comparison

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
