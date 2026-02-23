#!/bin/bash
# =============================================================================
# CI Smoke Test Suite
# =============================================================================
# Runs a comprehensive smoke test suite for CI pipelines.
# Includes: pytest, linting, type checking, contract validation, and smoke experiments.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#
# Usage:
#   ./scripts/ci_smoke.sh           # Run all checks
#   ./scripts/ci_smoke.sh --quick   # Skip smoke experiment (faster)
#   ./scripts/ci_smoke.sh --help    # Show help
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUICK_MODE=false
VERBOSE=false
START_TIME=$(date +%s)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track failures
declare -a FAILURES=()

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_check() {
    local name="$1"
    local cmd="$2"
    local start_time check_time
    
    echo ""
    echo "============================================================================="
    echo -e "${BLUE}Running: ${name}${NC}"
    echo "============================================================================="
    
    start_time=$(date +%s)
    
    # Run the command, capturing both stdout and stderr
    set +e
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    set -e
    
    check_time=$(($(date +%s) - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log_success "$name (took ${check_time}s)"
        if [ "$VERBOSE" = true ] && [ -n "$output" ]; then
            echo "$output"
        fi
    else
        log_error "$name (took ${check_time}s)"
        FAILURES+=("$name")
        echo "$output"
    fi
    
    return $exit_code
}

print_summary() {
    local end_time total_time
    
    end_time=$(date +%s)
    total_time=$((end_time - START_TIME))
    
    echo ""
    echo "============================================================================="
    echo "CI Smoke Test Summary"
    echo "============================================================================="
    echo ""
    
    if [ ${#FAILURES[@]} -eq 0 ]; then
        log_success "All checks passed in ${total_time}s"
        echo ""
        return 0
    else
        log_error "${#FAILURES[@]} check(s) failed:"
        for failure in "${FAILURES[@]}"; do
            echo "  - $failure"
        done
        echo ""
        log_error "CI failed after ${total_time}s"
        return 1
    fi
}

show_help() {
    cat << EOF
CI Smoke Test Suite

Usage: $0 [OPTIONS]

Options:
    -q, --quick      Quick mode - skip smoke experiment (faster, ~2 min)
    -v, --verbose    Show output from successful checks
    -h, --help       Show this help message

Checks performed:
    1. pytest        - Unit tests (quick mode)
    2. ruff check    - Linting
    3. ty check      - Type checking
    4. contracts     - Config contract validation
    5. schema        - Schema validation for artifacts
    6. smoke-exp     - Smoke experiment (skipped in quick mode)

Exit codes:
    0 - All checks passed
    1 - One or more checks failed

Examples:
    $0                    # Full CI suite
    $0 --quick            # Skip smoke experiment
    $0 -v                 # Verbose output

EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Change to project root
# =============================================================================

cd "$PROJECT_ROOT"

echo ""
echo "============================================================================="
echo "CI Smoke Test Suite"
echo "============================================================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Quick mode:   $QUICK_MODE"
echo "Verbose:      $VERBOSE"
echo ""

# =============================================================================
# Check 1: Pytest (quick mode)
# =============================================================================

run_check "pytest (quick mode)" \
    "uv run pytest tests/ -v --tb=short -x -q 2>/dev/null || uv run pytest tests/ -v --tb=short -x"

# =============================================================================
# Check 2: Ruff Linting
# =============================================================================

run_check "ruff check" \
    "uv run ruff check ."

# =============================================================================
# Check 3: Type Checking
# =============================================================================

run_check "ty check" \
    "uv run ty check ."

# =============================================================================
# Check 4: Config Contract Validation
# =============================================================================

run_check "validate_contracts" \
    "uv run python scripts/validate_contracts.py"

# =============================================================================
# Check 5: Schema Validation
# =============================================================================

# Validate any existing result files against schemas
run_check "schema validation" \
    "uv run python -c \"
import sys
import json
from pathlib import Path

# Import schema validators
sys.path.insert(0, str(Path('$PROJECT_ROOT')))
from bench.schema.v1 import ManifestV1, ResultV1

errors = []

# Check for existing manifests
results_dir = Path('results')
if results_dir.exists():
    for manifest_file in results_dir.glob('**/manifest.json'):
        try:
            data = json.loads(manifest_file.read_text())
            ManifestV1.model_validate(data)
            print(f'  Valid: {manifest_file}')
        except Exception as e:
            errors.append(f'{manifest_file}: {e}')

    # Check for results files
    for results_file in results_dir.glob('**/results.jsonl'):
        try:
            for i, line in enumerate(results_file.read_text().strip().split('\n')):
                if line:
                    data = json.loads(line)
                    ResultV1.model_validate(data)
            print(f'  Valid: {results_file}')
        except Exception as e:
            errors.append(f'{results_file}: {e}')

if errors:
    for err in errors:
        print(f'  ERROR: {err}', file=sys.stderr)
    sys.exit(1)

print('  No schema validation errors')
\""

# =============================================================================
# Check 6: Smoke Experiment (optional in quick mode)
# =============================================================================

if [ "$QUICK_MODE" = true ]; then
    log_warn "Skipping smoke experiment (--quick mode)"
else
    # Use the smoke config with minimal tasks
    SMOKE_CONFIG="configs/experiments/r_bench_smoke.yaml"
    
    if [ -f "$SMOKE_CONFIG" ]; then
        run_check "smoke experiment" \
            "uv run python scripts/run_experiment.py --config $SMOKE_CONFIG --dry-run"
    else
        log_warn "Smoke config not found: $SMOKE_CONFIG (skipping)"
    fi
fi

# =============================================================================
# Print Summary
# =============================================================================

print_summary
