#!/bin/bash
# Scheduled Task Mining Script for R Testing Tasks
# 
# This script mines merged PRs from configured R package repositories
# to extract high-quality testing tasks using LLM evaluation.
#
# Usage:
#   ./scripts/scheduled_mine.sh                    # Normal run
#   ./scripts/scheduled_mine.sh --dry-run          # Preview without changes
#   ./scripts/scheduled_mine.sh --since-days 60    # Look back 60 days
#
# Cron example (run monthly on 1st at 2 AM):
#   0 2 1 * * /home/m0hawk/Documents/trainR/scripts/scheduled_mine.sh
#
# Environment requirements:
#   - GITHUB_TOKEN: GitHub personal access token
#   - OPENAI_API_KEY or ANTHROPIC_API_KEY: LLM API key

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Date for logging and output organization
DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Directories
LOG_DIR="logs/mining"
OUTPUT_DIR="tasks/mined/${DATE}"
CONFIG_FILE="configs/repos_to_mine.yaml"

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${LOG_DIR}/mining_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check for required environment variables
    local missing=()
    
    if [ -z "$GITHUB_TOKEN" ]; then
        missing+=("GITHUB_TOKEN")
    fi
    
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        missing+=("OPENAI_API_KEY or ANTHROPIC_API_KEY")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log "ERROR: Missing required environment variables: ${missing[*]}"
        log "Please set these environment variables before running the script."
        exit 1
    fi
    
    # Check for required files
    if [ ! -f "$CONFIG_FILE" ]; then
        log "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Check for uv
    if ! command -v uv &> /dev/null; then
        log "ERROR: uv is not installed. Please install it first."
        exit 1
    fi
    
    log "Prerequisites check passed."
}

# Function to show system info
show_system_info() {
    log "=== System Information ==="
    log "Project directory: $PROJECT_DIR"
    log "Config file: $CONFIG_FILE"
    log "Output directory: $OUTPUT_DIR"
    log "Log file: $LOG_FILE"
    log "Python version: $(python3 --version 2>&1 || echo 'Not found')"
    log "uv version: $(uv --version 2>&1 || echo 'Not found')"
    log "=========================="
}

# Function to run the mining script
run_mining() {
    local extra_args=("$@")
    
    log "Starting PR mining..."
    log "Extra arguments: ${extra_args[*]:-none}"
    
    # Run the Python mining script
    uv run python scripts/mine_prs.py \
        --repos-file "$CONFIG_FILE" \
        --output "$OUTPUT_DIR" \
        --since-days "${SINCE_DAYS:-30}" \
        --min-quality "${MIN_QUALITY:-7}" \
        --max-prs "${MAX_PRS:-50}" \
        "${extra_args[@]}" \
        2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -ne 0 ]; then
        log "ERROR: Mining script failed with exit code $exit_code"
        return $exit_code
    fi
    
    return 0
}

# Function to generate summary
generate_summary() {
    log ""
    log "=== Mining Summary ==="
    
    # Count tasks created
    local task_count=$(find "$OUTPUT_DIR" -name "*.json" -type f 2>/dev/null | wc -l)
    log "Tasks collected: $task_count"
    
    # Show task types breakdown if jq is available
    if command -v jq &> /dev/null && [ $task_count -gt 0 ]; then
        log ""
        log "Task type breakdown:"
        for task_file in "$OUTPUT_DIR"/*.json; do
            if [ -f "$task_file" ]; then
                jq -r '.metadata.task_type' "$task_file" 2>/dev/null
            fi
        done | sort | uniq -c | sort -rn | while read count type; do
            log "  $type: $count"
        done
        
        log ""
        log "Difficulty breakdown:"
        for task_file in "$OUTPUT_DIR"/*.json; do
            if [ -f "$task_file" ]; then
                jq -r '.metadata.difficulty' "$task_file" 2>/dev/null
            fi
        done | sort | uniq -c | sort -rn | while read count diff; do
            log "  $diff: $count"
        done
    fi
    
    log ""
    log "Output directory: $(realpath "$OUTPUT_DIR")"
    log "Log file: $(realpath "$LOG_FILE")"
    log "======================"
}

# Function to optionally merge into main task set
merge_to_main() {
    if [ "$MERGE_TO_MAIN" = "true" ]; then
        log ""
        log "Merging tasks into main task set..."
        
        if [ -f "scripts/merge_tasks.py" ]; then
            uv run python scripts/merge_tasks.py \
                --source "$OUTPUT_DIR" \
                --dest "tasks/" \
                2>&1 | tee -a "$LOG_FILE"
        else
            log "WARNING: merge_tasks.py not found, skipping merge"
        fi
    fi
}

# Function to cleanup old logs (keep last 30 days)
cleanup_old_logs() {
    log "Cleaning up old logs..."
    find "$LOG_DIR" -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true
}

# Parse command line arguments
DRY_RUN=false
SINCE_DAYS=30
MIN_QUALITY=7
MAX_PRS=50
MERGE_TO_MAIN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --since-days)
            SINCE_DAYS="$2"
            shift 2
            ;;
        --min-quality)
            MIN_QUALITY="$2"
            shift 2
            ;;
        --max-prs)
            MAX_PRS="$2"
            shift 2
            ;;
        --merge)
            MERGE_TO_MAIN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Show what would be done without making changes"
            echo "  --since-days N     Look back N days for merged PRs (default: 30)"
            echo "  --min-quality N    Minimum quality score 1-10 (default: 7)"
            echo "  --max-prs N        Max PRs per repo (default: 50)"
            echo "  --merge            Merge collected tasks into main task set"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log "=========================================="
    log "   R Testing Task Mining - Started"
    log "=========================================="
    
    show_system_info
    check_prerequisites
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN MODE - No changes will be made"
        EXTRA_ARGS="--dry-run"
    else
        EXTRA_ARGS=""
    fi
    
    # Run the mining
    if run_mining $EXTRA_ARGS; then
        generate_summary
        merge_to_main
        cleanup_old_logs
    else
        log "ERROR: Mining failed"
        exit 1
    fi
    
    log ""
    log "=========================================="
    log "   R Testing Task Mining - Complete"
    log "=========================================="
}

# Run main function
main
