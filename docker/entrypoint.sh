#!/bin/bash
set -e

# Entrypoint for trainr-eval Docker container
# ============================================
# Expects environment variables:
# - Z_AI_API_KEY or ANTHROPIC_API_KEY: API key for the LLM provider
# - SKILL_CONTENT: The skill markdown content (base64 encoded)
# - TASK_CONTENT: The task instruction (base64 encoded)
# - PACKAGE_PATH: Path to the R package to test (default: /workspace/packages/cli)

# Configuration
PACKAGE_PATH="${PACKAGE_PATH:-/workspace/packages/cli}"
SKILL_DIR="/workspace/.claude/skills/testing-r-packages"
WORKSPACE="/workspace"

# Logging function
log() {
    echo "[$(date -Iseconds)] $1" >&2
}

log "Starting trainr-eval container..."

# Validate required environment variables
if [ -z "$Z_AI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    log "ERROR: Either Z_AI_API_KEY or ANTHROPIC_API_KEY must be set"
    exit 1
fi

# Set API key (Z_AI_API_KEY takes precedence)
export ANTHROPIC_API_KEY="${Z_AI_API_KEY:-$ANTHROPIC_API_KEY}"

# Setup workspace skill file
mkdir -p "$SKILL_DIR"
if [ -n "$SKILL_CONTENT" ]; then
    echo "$SKILL_CONTENT" | base64 -d > "$SKILL_DIR/SKILL.md" 2>/dev/null || \
        echo "$SKILL_CONTENT" > "$SKILL_DIR/SKILL.md"
    log "Skill file written"
fi

# Setup task file
if [ -n "$TASK_CONTENT" ]; then
    echo "$TASK_CONTENT" | base64 -d > "$WORKSPACE/TASK.md" 2>/dev/null || \
        echo "$TASK_CONTENT" > "$WORKSPACE/TASK.md"
    log "Task file written"
fi

# Build the agent prompt
# Derive the package name from PACKAGE_PATH for the prompt
PKG_REL_PATH=$(echo "$PACKAGE_PATH" | sed 's|/workspace/||')
AGENT_PROMPT="Read .claude/skills/testing-r-packages/SKILL.md to understand testing patterns. Read TASK.md for the specific task. Explore ${PKG_REL_PATH}/ to understand the codebase. Write appropriate tests to ${PKG_REL_PATH}/tests/testthat/test-generated.R. Run devtools::test() to verify your tests work."

# Override prompt if provided
if [ -n "$CUSTOM_PROMPT" ]; then
    if echo "$CUSTOM_PROMPT" | base64 -d > /dev/null 2>&1; then
        AGENT_PROMPT=$(echo "$CUSTOM_PROMPT" | base64 -d)
    else
        AGENT_PROMPT="$CUSTOM_PROMPT"
    fi
fi

# Write agent prompt to file (avoids quoting issues with su -c)
echo "$AGENT_PROMPT" > /tmp/agent_prompt.txt
chown trainr:trainr /tmp/agent_prompt.txt

# Write env vars for trainr user
# Model selection via LLM_MODEL env var (default: glm-4.5)
MODEL_SONNET="${LLM_MODEL:-glm-4.5}"
cat > /tmp/trainr_env.sh <<EOF
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export MODEL_SONNET="$MODEL_SONNET"
export PATH="/home/trainr/.local/bin:\$PATH"
EOF
chown trainr:trainr /tmp/trainr_env.sh

# Create cc-mirror variant as trainr user (Claude CLI refuses root)
log "Setting up cc-mirror trainr-eval variant with model: $MODEL_SONNET"
su - trainr -s /bin/bash -c "
    source /tmp/trainr_env.sh
    echo yes | npx cc-mirror quick --provider zai --name trainr-eval --api-key \"\$ANTHROPIC_API_KEY\" --model-sonnet \"\$MODEL_SONNET\" --no-tui 2>/dev/null || {
        mkdir -p /home/trainr/.local/bin
        printf '#!/bin/bash\nexec claude \"\$@\"\n' > /home/trainr/.local/bin/trainr-eval
        chmod +x /home/trainr/.local/bin/trainr-eval
    }
"

# Pre-flight check: verify the agent can execute
log "Running pre-flight check..."
if ! su - trainr -s /bin/bash -c "source /tmp/trainr_env.sh && command -v trainr-eval" &>/dev/null; then
    log "ERROR: trainr-eval binary not found after setup"
    exit 1
fi

log "Running trainr-eval agent..."

# Run the agent as non-root user (Claude CLI requires this for --dangerously-skip-permissions)
cd "$WORKSPACE"
set +e
AGENT_OUTPUT=$(su - trainr -s /bin/bash -c "
    source /tmp/trainr_env.sh
    cd /workspace
    PROMPT=\$(cat /tmp/agent_prompt.txt)
    trainr-eval --dangerously-skip-permissions -p "$PROMPT" --max-turns 50 2>&1
")
AGENT_EXIT_CODE=$?
set -e

log "Agent completed with exit code: $AGENT_EXIT_CODE"

# Write agent output to file for R to read (handles special characters safely)
echo "$AGENT_OUTPUT" > /tmp/agent_output.txt

# Run final evaluation with testthat
log "Running final testthat evaluation..."

cd "$PACKAGE_PATH"

# Ensure testthat directory structure exists
mkdir -p tests/testthat

set +e
EVAL_OUTPUT=$(Rscript -e "
    library(jsonlite)

    # Read agent output from file (handles special characters safely)
    agent_output <- tryCatch({
        paste(readLines('/tmp/agent_output.txt', warn = FALSE), collapse = '\n')
    }, error = function(e) {
        'Failed to read agent output'
    })

    # Check if test file exists
    test_file <- 'tests/testthat/test-generated.R'
    if (!file.exists(test_file)) {
        cat(toJSON(list(
            success = FALSE,
            error = 'Agent did not create test file',
            agent_output = agent_output,
            results = list(),
            summary = list(
                total_tests = 0,
                total_passed = 0,
                total_failed = 0,
                total_skipped = 0,
                total_warnings = 0,
                success = FALSE
            )
        ), auto_unbox = TRUE, pretty = TRUE))
        quit(save = 'no', status = 1)
    }

    tryCatch({
        # Load the package
        devtools::load_all('.', quiet = TRUE)

        # Run tests with ListReporter for structured results
        reporter <- testthat::ListReporter\$new()
        result <- testthat::test_file(test_file, reporter = reporter)

        # as.data.frame works reliably across testthat versions
        df <- as.data.frame(result)

        total_tests <- nrow(df)
        total_passed <- sum(df\$passed, na.rm = TRUE)
        total_failed <- sum(df\$failed, na.rm = TRUE)
        total_skipped <- sum(df\$skipped, na.rm = TRUE)
        total_warnings <- sum(df\$warning, na.rm = TRUE)
        all_pass <- total_failed == 0 && total_tests > 0

        test_details <- lapply(seq_len(nrow(df)), function(i) {
            list(
                test = df\$test[i],
                passed = as.integer(df\$passed[i]),
                failed = as.integer(df\$failed[i]),
                skipped = as.integer(df\$skipped[i]),
                warning = as.integer(df\$warning[i])
            )
        })

        output <- list(
            success = all_pass,
            results = test_details,
            summary = list(
                total_tests = total_tests,
                total_passed = total_passed,
                total_failed = total_failed,
                total_skipped = total_skipped,
                total_warnings = total_warnings,
                success = all_pass
            ),
            agent_output = agent_output
        )

        cat(toJSON(output, auto_unbox = TRUE, pretty = TRUE))
    }, error = function(e) {
        cat(toJSON(list(
            success = FALSE,
            error = paste('Evaluation error:', conditionMessage(e)),
            agent_output = agent_output,
            results = list(),
            summary = list(
                total_tests = 0,
                total_passed = 0,
                total_failed = 0,
                total_skipped = 0,
                total_warnings = 0,
                success = FALSE
            )
        ), auto_unbox = TRUE, pretty = TRUE))
    })
" 2>&1)
EVAL_EXIT_CODE=$?
set -e

log "Evaluation completed with exit code: $EVAL_EXIT_CODE"

# Output the evaluation results (to stdout for Docker caller to capture)
echo "$EVAL_OUTPUT"

# Determine final exit code
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    if echo "$EVAL_OUTPUT" | grep -q '"success"[[:space:]]*:[[:space:]]*true'; then
        log "All tests passed!"
        exit 0
    else
        log "Some tests failed"
        exit 1
    fi
else
    log "Evaluation script failed"
    exit $EVAL_EXIT_CODE
fi
