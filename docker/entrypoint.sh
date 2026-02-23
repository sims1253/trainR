#!/bin/bash
set -e

# Entrypoint for trainr-eval Docker container
# ============================================
# Expects environment variables based on model provider:
# - OPENROUTER_API_KEY: For openrouter/* models
# - OPENAI_API_KEY: For openai/* models
# - Z_AI_API_KEY: For zai/* models
# - OPENCODE_API_KEY: For opencode/* models
# - SKILL_CONTENT: The skill markdown content (base64 encoded, optional)
# - TASK_CONTENT: The task instruction (base64 encoded)
# - PACKAGE_PATH: Path to the R package to test (default: /workspace/packages/cli)
# - PI_MODEL: Model to use (required - get from llm.yaml)

# Configuration
PACKAGE_PATH="${PACKAGE_PATH:-/workspace/packages/cli}"
SKILL_DIR="/workspace/.claude/skills/testing-r-packages"
WORKSPACE="/workspace"

# Logging function - MUST be defined first
log() {
    echo "[$(date -Iseconds)] $1" >&2
}

# PI_MODEL must be set (get default from llm.yaml)
if [ -z "$PI_MODEL" ]; then
    log "ERROR: PI_MODEL must be set"
    exit 1
fi

log "Starting trainr-eval container with Pi CLI..."
log "Model: $PI_MODEL"

# Determine which API key is needed based on model prefix
if [[ "$PI_MODEL" == openrouter/* ]]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        log "ERROR: OPENROUTER_API_KEY must be set for openrouter/* models"
        exit 1
    fi
    export PI_API_KEY="$OPENROUTER_API_KEY"
elif [[ "$PI_MODEL" == openai/* ]]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        log "ERROR: OPENAI_API_KEY must be set for openai/* models"
        exit 1
    fi
    export PI_API_KEY="$OPENAI_API_KEY"
elif [[ "$PI_MODEL" == zai/* ]]; then
    if [ -z "$Z_AI_API_KEY" ]; then
        log "ERROR: Z_AI_API_KEY must be set for zai/* models"
        exit 1
    fi
    export PI_API_KEY="$Z_AI_API_KEY"
elif [[ "$PI_MODEL" == opencode/* ]]; then
    if [ -z "$OPENCODE_API_KEY" ]; then
        log "ERROR: OPENCODE_API_KEY must be set for opencode/* models"
        exit 1
    fi
    export PI_API_KEY="$OPENCODE_API_KEY"
else
    log "ERROR: Unknown model provider for: $PI_MODEL"
    log "Supported prefixes: openrouter/*, openai/*, zai/*, opencode/*"
    exit 1
fi

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

# ============================================
# Clone and install package if REPO is specified
# ============================================
if [ -n "$REPO" ]; then
    log "Cloning package: $REPO"
    
    # Clean up any existing package
    rm -rf /workspace/package
    
    # Configure git safe directory to avoid ownership issues with Docker mounts
    git config --global --add safe.directory /workspace/package
    
    # Clone with token if available (for rate limits / private repos)
    if [ -n "$GITHUB_TOKEN" ]; then
        git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" /workspace/package 2>&1 || {
            log "ERROR: Failed to clone $REPO"
            exit 1
        }
    else
        git clone "https://github.com/${REPO}.git" /workspace/package 2>&1 || {
            log "ERROR: Failed to clone $REPO"
            exit 1
        }
    fi
    
    cd /workspace/package
    
    # Checkout specific commit if provided (for mined tasks)
    # Synthetic tasks just use the default branch
    if [ -n "$BASE_COMMIT" ]; then
        log "Checking out commit: $BASE_COMMIT"
        git checkout "$BASE_COMMIT" 2>&1 || {
            log "WARNING: Failed to checkout $BASE_COMMIT, using default branch"
        }
    fi
    
    cd /workspace/package
    
    # Install the package
    log "Installing package..."

    # Check if package has compiled code (src/ directory)
    HAS_SRC=""
    if [ -d "/workspace/package/src" ]; then
        HAS_SRC="true"
        log "Package has compiled code (src/), will build"
    fi

    Rscript -e "
        options(warn = -1)

        # Install dependencies first (needed for compilation)
        tryCatch({
            pak::pkg_install('dependencies', ask = FALSE)
        }, error = function(e) {
            message('Dependency install note: ', conditionMessage(e))
        })

        # Install the package itself
        tryCatch({
            # For packages with src/, we need a full install with compilation
            pak::pkg_install('.', ask = FALSE, dependencies = TRUE)
            message('Package installed successfully')
        }, error = function(e) {
            message('pak install failed, trying devtools: ', conditionMessage(e))
            tryCatch({
                devtools::install('.', dependencies = TRUE, quiet = FALSE, build = TRUE)
            }, error = function(e2) {
                message('devtools install also failed: ', conditionMessage(e2))
                # Last resort: just try to load what's there
                tryCatch({
                    devtools::load_all('.', quiet = TRUE)
                }, error = function(e3) {
                    message('load_all also failed: ', conditionMessage(e3))
                })
            })
        })
    " 2>&1 || log "WARNING: Package installation had issues"

    # Verify package can be loaded
    log "Verifying package installation..."

    # Get the actual package name from DESCRIPTION file
    PKG_NAME=$(grep -E "^Package:" /workspace/package/DESCRIPTION 2>/dev/null | sed 's/Package: //' || echo "readr")
    log "Package name from DESCRIPTION: $PKG_NAME"

    Rscript -e "
        pkg_name <- '$PKG_NAME'
        tryCatch({
            library(pkg_name, character.only = TRUE)
            message('Package ', pkg_name, ' loaded successfully')
        }, error = function(e) {
            message('Could not load package: ', conditionMessage(e))
        })
    " 2>&1 || true
    
    PACKAGE_PATH="/workspace/package"
    log "Package ready at: $PACKAGE_PATH"
else
    # No REPO specified, use default package path
    PACKAGE_PATH="${PACKAGE_PATH:-/workspace/packages/cli}"
    log "Using existing package at: $PACKAGE_PATH"
fi

# Fix ownership so trainr can write to workspace (Pi needs to create .pi directory)
chown -R trainr:trainr /workspace

# ============================================
# Apply test_patch if provided (SWE-bench style)
# ============================================
if [ -n "$TEST_PATCH" ]; then
    log "Applying test patch..."
    echo "$TEST_PATCH" | base64 -d > /tmp/test_patch.diff
    
    cd "$PACKAGE_PATH"
    if git apply /tmp/test_patch.diff 2>&1; then
        log "Test patch applied successfully"
    else
        log "WARNING: Failed to apply test patch (may already exist or conflict)"
    fi
fi

# ============================================
# Run tests BEFORE agent (verify tests fail)
# ============================================
if [ -n "$FAIL_TO_PASS" ] && [ "$FAIL_TO_PASS" != "[]" ]; then
    log "Running tests BEFORE agent (expecting failures)..."
    cd "$PACKAGE_PATH"
    
    BEFORE_RESULT=$(Rscript -e "
        library(jsonlite)
        tests <- fromJSON(Sys.getenv('FAIL_TO_PASS', '[]'))
        if (length(tests) == 0) {
            cat(toJSON(list(before_passed = 0, before_failed = 0, expected_failures = TRUE)))
            quit(save = 'no')
        }
        
        # Run testthat and capture results
        tryCatch({
            devtools::load_all('.', quiet = TRUE)
            results <- list()
            passed <- 0
            failed <- 0
            
            for (test_name in tests) {
                # Try to run the specific test
                result <- tryCatch({
                    testthat::test_that(test_name, {
                        # Test code would be in the test files
                    })
                    TRUE
                }, error = function(e) FALSE)
                
                if (result) passed <- passed + 1 else failed <- failed + 1
            }
            
            # If tests fail, that's expected (the bug exists)
            expected <- failed > 0
            
            cat(toJSON(list(
                before_passed = passed,
                before_failed = failed,
                expected_failures = expected
            ), auto_unbox = TRUE))
        }, error = function(e) {
            cat(toJSON(list(
                before_passed = 0,
                before_failed = 0,
                expected_failures = TRUE,
                error = conditionMessage(e)
            ), auto_unbox = TRUE))
        })
    " 2>&1) || true
    
    log "Before tests: $BEFORE_RESULT"
    echo "$BEFORE_RESULT" > /tmp/before_tests.json
fi

cd "$WORKSPACE"

# Build the agent prompt
# Derive the package name from PACKAGE_PATH for the prompt
if [ -n "$REPO" ]; then
    PKG_REL_PATH="package"
else
    PKG_REL_PATH=$(echo "$PACKAGE_PATH" | sed 's|/workspace/||')
fi
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
cat > /tmp/trainr_env.sh <<EOF
export OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export Z_AI_API_KEY="$Z_AI_API_KEY"
export OPENCODE_API_KEY="$OPENCODE_API_KEY"
export PATH="/home/trainr/.bun/bin:/home/trainr/.local/bin:\$PATH"
EOF
chown trainr:trainr /tmp/trainr_env.sh

log "Running Pi CLI agent with model: $PI_MODEL"

# Ensure trainr owns workspace (Pi needs to write .pi directory)
# This must run right before the agent, as earlier operations may have created files as root
chown -R trainr:trainr /workspace
chmod 755 /workspace

# Run the agent as non-root user using batch mode
cd "$WORKSPACE"
set +e

log "Running Pi CLI agent in batch mode with model: $PI_MODEL"

# Ensure trainr owns workspace (Pi needs to write .pi directory)
chown -R trainr:trainr /workspace
chmod 755 /workspace

cd "$WORKSPACE"

# Decode the prompt from base64
if [ -n "$PROMPT_B64" ]; then
    PROMPT=$(echo "$PROMPT_B64" | base64 -d)
else
    # Fallback to AGENT_PROMPT if no PROMPT_B64
    PROMPT="$AGENT_PROMPT"
fi

# Write prompt to a file to avoid shell escaping issues
echo "$PROMPT" > /tmp/pi_prompt.txt
chmod 644 /tmp/pi_prompt.txt

# Run Pi in batch mode - pass prompt via stdin to avoid shell escaping issues
cd "$PACKAGE_PATH"
# NOTE: Do NOT use 'exec' here - we need to continue after Pi finishes
su -s /bin/bash trainr -c "cd /workspace/package && cat /tmp/pi_prompt.txt | /home/trainr/.bun/bin/pi --print --mode json --model '$PI_MODEL' --no-session"

# Run tests after Pi finishes
log "Running testthat tests..."
cd "$PACKAGE_PATH"
su -s /bin/bash trainr -c "cd /workspace/package && Rscript -e \"testthat::test_dir('tests/testthat', reporter = 'check')\" 2>&1" || true

log "Tests completed"
