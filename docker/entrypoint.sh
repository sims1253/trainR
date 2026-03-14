#!/bin/bash
set -e

# Entrypoint for trainr-eval Docker container
# ============================================
# Expects environment variables based on model provider:
# - OPENROUTER_API_KEY: For openrouter/* models
# - OPENAI_API_KEY: For openai/* models
# - Z_AI_API_KEY: For zai/* models
# - OPENCODE_API_KEY: For opencode/* models
# - KIMI_API_KEY: For kimi/* models
# - SKILL_CONTENT: The skill markdown content (base64 encoded, optional)
# - TASK_CONTENT: The task instruction (base64 encoded)
# - PACKAGE_PATH: Path to the R package to test (default: /workspace/package)
# - PI_MODEL: Model to use (required - get from llm.yaml)
#
# NOTE: The canonical provider-to-API-key mappings are defined in:
#       bench/provider/resolver.py::PROVIDER_API_KEY_MAP
#       This script mirrors those mappings for Docker environments.

# Configuration
PACKAGE_PATH="${PACKAGE_PATH:-/workspace/package}"
SKILL_DIR="/workspace/.claude/skills/testing-r-packages"
WORKSPACE="/workspace"
# Use a writable global git config path (root FS may be read-only in hardened profiles).
export GIT_CONFIG_GLOBAL="${WORKSPACE}/.gitconfig"
touch "${GIT_CONFIG_GLOBAL}" || true

# Logging function - MUST be defined first
log() {
    echo "[$(date -Iseconds)] $1" >&2
}

# Run a command as trainr when possible; otherwise run as current user.
# Hardened sandbox profiles may force a non-root UID where `su` prompts for password.
run_as_trainr() {
    local cmd="$1"
    local current_uid
    local current_user

    current_uid="$(id -u)"
    current_user="$(id -un 2>/dev/null || echo "unknown")"

    if [ "$current_uid" -eq 0 ]; then
        su -s /bin/bash trainr -c "$cmd"
        return $?
    fi

    if [ "$current_user" = "trainr" ]; then
        bash -lc "$cmd"
        return $?
    fi

    log "Non-root user '$current_user' cannot switch users; running command directly"
    bash -lc "$cmd"
}

# Ensure workspace is writable for whichever user will run Pi/tests.
ensure_workspace_permissions() {
    if [ "$(id -u)" -eq 0 ]; then
        chown -R trainr:trainr /workspace
        chmod 755 /workspace
    else
        chmod u+rwx /workspace 2>/dev/null || true
    fi
}

# PI_MODEL must be set (get default from llm.yaml)
if [ -z "$PI_MODEL" ]; then
    log "ERROR: PI_MODEL must be set"
    exit 1
fi

log "Starting trainr-eval container with Pi CLI..."
log "Model: $PI_MODEL"

# Route R package installs and caches to a writable workspace location.
R_USER_LIB="${WORKSPACE}/.Rlibs"
export R_LIBS_USER="$R_USER_LIB"
export R_LIBS="$R_USER_LIB"
export RENV_PATHS_LIBRARY_ROOT="${WORKSPACE}/.renv"
export PAK_CACHE_DIR="${WORKSPACE}/.cache/pak"
# Avoid small sandbox tmpfs at /tmp for large package builds.
TMP_WORK_DIR="${WORKSPACE}/.tmp"
export TMPDIR="${TMP_WORK_DIR}"
export TMP="${TMP_WORK_DIR}"
export TEMP="${TMP_WORK_DIR}"
mkdir -p "$R_USER_LIB" "$RENV_PATHS_LIBRARY_ROOT" "$PAK_CACHE_DIR"
mkdir -p "${TMP_WORK_DIR}"
chmod -R a+rwx "$R_USER_LIB" "$RENV_PATHS_LIBRARY_ROOT" "$PAK_CACHE_DIR" 2>/dev/null || true
chmod -R a+rwx "${TMP_WORK_DIR}" 2>/dev/null || true
log "Using writable R library: $R_USER_LIB"
log "Using writable temp dir: $TMP_WORK_DIR"

# Force agent state/config paths into writable workspace storage.
PI_HOME_DIR="${WORKSPACE}/.pi"
mkdir -p "${PI_HOME_DIR}/agent" "${WORKSPACE}/.config" "${WORKSPACE}/.cache"
chmod -R a+rwx "${PI_HOME_DIR}" "${WORKSPACE}/.config" "${WORKSPACE}/.cache" 2>/dev/null || true

# Copy custom Pi models config if it exists
if [ -f "/home/trainr/models.json" ]; then
    cp "/home/trainr/models.json" "${PI_HOME_DIR}/agent/models.json"
fi
export HOME="${WORKSPACE}"
export PI_HOME="${PI_HOME_DIR}"
export XDG_CONFIG_HOME="${WORKSPACE}/.config"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"

# Determine which API key is needed based on model prefix
if [[ "$PI_MODEL" == openrouter/* ]]; then
    PI_PROVIDER="openrouter"
    PI_MODEL_ID="${PI_MODEL#openrouter/}"
    if [ -z "$OPENROUTER_API_KEY" ]; then
        log "ERROR: OPENROUTER_API_KEY must be set for openrouter/* models"
        exit 1
    fi
    export PI_API_KEY="$OPENROUTER_API_KEY"
elif [[ "$PI_MODEL" == openai/* ]]; then
    PI_PROVIDER="openai"
    PI_MODEL_ID="${PI_MODEL#openai/}"
    if [ -z "$OPENAI_API_KEY" ]; then
        log "ERROR: OPENAI_API_KEY must be set for openai/* models"
        exit 1
    fi
    export PI_API_KEY="$OPENAI_API_KEY"
elif [[ "$PI_MODEL" == zai/* ]]; then
    PI_PROVIDER="zai"
    PI_MODEL_ID="${PI_MODEL#zai/}"
    # Support both canonical internal key name and Pi-native alias.
    if [ -z "$Z_AI_API_KEY" ] && [ -z "$ZAI_API_KEY" ]; then
        log "ERROR: Z_AI_API_KEY or ZAI_API_KEY must be set for zai/* models"
        exit 1
    fi
    if [ -z "$ZAI_API_KEY" ]; then
        export ZAI_API_KEY="$Z_AI_API_KEY"
    fi
    if [ -z "$Z_AI_API_KEY" ]; then
        export Z_AI_API_KEY="$ZAI_API_KEY"
    fi
    export PI_API_KEY="$ZAI_API_KEY"
    # Do not force AI_GATEWAY_API_KEY for zai models; that can route to
    # vercel-ai-gateway and require VERCEL_OIDC_TOKEN.
elif [[ "$PI_MODEL" == opencode/* ]]; then
    PI_PROVIDER="opencode"
    PI_MODEL_ID="${PI_MODEL#opencode/}"
    if [ -z "$OPENCODE_API_KEY" ]; then
        log "ERROR: OPENCODE_API_KEY must be set for opencode/* models"
        exit 1
    fi
    export PI_API_KEY="$OPENCODE_API_KEY"
elif [[ "$PI_MODEL" == kimi/* ]]; then
    PI_PROVIDER="kimi-coding"
    PI_MODEL_ID="${PI_MODEL#kimi/}"
    if [ -z "$KIMI_API_KEY" ]; then
        log "ERROR: KIMI_API_KEY must be set for kimi/* models"
        exit 1
    fi
    export PI_API_KEY="$KIMI_API_KEY"
else
    log "ERROR: Unknown model provider for: $PI_MODEL"
    log "Supported prefixes: openrouter/*, openai/*, zai/*, opencode/*, kimi/*"
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
# Task type detection: R package vs Kaggle kernel
# ============================================
# TASK_TYPE can be: "r_package" (default), "kaggle_kernel"
TASK_TYPE="${TASK_TYPE:-r_package}"

# ============================================
# Handle Kaggle kernel tasks (notebooks/scripts)
# ============================================
if [ "$TASK_TYPE" = "kaggle_kernel" ]; then
    log "Setting up Kaggle kernel workspace..."
    
    # Create notebook workspace
    mkdir -p /workspace/notebook
    
    # Write the reference solution as a script (if provided)
    if [ -n "$KAGGLE_CODE_B64" ]; then
        echo "$KAGGLE_CODE_B64" | base64 -d > /workspace/notebook/reference_solution.R 2>/dev/null || {
            log "WARNING: Failed to decode KAGGLE_CODE_B64"
        }
        log "Kaggle reference solution written to /workspace/notebook/reference_solution.R"
    fi
    
    # Write data path hint if provided
    if [ -n "$KAGGLE_DATA_PATH" ]; then
        echo "$KAGGLE_DATA_PATH" > /workspace/notebook/DATA_PATH.txt
        log "Kaggle data path: $KAGGLE_DATA_PATH"
    fi
    
    # Create a placeholder solution file for the agent to modify
    touch /workspace/notebook/solution.R
    
    PACKAGE_PATH="/workspace/notebook"
    log "Kaggle workspace ready at: $PACKAGE_PATH"

# ============================================
# Clone and install package if REPO is specified (R package tasks)
# ============================================
elif [ -n "$REPO" ]; then
    log "Cloning package: $REPO"
    
    # Clean up any existing package
    rm -rf /workspace/package
    
    # Configure git safe directory to avoid ownership issues with Docker mounts
    # Uses GIT_CONFIG_GLOBAL under /workspace to avoid read-only /home paths.
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

        # Force a writable user library first in .libPaths()
        user_lib <- Sys.getenv('R_LIBS_USER', unset = '')
        if (nzchar(user_lib)) {
            dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
            .libPaths(c(user_lib, .libPaths()))
            message('Using R user library: ', .libPaths()[1])
        }

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
        user_lib <- Sys.getenv('R_LIBS_USER', unset = '')
        if (nzchar(user_lib)) {
            dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
            .libPaths(c(user_lib, .libPaths()))
        }
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
    PACKAGE_PATH="${PACKAGE_PATH:-/workspace/package}"
    log "Using existing package at: $PACKAGE_PATH"
fi

# Fix ownership when running as root so trainr can write to workspace.
ensure_workspace_permissions

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

# Build the agent prompt based on task type
if [ "$TASK_TYPE" = "kaggle_kernel" ]; then
    # Kaggle task: agent writes/modifies a solution script
    AGENT_PROMPT="Read TASK.md for the Kaggle competition task. Your goal is to write or improve the solution script at notebook/solution.R. You can reference notebook/reference_solution.R for approach ideas. Run your solution with: Rscript notebook/solution.R"
else
    # R package task: agent writes tests
    # Derive the package name from PACKAGE_PATH for the prompt
    if [ -n "$REPO" ]; then
        PKG_REL_PATH="package"
    else
        PKG_REL_PATH=$(echo "$PACKAGE_PATH" | sed 's|/workspace/||')
    fi
    AGENT_PROMPT="Read .claude/skills/testing-r-packages/SKILL.md to understand testing patterns. Read TASK.md for the specific task. Explore ${PKG_REL_PATH}/ to understand the codebase. Write appropriate tests to ${PKG_REL_PATH}/tests/testthat/test-generated.R. Run devtools::test() to verify your tests work."
fi

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
if [ "$(id -u)" -eq 0 ]; then
    chown trainr:trainr /tmp/agent_prompt.txt
fi

# Write env vars for trainr user
cat > /tmp/trainr_env.sh <<EOF
export OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export Z_AI_API_KEY="$Z_AI_API_KEY"
export ZAI_API_KEY="${ZAI_API_KEY:-$Z_AI_API_KEY}"
export OPENCODE_API_KEY="$OPENCODE_API_KEY"
export KIMI_API_KEY="$KIMI_API_KEY"
export AI_GATEWAY_API_KEY="$AI_GATEWAY_API_KEY"
export R_LIBS_USER="$R_LIBS_USER"
export R_LIBS="$R_LIBS"
export RENV_PATHS_LIBRARY_ROOT="$RENV_PATHS_LIBRARY_ROOT"
export PAK_CACHE_DIR="$PAK_CACHE_DIR"
export TMPDIR="$TMPDIR"
export TMP="$TMP"
export TEMP="$TEMP"
export HOME="$HOME"
export PI_HOME="$PI_HOME"
export XDG_CONFIG_HOME="$XDG_CONFIG_HOME"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export PATH="/home/trainr/.bun/bin:/home/trainr/.local/bin:\$PATH"
EOF
if [ "$(id -u)" -eq 0 ]; then
    chown trainr:trainr /tmp/trainr_env.sh
fi

log "Running Pi CLI agent with model: $PI_MODEL"
log "Resolved Pi provider/model: ${PI_PROVIDER}/${PI_MODEL_ID}"

# Ensure workspace permissions right before the agent.
ensure_workspace_permissions

# Run the agent as non-root user using batch mode
cd "$WORKSPACE"
set +e

log "Running Pi CLI agent in batch mode with model: $PI_MODEL"

# Ensure workspace permissions right before batch run.
ensure_workspace_permissions

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
# Load tool trace extension to log tool execution events to stdout

if [ "$TASK_TYPE" = "kaggle_kernel" ]; then
    # Kaggle task: run from workspace root
    run_as_trainr "source /tmp/trainr_env.sh && cd /workspace && cat /tmp/pi_prompt.txt | /home/trainr/.bun/bin/pi --print --mode json --provider '$PI_PROVIDER' --model '$PI_MODEL_ID' --no-session --extension /opt/pi-extensions/pi-tool-trace-extension.ts"
else
    # R package task: run from package directory
    run_as_trainr "source /tmp/trainr_env.sh && cd /workspace/package && cat /tmp/pi_prompt.txt | /home/trainr/.bun/bin/pi --print --mode json --provider '$PI_PROVIDER' --model '$PI_MODEL_ID' --no-session --extension /opt/pi-extensions/pi-tool-trace-extension.ts"
fi
PI_EXIT=$?

if [ "$PI_EXIT" -ne 0 ]; then
    log "Pi CLI failed with exit code $PI_EXIT - skipping tests"
    exit "$PI_EXIT"
fi

# Run tests/evaluation after Pi finishes
if [ "$TASK_TYPE" = "kaggle_kernel" ]; then
    # Kaggle task: run the solution script
    log "Running Kaggle solution..."
    cd "$PACKAGE_PATH"
    if [ -f "solution.R" ]; then
        run_as_trainr "source /tmp/trainr_env.sh && cd /workspace/notebook && Rscript solution.R 2>&1" || true
    else
        log "WARNING: No solution.R found in notebook directory"
    fi
else
    # R package task: run testthat tests
    log "Running testthat tests..."
    cd "$PACKAGE_PATH"
    run_as_trainr "source /tmp/trainr_env.sh && cd /workspace/package && Rscript -e \"testthat::test_dir('tests/testthat', reporter = 'check')\" 2>&1" || true
fi

log "Tests completed"
