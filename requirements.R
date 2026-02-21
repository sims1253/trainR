# R dependencies for posit-gskill
# Install with: Rscript requirements.R
# Uses renv for reproducible package management

# Initialize renv for reproducibility
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org/")
}

# Initialize renv project if not already done
if (!file.exists("renv.lock")) {
  renv::init()
}

# Core packages with pinned versions for reproducibility
packages <- list(
  # Testing framework
  testthat = "3.2.1",
  devtools = "2.4.4",
  covr = "3.6.4",
  
  # Test utilities
  withr = "3.0.2",
  mockr = "0.2.0",
  
  # Package management
  pak = "0.7.2",
  
  # Data handling
  jsonlite = "1.8.9",
  
  # R code parsing (for AST analysis)
  parsed = NULL  # Use latest if available
)

# Install packages using renv for version control
cat("\nInstalling packages with renv...\n")
for (pkg_name in names(packages)) {
  pkg_version <- packages[[pkg_name]]
  
  if (!is.null(pkg_version)) {
    # Install specific version
    spec <- paste0(pkg_name, "@", pkg_version)
    cat(sprintf("  Installing %s...\n", spec))
    tryCatch({
      renv::install(spec)
    }, error = function(e) {
      # Fallback to latest if version not available
      cat(sprintf("    Version %s not available, installing latest...\n", pkg_version))
      renv::install(pkg_name)
    })
  } else {
    # Install latest
    cat(sprintf("  Installing %s (latest)...\n", pkg_name))
    tryCatch({
      renv::install(pkg_name)
    }, error = function(e) {
      cat(sprintf("    Package %s not available, skipping...\n", pkg_name))
    })
  }
}

# Snapshot current state to lockfile
cat("\nSnapshotting package state...\n")
renv::snapshot()

# Verify installations
cat("\nInstalled packages:\n")
for (pkg_name in names(packages)) {
  if (requireNamespace(pkg_name, quietly = TRUE)) {
    version <- as.character(packageVersion(pkg_name))
    cat(sprintf("  ✓ %s (%s)\n", pkg_name, version))
  } else {
    cat(sprintf("  ✗ %s (FAILED)\n", pkg_name))
  }
}

cat("\nR environment ready for posit-gskill!\n")
cat("Run 'renv::restore()' to restore from lockfile in the future.\n")
