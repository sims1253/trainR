---
name: testing-r-packages
description: Comprehensive guidance for testing R packages using testthat (3rd edition). Covers test organization, BDD-style testing with describe/it, fixtures using withr, snapshot testing, and mocking with local_mocked_bindings.
version: 1.0.0
author: Posit
tags:
  - R
  - testing
  - testthat
  - unit-tests
  - fixtures
  - mocking
  - snapshots
---

# R Package Testing with testthat (3rd Edition)

Comprehensive guidance for writing tests for R packages using the testthat package (3rd edition). Covers test organization, BDD-style testing with describe/it, test fixtures using withr, snapshot testing, and mocking.

## When to Use This Skill

Use this skill when:
- Writing tests for R packages
- Setting up test infrastructure for a new package
- Converting from base R testing to testthat
- Implementing test fixtures, mocking, or snapshot testing
- Organizing test suites for complex packages

## Test Organization

### Directory Structure

```
my_package/
├── R/
│   └── my_function.R
├── tests/
│   ├── testthat.R        # Test runner setup
│   └── testthat/
│       ├── test-my_function.R
│       ├── test-utils.R
│       └── helper-*.R    # Shared test helpers
```

### Test File Naming

- Test files: `test-{function_or_feature}.R`
- Helper files: `helper-{name}.R`
- Setup files: `setup-{name}.R`

```r
# tests/testthat/test-str_utils.R

test_that("str_trim handles basic input", {
  expect_equal(str_trim("  hello  "), "hello")
})
```

## BDD-Style Testing with describe/it

Use `describe()` and `it()` for organized, readable test blocks:

```r
describe("str_trim()", {
  it("removes leading whitespace", {
    expect_equal(str_trim("  hello"), "hello")
  })
  
  it("removes trailing whitespace", {
    expect_equal(str_trim("hello  "), "hello")
  })
  
  it("handles NA values gracefully", {
    expect_equal(str_trim(NA), NA_character_)
  })
  
  it("preserves internal whitespace", {
    expect_equal(str_trim("  hello world  "), "hello world")
  })
})
```

### Nested describe Blocks

```r
describe("package_name::function_name()", {
  describe("with valid input", {
    it("returns expected output", {
      expect_equal(function_name("input"), "expected")
    })
  })
  
  describe("with invalid input", {
    it("throws informative error", {
      expect_error(function_name(NULL), "must not be NULL")
    })
  })
})
```

## Test Fixtures with withr

Use `withr` functions to manage side effects and ensure cleanup:

### Local Functions (Preferred)

Use `local_*()` functions within tests for automatic cleanup:

```r
test_that("file processing works", {
  # Create temporary file that's automatically deleted
  tmp_file <- withr::local_tempfile(fileext = ".csv")
  write.csv(data, tmp_file)
  
  result <- process_file(tmp_file)
  expect_s3_class(result, "data.frame")
})

test_that("environment variables are restored", {
  # Set env var that's automatically restored
  withr::local_envvar(MY_VAR = "test_value")
  
  result <- my_function()
  expect_equal(Sys.getenv("MY_VAR"), "test_value")
})

test_that("working directory is restored", {
  # Change directory that's automatically restored
  withr::local_dir(tempdir())
  
  result <- getwd()
  expect_true(grepl(basename(tempdir()), result))
})

test_that("options are restored", {
  # Set option that's automatically restored
  withr::local_options(my.package.verbose = TRUE)
  
  result <- verbose_function()
  expect_true(attr(result, "verbose"))
})
```

### Common withr Functions

| Function | Purpose | Cleanup |
|----------|---------|---------|
| `local_tempfile()` | Create temp file | Deletes file |
| `local_tempdir()` | Create temp directory | Deletes directory |
| `local_envvar()` | Set environment variable | Restores value |
| `local_dir()` | Change working directory | Restores directory |
| `local_options()` | Set options | Restores options |
| `local_par()` | Set graphics parameters | Restores parameters |
| `local_library()` | Attach package | Detaches package |

### Setup/Teardown for Multiple Tests

```r
# Use local_test_context for shared setup
setup_test_env <- function() {
  tmp_dir <- withr::local_tempdir()
  withr::local_dir(tmp_dir)
  invisible(tmp_dir)
}

test_that("test 1 uses isolated environment", {
  setup_test_env()
  # Test code here
})

test_that("test 2 uses fresh environment", {
  setup_test_env()
  # Each test gets fresh setup
})
```

## Snapshot Testing

Snapshot tests capture and compare output to stored reference:

### Basic Snapshot Tests

```r
test_that("output format matches snapshot", {
  x <- format_report(my_data)
  expect_snapshot(x)
})

test_that("cli output matches snapshot", {
  expect_snapshot(cli::cli_alert("Important message"))
})

test_that("error message matches snapshot", {
  expect_snapshot_error(stop("Something went wrong"))
})
```

### Snapshot Variants

```r
# Compare printed output
expect_snapshot_print(x)

# Compare str() output
expect_snapshot_str(x)

# Compare cat() output
expect_snapshot_cat(x)

# Compare error messages
expect_snapshot_error(stop("Error message"))

# Compare warnings
expect_snapshot_warning(warning("Warning message"))
```

### Updating Snapshots

When output intentionally changes, update snapshots:

```r
# In test file
testthat::snapshot_review()
# Or via command line:
# testthat::snapshot_accept("test-file.R")
```

### Testing with cli Package

```r
test_that("cli output is correct", {
  # Use expect_snapshot for cli output
  local_edition(3)  # Ensure 3rd edition for consistent snapshots
  
  expect_snapshot({
    cli::cli_h1("Header")
    cli::cli_alert("Alert message")
    cli::cli_ul(c("item 1", "item 2"))
  })
})
```

## Mocking with testthat

Use mocking to isolate tests from external dependencies:

### Basic Mocking with local_mocked_bindings

```r
test_that("function uses mocked dependency", {
  # Create mock function
  mock_api_call <- function(url) {
    list(status = 200, data = "mocked response")
  }
  
  # Apply mock within test
  local_mocked_bindings(api_call = mock_api_call)
  
  result <- my_function()
  expect_equal(result$status, 200)
})
```

### Mocking Package Functions

```r
test_that("Sys.time is mocked", {
  # Mock time-dependent code
  local_mocked_bindings(
    Sys.time = function() as.POSIXct("2024-01-01 12:00:00")
  )
  
  result <- timestamp_function()
  expect_equal(result, "2024-01-01 12:00:00")
})
```

### Mocking Multiple Functions

```r
test_that("multiple dependencies are mocked", {
  local_mocked_bindings(
    read_file = function(path) "mocked content",
    write_file = function(path, content) NULL,
    network_call = function() list(success = TRUE)
  )
  
  result <- process_data()
  expect_equal(result, "processed mocked content")
})
```

### Advanced Mocking with Return Values

```r
test_that("mock returns different values on each call", {
  call_count <- 0
  mock_sequential <- function() {
    call_count <<- call_count + 1
    paste("call", call_count)
  }
  
  local_mocked_bindings(get_value = mock_sequential)
  
  expect_equal(get_value(), "call 1")
  expect_equal(get_value(), "call 2")
  expect_equal(get_value(), "call 3")
})
```

## Expectations Reference

### Equality

```r
# Exact equality
expect_equal(x, expected)
expect_identical(x, expected)  # Also checks attributes

# Type-specific
expect_type(x, "list")
expect_s3_class(x, "data.frame")
expect_s4_class(x, "MyClass")
expect_vector(x, ptype = integer(), size = 5)
```

### Comparisons

```r
# Numeric comparisons
expect_gt(x, 0)
expect_gte(x, 0)
expect_lt(x, 100)
expect_lte(x, 100)

# Tolerance for floating point
expect_equal(0.1 + 0.2, 0.3, tolerance = 1e-8)
```

### Strings and Patterns

```r
# String matching
expect_match(x, "^pattern$")
expect_setequal(x, c("a", "b"))
expect_mapequal(x, list(a = 1, b = 2))
```

### Conditions

```r
# Errors
expect_error(stop("message"))
expect_error(stop("message"), "message")
expect_error(stop("message"), class = "my_error")

# Warnings
expect_warning(warning("message"))
expect_no_warning(safe_function())

# Messages
expect_message(message("hello"))
expect_silent(no_output_function())
```

### Length and Content

```r
# Length
expect_length(x, 3)
expect_size(x, 10)  # For vectors

# Content
expect_true(condition)
expect_false(condition)
expect_null(x)
expect_named(x, c("a", "b"))
```

## Test File Patterns

### Complete Test File Example

```r
# tests/testthat/test-data_processor.R

# Setup helper (runs before each test)
setup({
  options(my.package.verbose = FALSE)
})

# Teardown (runs after each test)
teardown({
  options(my.package.verbose = NULL)
})

describe("process_data()", {
  it("handles numeric vectors", {
    result <- process_data(c(1, 2, 3))
    expect_type(result, "list")
    expect_named(result, c("mean", "sd"))
  })
  
  it("handles data frames", {
    df <- data.frame(x = 1:5, y = letters[1:5])
    result <- process_data(df)
    expect_s3_class(result, "processed_data")
  })
  
  it("throws error for invalid input", {
    expect_error(process_data("invalid"), "must be numeric")
  })
  
  it("uses mocked dependencies", {
    mock_fetch <- function() list(data = c(1, 2, 3))
    local_mocked_bindings(fetch_remote_data = mock_fetch)
    
    result <- process_data()
    expect_length(result$data, 3)
  })
})

describe("validate_input()", {
  it("returns TRUE for valid input", {
    expect_true(validate_input(1:10))
  })
  
  it("returns FALSE for invalid input", {
    expect_false(validate_input("string"))
  })
  
  it("handles NA values", {
    expect_true(validate_input(c(1, NA, 3)))
  })
})
```

## Best Practices

### 1. Write Focused Tests

```r
# Good: One assertion per test
test_that("mean returns numeric", {
  expect_type(mean(1:10), "double")
})

# Avoid: Multiple unrelated assertions
test_that("mean works", {
  expect_type(mean(1:10), "double")
  expect_equal(mean(1:10), 5.5)  # Separate test
  expect_length(mean(1:10), 1)   # Separate test
})
```

### 2. Use Descriptive Test Names

```r
# Good: Describes expected behavior
test_that("str_trim removes leading and trailing whitespace", {
  expect_equal(str_trim("  hello  "), "hello")
})

# Avoid: Generic name
test_that("str_trim works", {
  expect_equal(str_trim("  hello  "), "hello")
})
```

### 3. Test Edge Cases

```r
describe("divide_safely()", {
  it("handles division by zero", {
    expect_equal(divide_safely(10, 0), NA_real_)
  })
  
  it("handles NA inputs", {
    expect_equal(divide_safely(NA, 2), NA_real_)
    expect_equal(divide_safely(10, NA), NA_real_)
  })
  
  it("handles very large numbers", {
    expect_true(is.finite(divide_safely(1e300, 1e-300)))
  })
})
```

### 4. Isolate Tests with Fixtures

```r
# Good: Each test is isolated
test_that("test 1 is isolated", {
  tmp <- withr::local_tempfile()
  # tmp is automatically cleaned up
})

test_that("test 2 is isolated", {
  tmp <- withr::local_tempfile()
  # Fresh temp file, no interference
})
```

### 5. Skip Tests Conditionally

```r
test_that("API integration works", {
  skip_if_not_installed("httr")
  skip_if_offline()
  skip_on_cran()
  
  result <- api_call()
  expect_s3_class(result, "response")
})
```

## Anti-Patterns to Avoid

### 1. Not Cleaning Up Side Effects

```r
# Bad: Side effects leak between tests
test_that("test 1 modifies global state", {
  options(my.package.value = "modified")
  # Option persists to other tests!
})

# Good: Automatic cleanup
test_that("test 1 is isolated", {
  withr::local_options(my.package.value = "modified")
  # Option restored after test
})
```

### 2. Over-Mocking

```r
# Bad: Mocking everything makes test meaningless
test_that("over-mocked test", {
  local_mocked_bindings(
    func1 = function() 1,
    func2 = function() 2,
    func3 = function() 3  # All real code mocked away
  )
  expect_equal(real_function(), ???)  # What are we testing?
})

# Good: Mock only external dependencies
test_that("properly mocked test", {
  local_mocked_bindings(http_call = function() mock_response)
  result <- my_function()  # Tests real logic
  expect_equal(result$status, 200)
})
```

### 3. Hardcoded Absolute Paths

```r
# Bad: Hardcoded paths
test_that("reads file", {
  data <- read.csv("/home/user/data.csv")  # Fails elsewhere
})

# Good: Relative or temp paths
test_that("reads file", {
  tmp <- withr::local_tempfile(fileext = ".csv")
  write.csv(test_data, tmp)
  data <- read.csv(tmp)
  expect_s3_class(data, "data.frame")
})
```

### 4. Relying on Test Order

```r
# Bad: Test depends on previous test
test_that("setup creates object", {
  .GlobalEnv$shared_object <- create_object()
})

test_that("uses shared object", {
  # Assumes previous test ran first - flaky!
  expect_s3_class(shared_object, "my_class")
})

# Good: Each test is self-contained
test_that("uses object", {
  obj <- create_object()
  expect_s3_class(obj, "my_class")
})
```

## Running Tests

### Command Line

```bash
# Run all tests
Rscript -e "testthat::test_dir('tests/testthat')"

# Run specific test file
Rscript -e "testthat::test_file('tests/testthat/test-utils.R')"

# Run with devtools
Rscript -e "devtools::test()"
```

### Coverage

```r
# Check test coverage
covr::package_coverage()

# Generate HTML report
covr::report()
```

## Quick Reference

| Task | Function |
|------|----------|
| Create test file | `test_file <- "tests/testthat/test-feature.R"` |
| BDD-style tests | `describe()`, `it()` |
| Expectations | `expect_equal()`, `expect_error()`, `expect_s3_class()` |
| Fixtures | `withr::local_tempfile()`, `withr::local_envvar()` |
| Mocking | `local_mocked_bindings()` |
| Snapshots | `expect_snapshot()` |
| Skip tests | `skip_if_not_installed()`, `skip_on_cran()` |
| Run tests | `devtools::test()`, `testthat::test_dir()` |
| Coverage | `covr::package_coverage()` |

## Related Skills

- **cli**: Command-line interface styling for R packages
- **withr**: Managing side effects and state
- **r-package-dev**: General R package development patterns
- **r-tidyverse**: Modern tidyverse coding patterns
