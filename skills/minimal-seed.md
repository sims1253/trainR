# R Package Testing

Write tests for R functions using testthat.

## Basic Pattern

```r
test_that("function works correctly", {
  expect_equal(my_function(input), expected_output)
})
```

## Guidelines

- Test normal inputs
- Test edge cases (NA, empty, boundary values)  
- Test error conditions with expect_error()
- Use descriptive test names
