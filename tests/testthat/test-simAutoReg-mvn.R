## ---- test-simAutoReg-mvn
lapply(
  X = 1,
  FUN = function(i,
                 n,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    location <- c(1, 1, 1)
    scale <- diag(3)
    chol_scale <- chol(scale)
    y <- SimMVN(
      n = n,
      location = location,
      chol_scale = chol_scale
    )
    location_est <- round(colMeans(y), digits = 0)
    scale_est <- round(var(y), digits = 0)
    chol_scale_est <- chol(scale_est)
    testthat::test_that(
      paste(text, "location"),
      {
        testthat::expect_true(
          all(
            abs(
              location - location_est
            ) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "scale"),
      {
        testthat::expect_true(
          all(
            abs(
              scale - scale_est
            ) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "chol"),
      {
        testthat::expect_true(
          all(
            abs(
              chol_scale - chol_scale_est
            ) <= tol
          )
        )
      }
    )
  },
  n = 1000L,
  tol = 0.05,
  text = "test-simAutoReg-mvn"
)