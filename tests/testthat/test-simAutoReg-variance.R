## ---- test-simAutoReg-variance
lapply(
  X = 1,
  FUN = function(i,
                 n,
                 location,
                 scale,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    y <- SimVariance(
      n = n,
      location = location,
      chol_scale = chol(scale)
    )
    location_est <- colMeans(log(y))
    scale_est <- stats::var(log(y))
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
              scale - round(scale_est, digits = 1)
            ) <= tol
          )
        )
      }
    )
  },
  n = 10000L,
  location <- c(0.5, -0.2, 0.1),
  scale = matrix(
    data = c(0.5, 0.3, 0.3, 0.3, 0.5, 0.2, 0.3, 0.2, 0.5),
    nrow = 3,
    byrow = TRUE
  ),
  tol = 0.05,
  text = "test-simAutoReg-variance"
)
