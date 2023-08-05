## ---- test-simAutoReg-ar
lapply(
  X = 1,
  FUN = function(i,
                 time,
                 burn_in,
                 constant,
                 coef,
                 sd,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    data <- SimAR(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      sd = sd
    )
    yx <- YX(data = data, p = length(coef))
    coef_est <- simAutoReg:::.FitVAROLS(Y = yx$Y, X = yx$X)
    testthat::test_that(
      paste(text, "time"),
      {
        testthat::expect_true(
          time - dim(data)[1] == 0
        )
      }
    )
    testthat::test_that(
      paste(text, "constant and coef"),
      {
        testthat::expect_true(
          all(
            abs(
              c(constant, coef) - coef_est
            ) <= tol
          )
        )
      }
    )
  },
  time = 10000L,
  burn_in = 200L,
  constant = 2,
  coef = c(0.5, -0.3),
  sd = 0.1,
  tol = 0.05,
  text = "test-simAutoReg-ar"
)
