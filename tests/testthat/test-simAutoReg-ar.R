## ---- test-simAutoReg-ar
lapply(
  X = 1:5,
  FUN = function(p,
                 time,
                 burn_in,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    constant <- round(
      x = stats::runif(
        n = 1,
        min = 0,
        max = 0.8
      ),
      digits = 2
    )
    coef <- round(
      x = as.vector(SimARCoef(p = p)),
      digits = 2
    )
    sd <- round(
      x = stats::runif(
        n = 1,
        min = 0.01,
        max = 0.10
      ),
      digits = 2
    )
    data <- SimAR(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      sd = sd
    )
    yx <- YX(data = data, p = p)
    coef_est <- round(
      x = as.vector(
        simAutoReg:::.FitVAROLS(
          Y = yx$Y,
          X = yx$X
        )
      ),
      digits = 2
    )
    testthat::test_that(
      paste(text, p, "time"),
      {
        testthat::expect_true(
          time - dim(data)[1] == 0
        )
      }
    )
    testthat::test_that(
      paste(text, p, "constant and coef"),
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
  time = 1000000L,
  burn_in = 10L,
  tol = 0.05,
  text = "test-simAutoReg-ar"
)
