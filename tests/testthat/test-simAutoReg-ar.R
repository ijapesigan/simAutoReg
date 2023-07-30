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
    data <- as.vector(
      SimAR(
        time = time,
        burn_in = burn_in,
        constant = constant,
        coef = coef,
        sd = sd
      )
    )
    testthat::test_that(
      paste(text, "time"),
      {
        testthat::expect_true(
          all(
            abs(
              time - length(data)
            ) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "coef"),
      {
        testthat::expect_true(
          all(
            abs(
              coef - coef(
                stats::ar(
                  data,
                  aic = FALSE,
                  order.max = length(coef),
                  method = "ols"
                )
              )
            ) <= tol
          )
        )
      }
    )
  },
  time = 1000L,
  burn_in = 200L,
  constant = 2,
  coef = c(0.5, -0.3),
  sd = 0.1,
  tol = 0.05,
  text = "test-simAutoReg-ar"
)
