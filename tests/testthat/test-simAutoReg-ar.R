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
    testthat::test_that(
      text,
      {
        testthat::expect_true(
          all(
            abs(
              coef - coef(
                stats::ar(
                  SimAR(
                    time = time,
                    burn_in = burn_in,
                    constant = constant,
                    coef = coef,
                    sd = sd
                  ),
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
  time = 1000000L,
  burn_in = 200L,
  constant = 2,
  coef = c(0.5, -0.3),
  sd = 0.1,
  tol = 0.05,
  text = "test-simAutoReg-ar"
)
