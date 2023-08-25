## ---- test-simAutoReg-var
lapply(
  X = 1:2,
  FUN = function(p,
                 time,
                 burn_in,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    lapply(
      X = 2:3,
      FUN = function(k) {
        constant <- round(
          x = stats::runif(
            n = k,
            min = 0,
            max = 0.8
          ),
          digits = 2
        )
        coef <- round(
          x = SimVARCoef(k = k, p = p),
          digits = 2
        )
        CheckVARCoef(coef)
       cov <- diag(
          x = 0.1,
          nrow = k,
          ncol = k
        )
        chol_cov <- chol(cov)
        y <- SimVAR(
          time = time,
          burn_in = burn_in,
          constant = constant,
          coef = coef,
          chol_cov = chol_cov
        )
        dims <- dim(y)
        yx <- YX(y, p = p)
        coef_est <- round(
          x = .FitVAROLS(Y = yx$Y, X = yx$X),
          digits = 2
        )
        testthat::test_that(
          paste(text, p, k, "time"),
          {
            testthat::expect_true(
              all(
                abs(
                  time - dims[1]
                ) <= tol
              )
            )
          }
        )
        testthat::test_that(
          paste(text, p, k, "constant and coef"),
          {
            testthat::expect_true(
              all(
                abs(
                  cbind(
                    constant,
                    coef
                  ) - coef_est
                ) <= tol
              )
            )
          }
        )
      }
    )
  },
  time = 1000000L,
  burn_in = 10L,
  tol = 0.05,
  text = "test-simAutoReg-var"
)
