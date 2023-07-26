## ---- test-simAutoReg-var
lapply(
  X = 1,
  FUN = function(i,
                 time,
                 burn_in,
                 constant,
                 coef,
                 chol_cov,
                 tol,
                 text) {
    message(text)
    y <- SimVAR(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      chol_cov = chol_cov
    )
    yx <- YX(y, 2)
    est <- FitVAROLS(Y = yx$Y, X = yx$X)
    testthat::test_that(
      text,
      {
        testthat::expect_true(
          all(
            abs(
              coef - cbind(
                est[1:3, ],
                est[4:6, ]
              )
            ) <= tol
          )
        )
      }
    )
  },
  time = 1000000L,
  burn_in = 200L,
  constant <- c(1, 1, 1),
  coef = matrix(
    data = c(
      0.5, 0.0, 0.0, 0.4, 0.0, 0.0,
      0.0, 0.5, 0.0, 0.0, 0.4, 0.0,
      0.0, 0.0, 0.5, 0.0, 0.0, 0.4
    ),
    nrow = 3,
    byrow = TRUE
  ),
  chol_cov = chol(
    matrix(
      data = c(
        0.1, 0.0, 0.0,
        0.0, 0.1, 0.0,
        0.0, 0.0, 0.1
      ),
      nrow = 3,
      byrow = TRUE
    )
  ),
  tol = 0.05,
  text = "test-simAutoReg-var"
)
