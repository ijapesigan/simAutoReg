## ---- test-simAutoReg-var-lasso
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
    set.seed(42)
    y <- SimVAR(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      chol_cov = chol_cov
    )
    dims <- dim(y)
    yx <- YX(y, 2)
    Y_std <- StdMat(yx$Y)
    X_std <- StdMat(yx$X[, -1])
    lambdas <- 10^seq(-5, 5, length.out = 100)
    search <- SearchVARLasso(Y_std = Y_std, X_std = X_std, lambdas = lambdas)
    lasso <- OrigScale(SelectVARLasso(search), Y = yx$Y, X = yx$X[, -1])
    phi <- c(
      lasso[1, 1],
      lasso[2, 2],
      lasso[3, 3],
      lasso[1, 4],
      lasso[2, 5],
      lasso[3, 6]
    )
    lasso[1, 1] <- 0
    lasso[2, 2] <- 0
    lasso[3, 3] <- 0
    lasso[1, 4] <- 0
    lasso[2, 5] <- 0
    lasso[3, 6] <- 0
    testthat::test_that(
      paste(text, "sparsity"),
      {
        testthat::expect_true(
          sum(
            lasso
          ) == 0
        )
      }
    )
    testthat::test_that(
      paste(text, "auto"),
      {
        testthat::expect_true(
          all(
            abs(round(phi, digits = 1) - 0.3) <= tol
          )
        )
      }
    )
  },
  time = 1000L,
  burn_in = 200L,
  constant <- c(1, 1, 1),
  coef = matrix(
    data = c(
      0.3, 0.0, 0.0, 0.3, 0.0, 0.0,
      0.0, 0.3, 0.0, 0.0, 0.3, 0.0,
      0.0, 0.0, 0.3, 0.0, 0.0, 0.3
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
  tol = 0.10,
  text = "test-simAutoReg-var-lasso"
)
