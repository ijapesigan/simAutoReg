## ---- test-simAutoReg-var-exo
lapply(
  X = 1,
  FUN = function(i,
                 time,
                 burn_in,
                 constant,
                 coef,
                 chol_cov,
                 exo_coef,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    p <- ncol(coef) / length(constant)
    exo_mat <- SimMVN(
      n = time + burn_in,
      location = c(0, 0, 0),
      chol_scale = chol(diag(3))
    )
    y <- SimVARExo(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      chol_cov = chol_cov,
      exo_mat = exo_mat,
      exo_coef = exo_coef
    )
    dims <- dim(y)
    yx <- YXExo(
      data = y,
      p = p,
      exo_mat = exo_mat[(burn_in + 1):(burn_in + time), , drop = FALSE]
    )
    Y <- yx$Y
    X <- yx$X
    coef_est <- .FitVAROLS(Y = Y, X = X)
    coef_est[, 1] <- round(coef_est[, 1], digits = 0)
    coef_est[, -1] <- round(coef_est[, -1], digits = 1)
    testthat::test_that(
      paste(text, "time"),
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
      paste(text, "constant, coef, exo_coef"),
      {
        testthat::expect_true(
          all(
            abs(
              cbind(
                constant,
                coef,
                exo_coef
              ) - coef_est
            ) <= tol
          )
        )
      }
    )
  },
  time = 10000L,
  burn_in = 200L,
  constant <- c(1, 1, 1),
  coef = matrix(
    data = c(
      0.5, 0.2, 0.0, 0.4, 0.0, 0.1,
      0.0, 0.5, 0.2, 0.0, 0.4, 0.0,
      0.0, 0.0, 0.5, 0.0, 0.0, 0.4
    ),
    nrow = 3,
    byrow = TRUE
  ),
  chol_cov = chol(
    diag(3)
  ),
  exo_coef = matrix(
    data = c(
      0.5, 0.0, 0.0,
      0.0, 0.5, 0.0,
      0.0, 0.0, 0.5
    ),
    nrow = 3
  ),
  tol = 0.05,
  text = "test-simAutoReg-var-exo"
)
