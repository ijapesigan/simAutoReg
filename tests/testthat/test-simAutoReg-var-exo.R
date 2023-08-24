## ---- test-simAutoReg-var-exo
lapply(
  X = 1,
  FUN = function(i,
                 time,
                 burn_in,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    constant <- c(1, 1, 1)
    coef <- matrix(
      data = c(
        0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
        0.0, 0.0, 0.6, 0.0, 0.0, 0.3
      ),
      nrow = 3,
      byrow = TRUE
    )
    exo_coef <- matrix(
      data = c(
        0.5, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 0.5
      ),
      nrow = 3
    )
    cov <- diag(
      x = 0.1,
      nrow = k,
      ncol = k
    )
    chol_cov <- chol(cov)
    m <- 3
    exo_mat <- SimMVN(
      n = time + burn_in,
      location = rep(x = 0, times = m),
      chol_scale = chol(diag(m))
    )
    exo_coef <- diag(
      x = 0.01,
      nrow = k,
      ncol = m
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
      exo_mat = exo_mat[
        (burn_in + 1):(burn_in + time), ,
        drop = FALSE
      ]
    )
    Y <- yx$Y
    X <- yx$X
    coef_est <- FitVAROLS(Y = Y, X = X)
    coef_est[, 1] <- round(coef_est[, 1], digits = 0)
    coef_est[, -1] <- round(coef_est[, -1], digits = 2)
    testthat::test_that(
      paste(text, "constant and coef"),
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
  time = 1000000L,
  burn_in = 10L,
  tol = 0.05,
  text = "test-simAutoReg-var-exo"
)
