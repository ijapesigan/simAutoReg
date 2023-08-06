## ---- test-TODO
lapply(
  X = 1,
  FUN = function(i,
                 text) {
    message(text)
    cat("\nTest for SimVARZIP.\n")
    set.seed(42)
    time <- 50L
    burn_in <- 10L
    k <- 3
    p <- 2
    constant <- c(1, 1, 1)
    coef <- matrix(
      data = c(
        0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
        0.0, 0.0, 0.6, 0.0, 0.0, 0.3
      ),
      nrow = k,
      byrow = TRUE
    )
    chol_cov <- chol(diag(3))
    SimVARZIP(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      chol_cov = chol_cov
    )
    cat("\nTest for SimVARZIPExo.\n")
    exo_mat <- SimMVN(
      n = time + burn_in,
      location = c(0, 0, 0),
      chol_scale = chol(diag(3))
    )
    exo_coef <- matrix(
      data = c(
        0.5, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 0.5
      ),
      nrow = 3
    )
    SimVARZIPExo(
      time = time,
      burn_in = burn_in,
      constant = constant,
      coef = coef,
      chol_cov = chol_cov,
      exo_mat = exo_mat,
      exo_coef = exo_coef
    )
  },
  text = "test-TODO"
)
