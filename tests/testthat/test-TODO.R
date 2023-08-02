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
    cat("\nTest for SimVariance.\n")
    set.seed(42)
    n <- 100
    location <- c(0.5, -0.2, 0.1)
    chol_scale <- chol(
      matrix(
        data = c(1.0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
        nrow = 3,
        byrow = TRUE
      )
    )
    SimVariance(n = n, location = location, chol_scale = chol_scale)
    cat("\nTest for PBootCI.\n")
    pb <- PBootVAROLS(data = VAR, p = 2, B = 100)
    PBootCI(pb)
    cat("\nTest for PBootSE.\n")
    PBootSE(pb)
  },
  text = "test-TODO"
)
