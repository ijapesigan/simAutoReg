## ---- test-simAutoReg-var-exo
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
        cov <- diag(k)
        diag(cov) <- 0.1
        chol_cov <- chol(cov)
        m <- 2
        exo_mat <- SimMVN(
          n = time + burn_in,
          location = rep(x = 0, times = m),
          chol_scale = chol(diag(m))
        )
        y <- SimVARExo(
          time = time,
          burn_in = burn_in,
          constant = constant,
          coef = coef,
          chol_cov = chol_cov,
          exo_mat = exo_mat,
          exo_coef = matrix(
            data = 0.01,
            nrow = k,
            ncol = m
          )
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
        coef_est <- .FitVAROLS(Y = Y, X = X)
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
  text = "test-simAutoReg-var-exo"
)
