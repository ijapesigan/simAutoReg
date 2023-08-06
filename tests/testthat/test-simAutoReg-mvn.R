## ---- test-simAutoReg-mvn
lapply(
  X = 1:5,
  FUN = function(p,
                 n,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    location <- round(
      x = stats::rnorm(n = p),
      digits = 2
    )
    scale <- round(
      x = SimPD(p = p),
      digits = 2
    )
    y <- SimMVN(
      n = n,
      location = location,
      chol_scale = chol(scale)
    )
    location_est <- round(
      x = colMeans(y),
      digits = 2
    )
    scale_est <- round(
      x = stats::var(y),
      digits = 2
    )
    testthat::test_that(
      paste(text, p, "location"),
      {
        testthat::expect_true(
          all(
            abs(
              location - location_est
            ) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, p, "scale"),
      {
        testthat::expect_true(
          all(
            abs(
              scale - scale_est
            ) <= tol
          )
        )
      }
    )
  },
  n = 1000000L,
  tol = 0.05,
  text = "test-simAutoReg-mvn"
)
