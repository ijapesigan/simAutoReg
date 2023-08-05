#' Simulate Data from a Vector Autoregressive (VAR) Model with Exogenous Variables
#'
#' This function generates synthetic time series data
#' from a Vector Autoregressive (VAR) model with exogenous variables.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param time Integer.
#'   Number of time points to simulate.
#' @param burn_in Integer.
#'   Number of burn-in observations to exclude before returning the results.
#' @param constant Numeric vector.
#'   The constant term vector of length `k`,
#'   where `k` is the number of variables.
#' @param coef Numeric matrix.
#'   Coefficient matrix with dimensions `k` by `(k * p)`.
#'   Each `k` by `k` block corresponds to the coefficient matrix
#'   for a particular lag.
#' @param chol_cov Numeric matrix.
#'   The Cholesky decomposition of the covariance matrix
#'   of the multivariate normal noise.
#'   It should have dimensions `k` by `k`.
#' @param exo_mat Numeric matrix.
#'   Matrix of exogenous covariates with dimensions `time + burn_in` by `x`.
#'   Each column corresponds to a different exogenous variable.
#' @param exo_coef Numeric vector.
#'   Coefficient matrix with dimensions `k` by `x`
#'   associated with the exogenous covariates.
#'
#' @return Numeric matrix containing the simulated time series data
#'   with dimensions `k` by `time`,
#'   where `k` is the number of variables and
#'   `time` is the number of observations.
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords simAutoReg sim
#' @export
SimVARExo <- function(time,
                      burn_in,
                      constant,
                      coef,
                      chol_cov,
                      exo_mat,
                      exo_coef) {
  y <- .SimVARExoCpp(
    time = burn_in + time,
    constant = constant,
    coef = coef,
    chol_cov = chol_cov,
    exo_mat = exo_mat,
    exo_coef = exo_coef
  )
  return(
    y[
      (burn_in + 1):(burn_in + time), ,
      drop = FALSE
    ]
  )
}
