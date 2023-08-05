#' Simulate Data from a Vector Autoregressive Zero-Inflated Poisson (VARZIP)
#' Model
#'
#' This function generates synthetic time series data
#' from a Vector Autoregressive Zero-Inflated Poisson (VARZIP) model.
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
#'
#' @return Numeric matrix containing the simulated time series data
#'   with dimensions `k` by `time`,
#'   where `k` is the number of variables
#'   and `time` is the number of observations.
#'
#' @examples
#' set.seed(42)
#' time <- 50L
#' burn_in <- 10L
#' k <- 3
#' p <- 2
#' constant <- c(1, 1, 1)
#' coef <- matrix(
#'   data = c(
#'     0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
#'     0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
#'     0.0, 0.0, 0.6, 0.0, 0.0, 0.3
#'   ),
#'   nrow = k,
#'   byrow = TRUE
#' )
#' chol_cov <- chol(diag(3))
#' y <- SimVARZIP(
#'   time = time,
#'   burn_in = burn_in,
#'   constant = constant,
#'   coef = coef,
#'   chol_cov = chol_cov
#' )
#' head(y)
#'
#' @details
#' The [simAutoReg::SimVARZIP()] function generates synthetic time series data
#' from a Vector Autoregressive (VAR)
#' with Zero-Inflated Poisson (ZIP) model for the first observed variable.
#' See [simAutoReg::SimVAR()] for more details on generating data for VAR(p).
#' The `SimVARZIP` function goes further by using the generated values
#' for the first variable to generate data from the ZIP model.
#' The exponential of the values from the first variable
#' from the original VAR(p) model
#' are used as the `intensity` parameter in the Poisson distribution
#' in the ZIP model.
#' Data from the ZIP model are used to replace the original values
#' for the first variable.
#' Values for the rest of the variables are unchanged.
#' The generated data includes a burn-in period,
#' which is excluded before returning the results.
#'
#' The steps involved in generating the time series data are as follows:
#'
#' - Extract the number of variables `k`
#'   and the number of lags `p` from the input.
#' - Create a matrix `data` of size `k` x (`time + burn_in`)
#'   to store the generated data.
#' - Set the initial values of the matrix `data`
#'   using the constant term `constant`.
#' - For each time point starting from the `p`-th time point
#'   to `time + burn_in - 1`:
#'   * Generate a vector of random process noise
#'     from a multivariate normal distribution
#'     with mean 0 and covariance matrix `chol_cov`.
#'   * Generate the VAR time series values for each variable `j`
#'     at time `t` by applying the autoregressive terms
#'     for each lag `lag` and each variable `l`.
#'     - Add the generated noise to the VAR time series values.
#'     - For the first variable,
#'       apply the Zero-Inflated Poisson (ZIP) model:
#'       * Compute the intensity `intensity`
#'         as the exponential of the first variable's value at time `t`.
#'       * Sample a random value `u`
#'         from a uniform distribution on \[0, 1\].
#'       * If `u` is less than `intensity / (1 + intensity)`,
#'         set the first variable's value to zero (inflation).
#'       * Otherwise, sample the first variable's value
#'         from a Poisson distribution
#'         with mean `intensity` (count process).
#' - Transpose the data matrix `data` and return only
#'   the required time period after burn-in as a numeric matrix.
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords simAutoReg sim
#' @export
SimVARZIP <- function(time,
                      burn_in,
                      constant,
                      coef,
                      chol_cov) {
  y <- .SimVARZIPCpp(
    time = burn_in + time,
    constant = constant,
    coef = coef,
    chol_cov = chol_cov
  )
  return(
    y[
      (burn_in + 1):(burn_in + time), ,
      drop = FALSE
    ]
  )
}
