#' Simulate Data from an Autoregressive Model with Constant Term
#'
#' This function generates synthetic time series data
#' from an autoregressive (AR) model.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param time Integer.
#'   Number of time points to simulate.
#' @param burn_in Integer.
#'   Number of burn-in periods before recording data.
#' @param constant Numeric.
#'   The constant term of the AR model.
#' @param coef Numeric vector.
#'   Autoregressive coefficients.
#' @param sd Numeric.
#'   The standard deviation of the random process noise.
#'
#' @return Numeric vector (column matrix) containing the simulated
#'   time series data.
#'
#' @examples
#' set.seed(42)
#' SimAR(time = 10, burn_in = 5, constant = 2, coef = c(0.5, -0.3), sd = 0.1)
#'
#' @details
#' The [simAutoReg::SimAR()] function generates synthetic time series data
#' from an autoregressive (AR) model.
#' The generated data follows the AR(p) model,
#' where `p` is the number of coefficients specified in `coef`.
#' The generated time series data includes a constant term
#' and autoregressive terms based on the provided coefficients.
#' Random noise, sampled from a normal distribution with mean 0
#' and standard deviation `sd`, is added to the time series.
#' A burn-in period is specified to exclude initial data points
#' from the output.
#'
#' The steps in generating the autoregressive time series with burn-in
#' are as follows:
#'
#' - Set the order of the AR model to `p` based on the length of `coef`.
#' - Create a vector data of size `time + burn_in`
#'   to store the generated AR time series data.
#' - Create a vector data of size `time + burn_in` of random process noise
#'   from a normal distribution with mean 0
#'   and standard deviation `sd`.
#' - Generate the autoregressive time series with burn-in using the formula:
#'   \deqn{Y_t = constant + \sum_{i=1}^{p} (coef[i] * Y_{t-i}) + noise_t}
#'   where \eqn{Y_t} is the time series data at time \eqn{t}, \eqn{constant}
#'   is the constant term,
#'   \eqn{coef[i]} are the autoregressive coefficients,
#'   \eqn{Y_{t-i}} are the lagged data points up to order `p`,
#'   and \eqn{noise_t} is the random noise at time \eqn{t}.
#' - Remove the burn-in period from the generated time series data.
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords simAutoReg sim
#' @export
SimAR <- function(time,
                  burn_in,
                  constant,
                  coef,
                  sd) {
  y <- .SimARCpp(
    time + burn_in,
    constant,
    coef,
    sd
  )
  return(
    y[
      (burn_in + 1):(burn_in + time),
      1,
      drop = FALSE
    ]
  )
}
