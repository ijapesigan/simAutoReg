#' Simulate Multivariate Normal Random Numbers with Optional Fixed Values
#'
#' This function generates multivariate normal random numbers
#' with optinal fixed values where the variance is zero.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param n Integer.
#'   Number of samples to generate.
#' @param location Numeric vector.
#'   Mean vector of length `k`, where `k` is the number of variables.
#' @param scale Numeric matrix.
#'   Covariance matrix of dimensions `k` by `k`.
#'   Values for variables with variance of `0` will be fixed
#'   to the corresponding value in `location`.
#'
#' @return Matrix containing the simulated multivariate normal random numbers,
#'   with dimensions `n` by `k`, where `n` is the number of samples
#'   and `k` is the number of variables.
#'
#' @examples
#' set.seed(42)
#' n <- 1000L
#' location <- c(0.5, -0.2, 0.1)
#' scale <- matrix(
#'   data = c(0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
#'   nrow = 3,
#'   byrow = TRUE
#' )
#' y <- SimMVNFixed(n = n, location = location, scale = scale)
#' colMeans(y)
#' var(y)
#'
#' @details
#' The [simAutoReg::SimMVNFixed()] function first identifies the indices of non-constant variables
#' (i.e., variables with variance not equal to 0) in the covariance matrix.
#' It then extracts the non-constant elements from the mean vector and the covariance matrix.
#' A Cholesky decomposition is performed on the covariance matrix of non-constant variables.
#' Random samples are generated for the non-constant variables using the Cholesky factor.
#' The generated data matrix is constructed by setting the non-constant variables
#' and constant variables to their corresponding values.
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords simAutoReg sim
#' @export
SimMVNFixed <- function(n,
                        location,
                        scale) {
  return(
    .SimMVNFixedCpp(
      n = n,
      location = location,
      scale = scale
    )
  )
}
