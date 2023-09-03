// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-variance.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Generate Random Data for the Variance Vector
//'
//' This function generates random data for the variance vector given by
//' \deqn{
//'   \boldsymbol{\sigma}^{2} =
//'   \exp \left( \boldsymbol{\mu} + \boldsymbol{\varepsilon} \right)
//'   \quad
//'   \text{with}
//'   \quad
//'   \boldsymbol{\varepsilon} \sim
//'   \mathcal{N} \left( \boldsymbol{0}, \boldsymbol{\Sigma} \right)
//' }.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param n Integer.
//'   Number of samples to generate.
//' @param location Numeric vector.
//'   The constant term \eqn{\boldsymbol{\mu}}.
//' @param chol_scale Numeric matrix.
//'   Cholesky decomposition of the covariance matrix \eqn{\boldsymbol{\Sigma}}
//'   for the multivariate normal random error \eqn{\boldsymbol{\varepsilon}}.
//'
//' @return Matrix with each row containing the simulated variance vector
//'   for each sample.
//'
//' @details
//' The [SimVariance()] function generates random data
//' for the variance vector
//' based on the exponential of a multivariate normal distribution.
//' Given the number of samples `n`,
//' the constant term \eqn{\boldsymbol{\mu}} represented
//' by the `location` vector,
//' and the Cholesky decomposition matrix \eqn{\boldsymbol{\Sigma}}
//' for the multivariate normal random error \eqn{\boldsymbol{\varepsilon}},
//' the function simulates \eqn{n} independent samples
//' of the variance vector \eqn{\boldsymbol{\sigma}^{2}}.
//' Each sample of the variance vector \eqn{\boldsymbol{\sigma}^{2}}
//' is obtained by
//' calculating the exponential of random variations
//' to the mean vector \eqn{\boldsymbol{\mu}}.
//' The random variations are generated using the Cholesky decomposition
//' of the covariance matrix \eqn{\boldsymbol{\Sigma}}.
//' Finally, the function returns a matrix with each column
//' containing the simulated
//' variance vector for each sample.
//'
//' @examples
//' set.seed(42)
//' n <- 10L
//' location <- c(0.5, -0.2, 0.1)
//' chol_scale <- chol(
//'   matrix(
//'     data = c(1.0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
//'     nrow = 3,
//'     byrow = TRUE
//'   )
//' )
//' SimVariance(n = n, location = location, chol_scale = chol_scale)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim variance
//' @export
// [[Rcpp::export]]
arma::mat SimVariance(int n, const arma::vec& location,
                      const arma::mat& chol_scale) {
  // Step 1: Simulate multivariate normal data
  arma::mat data = SimMVN(n, location, chol_scale);

  // Step 2: Transform the simulated data
  //         by taking the exponential of each element
  data = arma::exp(data);

  // Step 3: Return the transformed data
  return data;
}

// Dependencies
// simAutoReg-sim-mvn.cpp
