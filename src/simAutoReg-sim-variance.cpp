#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

//' Generate Random Data for the Variance Vector
//'
//' This function generates random data for the variance vector given by
//' \deqn{
//'   \boldsymbol{\sigma^{2}} =
//'   \exp \left( \boldsymbol{\mu} + \boldsymbol{\varepsilon} \right)
//'   \quad
//'   \text{with}
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
//' @examples
//' set.seed(42)
//' n <- 100
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
//' @export
// [[Rcpp::export]]
arma::mat SimVariance(int n, arma::vec location, arma::mat chol_scale) {
  int k = location.n_elem; // Number of variables

  // Generate multivariate normal random vectors epsilon
  arma::mat epsilon = chol_scale * arma::randn(k, n);

  // Add the location vector to each column of epsilon
  epsilon.each_col() += location;

  // Compute the variance vector for each sample
  arma::mat variance = arma::exp(epsilon);

  return variance;
}

/*** R
# Example usage
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
*/
