// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-mvn-fixed.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Multivariate Normal Random Numbers with Optional Fixed Values
//'
//' This function generates multivariate normal random numbers
//' with optinal fixed values where the variance is zero.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param n Integer.
//'   Number of samples to generate.
//' @param location Numeric vector.
//'   Mean vector of length `k`, where `k` is the number of variables.
//' @param scale Numeric matrix.
//'   Covariance matrix of dimensions `k` by `k`.
//'   Values for variables with variance of `0` will be fixed
//'   to the corresponding value in `location`.
//'
//' @return Matrix containing the simulated multivariate normal random numbers,
//'   with dimensions `n` by `k`, where `n` is the number of samples
//'   and `k` is the number of variables.
//'
//' @examples
//' set.seed(42)
//' n <- 1000L
//' location <- c(0.5, -0.2, 0.1)
//' scale <- matrix(
//'   data = c(0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
//'   nrow = 3,
//'   byrow = TRUE
//' )
//' y <- SimMVNFixed(n = n, location = location, scale = scale)
//' colMeans(y)
//' var(y)
//'
//' @details
//' The [simAutoReg::SimMVNFixed()] function first identifies the indices
//' of non-constant variables (i.e., variables with variance not equal to 0)
//' in the covariance matrix.
//' It then extracts the non-constant elements from the mean vector
//' and the covariance matrix.
//' A Cholesky decomposition is performed on the covariance matrix
//' of non-constant variables.
//' Random samples are generated for the non-constant variables
//' using the Cholesky factor.
//' The generated data matrix is constructed
//' by setting the non-constant variables
//' and constant variables to their corresponding values.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimMVNFixed(int n, const arma::vec& location,
                      const arma::mat& scale) {
  // Get the number of variables
  int k = location.n_elem;

  // Find indices of non-constant variables (variance not equal to 0)
  arma::uvec non_constant_indices = arma::find(arma::diagvec(scale) != 0);

  // Extract non-constant elements from location and scale matrix
  arma::vec non_constant_location = location(non_constant_indices);
  arma::mat non_constant_scale =
      scale(non_constant_indices, non_constant_indices);

  // Perform Cholesky decomposition on the scale matrix of non-constant
  // variables
  arma::mat L = arma::chol(non_constant_scale);

  // Generate random samples for the non-constant variables
  arma::mat samples = arma::randn(n, non_constant_indices.n_elem) * L.t() +
                      arma::repmat(non_constant_location.t(), n, 1);

  // Create a matrix to store the generated data
  arma::mat data(n, k, arma::fill::zeros);

  // Set the non-constant variables in the data matrix
  for (arma::uword i = 0; i < non_constant_indices.n_elem; i++) {
    data.col(non_constant_indices(i)) = samples.col(i);
  }

  // Set constant variables to their corresponding values
  for (int i = 0; i < k; i++) {
    if (arma::as_scalar(scale(i, i)) == 0) {
      data.col(i).fill(arma::as_scalar(location(i)));
    }
  }

  return data;
}
