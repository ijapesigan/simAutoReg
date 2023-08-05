// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-mvn-fixed.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".SimMVNFixedCpp")]]
arma::mat SimMVNFixedCpp(int n, const arma::vec& location,
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
