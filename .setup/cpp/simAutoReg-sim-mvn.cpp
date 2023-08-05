// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-mvn.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".SimMVNCpp")]]
arma::mat SimMVNCpp(int n, const arma::vec& location,
                    const arma::mat& chol_scale) {
  int k = location.n_elem;

  // Generate multivariate normal random numbers
  arma::mat random_data = arma::randn(n, k);
  arma::mat data = random_data * chol_scale + arma::repmat(location.t(), n, 1);

  return data;
}

// Dependencies
