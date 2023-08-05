// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-variance.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".SimVarianceCpp")]]
arma::mat SimVarianceCpp(int n, const arma::vec& location,
                         const arma::mat& chol_scale) {
  // Generate multivariate normal random numbers
  arma::mat mvn = SimMVNCpp(n, location, chol_scale);

  // Compute the variance vector for each sample
  arma::mat variance = arma::exp(mvn);

  return variance;
}

// Dependencies
// simAutoReg-sim-mvn.cpp
