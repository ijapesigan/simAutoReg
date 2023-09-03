#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Random Symmetric Positive Definite Matrix
//'
//' This function generates a random positive definite matrix.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param p Positive integer.
//'   Dimension of the `p` by `p` matrix.
//'
//' @examples
//' set.seed(42)
//' SimPD(p = 3)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim cov
//' @export
// [[Rcpp::export]]
arma::mat SimPD(int p) {
  // Step 1: Generate a p x p matrix filled with random values
  arma::mat data(p, p, arma::fill::randn);

  // Step 2: Make the matrix symmetric by multiplying it with its transpose
  data = data * data.t();

  // Step 3: Add a small positive diagonal to ensure positive definiteness
  data += 0.001 * arma::eye<arma::mat>(p, p);

  // Step 4: Return the positive definite matrix
  return data;
}
