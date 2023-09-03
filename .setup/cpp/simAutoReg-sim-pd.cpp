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
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimPD(int p) {
  // Create a p x p matrix filled with random values
  arma::mat L(p, p, arma::fill::randn);

  // Compute the product of the matrix and its
  // transpose to make it symmetric
  arma::mat A = L * L.t();

  A += 0.001 * arma::eye<arma::mat>(p, p);

  return A;
}
