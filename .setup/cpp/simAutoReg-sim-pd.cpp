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
  arma::mat L(p, p,
              arma::fill::randn);  // Generate a random lower triangular matrix
  arma::mat A = L * L.t();         // Compute the product of the matrix and its
                                   // transpose to make it positive definite
  return A;
}
