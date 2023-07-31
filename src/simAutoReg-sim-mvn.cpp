#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

//' Simulate Multivariate Normal Random Numbers
//'
//' This function generates multivariate normal random numbers.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param n Integer.
//'   Number of samples to generate.
//' @param location Numeric vector.
//'   Mean vector of length `k`, where `k` is the number of variables.
//' @param chol_scale Numeric matrix.
//'   Cholesky decomposition of the covariance matrix of dimensions `k` by `k`.
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
//'   data = c(1.0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
//'   nrow = 3,
//'   byrow = TRUE
//' )
//' chol_scale <- chol(scale)
//' y <- SimMVN(n = n, location = location, chol_scale = chol_scale)
//' colMeans(y)
//' var(y)
//'
//' @details
//' The [simAutoReg::SimMVN()] function generates
//' multivariate normal random numbers
//' using the Cholesky decomposition method.
//' Given the number of samples `n`, the mean vector `location` of length `k`
//' (where `k` is the number of variables),
//' and the Cholesky decomposition `chol_scale` of the covariance matrix
//' of dimensions `k` by `k`,
//' the function produces a matrix of multivariate normal random numbers.
//'
//' The steps involved in generating multivariate normal random numbers
//' are as follows:
//'
//' - Determine the number of variables `k` from the length of the mean vector.
//' - Generate random data from a standard multivariate normal distribution,
//'   resulting in an `n` by `k` matrix of random numbers.
//' - Transform the standard normal random data
//'   into multivariate normal random data
//'   using the Cholesky decomposition `chol_scale`.
//' - Add the mean vector `location` to the transformed data
//'   to obtain the final simulated multivariate normal random numbers.
//' - The function returns a matrix of simulated
//'   multivariate normal random numbers
//'   with dimensions `n` by `k`,
//'   where `n` is the number of samples and `k` is the number of variables.
//'   This matrix can be used for various statistical analyses and simulations.
//'
//' @seealso
//' The [chol()] function in R to obtain the Cholesky decomposition
//' of a covariance matrix.
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimMVN(int n,
                 const arma::vec& location,
                 const arma::mat& chol_scale)
{
  int k = location.n_elem;

  // Generate multivariate normal random numbers
  arma::mat random_data = arma::randn(n, k);
  arma::mat data = random_data * chol_scale + arma::repmat(location.t(), n, 1);

  return data;
}
