// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var-coef.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Random Vector Autoregressive Coefficients
//' for a Stationary VAR(p) Model
//'
//' This function generates stationary VAR(P) coefficients.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param k Positive integer. Number of autoregressive variables.
//' @param p Positive integer. Number of lags.
//'
//' @examples
//' set.seed(42)
//' SimVARCoef(k = 3, p = 2)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVARCoef(int k, int p) {
  arma::mat var_coefficients(k, k * p);

  while (true) {
    // Generate random coefficients between -0.9 and 0.9
    var_coefficients.randu();
    var_coefficients = -0.9 + 1.8 * var_coefficients;

    // Check if the eigenvalues of the companion matrix have moduli less than 1
    arma::mat companion_matrix(k * p, k * p, arma::fill::zeros);
    for (int i = 0; i < p; i++) {
      companion_matrix.submat(i * k, i * k, (i + 1) * k - 1, (i + 1) * k - 1) =
          var_coefficients.cols(i * k, (i + 1) * k - 1);
      if (i > 0) {
        companion_matrix.submat(i * k, (i - 1) * k, (i + 1) * k - 1,
                                i * k - 1) = arma::eye(k, k);
      }
    }

    arma::cx_vec eigenvalues = arma::eig_gen(companion_matrix);
    if (arma::all(arma::abs(eigenvalues) < 1)) {
      break;
    }
  }

  return var_coefficients;
}

