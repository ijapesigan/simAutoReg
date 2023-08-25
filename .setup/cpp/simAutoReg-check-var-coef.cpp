#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Check VAR(p) Coefficients for Stationarity
//'
//' The function checks if all the eigenvalues have moduli
//' (absolute values) less than 1.
//' If all eigenvalues have moduli less than 1,
//' it indicates that the VAR process is stable and, therefore, stationary.
//' If any eigenvalue has a modulus greater than or equal to 1,
//' it indicates that the VAR process is not stable and,
//' therefore, non-stationary.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef Numeric matrix.
//'   Coefficient matrix with dimensions `k` by `(k * p)`.
//'   Each `k` by `k` block corresponds to the coefficient matrix
//'   for a particular lag.
//'
//' @examples
//' set.seed(42)
//' CheckVARCoef(SimVARCoef(k = 3, p = 2))
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
bool CheckVARCoef(const arma::mat& coef) {
  int k = coef.n_rows;      // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  // Check if the eigenvalues of the companion matrix have moduli less than 1
  arma::mat companion(k * p, k * p, arma::fill::zeros);
  for (int i = 0; i < p; i++) {
    companion.submat(i * k, i * k, (i + 1) * k - 1, (i + 1) * k - 1) =
        coef.cols(i * k, (i + 1) * k - 1);
    if (i > 0) {
      companion.submat(i * k, (i - 1) * k, (i + 1) * k - 1, i * k - 1) =
          arma::eye(k, k);
    }
  }

  arma::cx_vec eigenvalues = arma::eig_gen(companion);
  return arma::all(arma::abs(eigenvalues) < 1);
}
