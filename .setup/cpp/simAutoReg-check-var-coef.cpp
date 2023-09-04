// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-check-var-coef.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

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
//' (coef <- SimVARCoef(k = 3, p = 2))
//' CheckVARCoef(coef = coef)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg check var
//' @export
// [[Rcpp::export]]
bool CheckVARCoef(const arma::mat& coef) {
  // Step 1: Determine the number of outcome variables and lags
  // Number of outcome variables
  int num_outcome_vars = coef.n_rows;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;

  // Step 2: Create a companion matrix for the VAR coefficients
  arma::mat companion_matrix(num_outcome_vars * num_lags, num_outcome_vars * num_lags, arma::fill::zeros);

  // Step 3: Fill the companion matrix using VAR coefficients
  for (int i = 0; i < num_lags; i++) {
    // Step 3.1: Fill the diagonal block of the companion matrix
    //           with VAR coefficients
    companion_matrix.submat(i * num_outcome_vars, i * num_outcome_vars, (i + 1) * num_outcome_vars - 1, (i + 1) * num_outcome_vars - 1) = coef.cols(i * num_outcome_vars, (i + 1) * num_outcome_vars - 1);

    // Step 3.2: Fill the sub-diagonal block with identity matrices (lags > 0)
    if (i > 0) {
      companion_matrix.submat(i * num_outcome_vars, (i - 1) * num_outcome_vars, (i + 1) * num_outcome_vars - 1, i * num_outcome_vars - 1) = arma::eye(num_outcome_vars, num_outcome_vars);
    }
  }

  // Step 4: Compute the eigenvalues of the companion matrix
  arma::cx_vec eigenvalues = arma::eig_gen(companion_matrix);

  // Step 5: Check if all eigenvalues have magnitudes less than 1
  //         (stability condition)
  return arma::all(arma::abs(eigenvalues) < 1);
}
