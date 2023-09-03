// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var-coef.cpp
// Ivan Jacob Agaloos Pesigan
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
//' @param k Positive integer.
//'   Number of autoregressive variables.
//' @param p Positive integer.
//'   Number of lags.
//'
//' @examples
//' set.seed(42)
//' SimVARCoef(k = 3, p = 2)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim coef var
//' @export
// [[Rcpp::export]]
arma::mat SimVARCoef(int k, int p) {
  // Step 1: Create a matrix to store VAR coefficients
  arma::mat coefs(k, k * p);

  // Step 2: Generate random coefficients between -0.9 and 0.9
  while (true) {
    // Generate random values in [0, 1]
    coefs.randu();
    // Scale and shift values to [-0.9, 0.9]
    coefs = -0.9 + 1.8 * coefs;

    // Step 3: Create a companion matrix from VAR coefficients
    arma::mat companion_matrix(k * p, k * p, arma::fill::zeros);

    // Step 4: Fill the companion matrix using VAR coefficients
    for (int i = 0; i < p; i++) {
      // Fill the diagonal block of the companion matrix with VAR coefficients
      companion_matrix.submat(i * k, i * k, (i + 1) * k - 1, (i + 1) * k - 1) =
          coefs.cols(i * k, (i + 1) * k - 1);

      // Fill the sub-diagonal block of the companion matrix
      // with identity matrices
      if (i > 0) {
        companion_matrix.submat(i * k, (i - 1) * k, (i + 1) * k - 1,
                                i * k - 1) = arma::eye(k, k);
      }
    }

    // Step 5: Compute the eigenvalues of the companion matrix
    arma::cx_vec eigenvalues = arma::eig_gen(companion_matrix);

    // Step 6: Check if all eigenvalues are inside the unit circle
    if (arma::all(arma::abs(eigenvalues) < 1)) {
      // Exit the loop if all eigenvalues satisfy the condition
      break;
    }
  }

  // Step 7: Return the generated VAR coefficients
  return coefs;
}
