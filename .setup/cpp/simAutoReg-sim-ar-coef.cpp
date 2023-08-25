// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-ar-coef.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Random Autoregressive Coefficients for a Stationary AR(p) Model
//'
//' This function generates autoregressive coefficients
//' for a stationary AR(p) model.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param p Positive integer. Number of lags.
//'
//' @examples
//' set.seed(42)
//' SimARCoef(p = 2)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::vec SimARCoef(int p) {
  bool is_stationary = false;
  arma::vec ar_coefficients(p);

  while (!is_stationary) {
    // Generate random coefficients between -0.9 and 0.9
    arma::vec random_coeffs = arma::randu<arma::vec>(p);
    ar_coefficients = -0.9 + 1.8 * random_coeffs;

    // Check if the roots lie inside the unit circle
    arma::cx_vec roots =
        arma::roots(arma::join_cols(arma::vec{1}, -ar_coefficients));
    if (arma::all(arma::abs(roots) < 1)) {
      is_stationary = true;
    }
  }

  return ar_coefficients;
}
