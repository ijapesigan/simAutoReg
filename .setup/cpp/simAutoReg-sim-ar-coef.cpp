// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-ar-coef.cpp
// Ivan Jacob Agaloos Pesigan
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
//' @keywords simAutoReg sim coef ar
//' @export
// [[Rcpp::export]]
arma::vec SimARCoef(int p) {
  // Step 1: Initialize a vector to store the generated stable
  //         autoregressive coefficients
  arma::vec coefs(p);

  // Step 2: Enter an infinite loop for coefficient generation
  //         and stability check
  while (true) {
    // Step 2.1: Generate random coefficients between -0.9 and 0.9
    arma::vec coefs = -0.9 + 1.8 * arma::randu<arma::vec>(p);

    // Step 2.2: Compute the roots of the characteristic polynomial
    //           of the autoregressive model
    arma::cx_vec roots = arma::roots(arma::join_cols(arma::vec{1}, -coefs));

    // Step 2.3: Check if all roots have magnitudes less than 1
    //           (stability condition)
    if (arma::all(arma::abs(roots) < 1)) {
      // Step 2.4: If the coefficients lead to a stable autoregressive model,
      //           exit the loop
      break;
    }
  }

  // Step 3: Return the generated stable autoregressive coefficients
  return coefs;
}
