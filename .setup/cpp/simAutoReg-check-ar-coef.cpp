// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-check-ar-coef.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Check AR(p) Coefficients for Stationarity
//'
//' This function checks for stationarity of the AR(p) coefficients.
//' Stationarity is determined based on the roots
//' of the autoregressive polynomial.
//' For a stationary AR(p) process,
//' all the roots of this autoregressive polynomial
//' must lie inside the unit circle in the complex plane.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef Numeric vector.
//'   Autoregressive coefficients.
//'
//' @examples
//' set.seed(42)
//' (coef <- SimARCoef(p = 2))
//' CheckARCoef(coef = coef)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg check ar
//' @export
// [[Rcpp::export]]
bool CheckARCoef(const arma::vec& coef) {
  // Step 1: Compute the roots of the characteristic polynomial
  arma::cx_vec roots = arma::roots(arma::join_cols(arma::vec{1}, -coef));

  // Step 2: Check if all roots have magnitudes less than 1
  //         (stability condition)
  return arma::all(arma::abs(roots) < 1);
}
