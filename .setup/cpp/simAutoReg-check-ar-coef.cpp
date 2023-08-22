// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-check-ar-coef.cpp
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
//' CheckARCoef(SimARCoef(p = 2))
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
bool CheckARCoef(const arma::vec& coef) {
  // Check if the roots lie inside the unit circle
  arma::cx_vec roots = arma::roots(arma::join_cols(arma::vec{1}, -coef));
  return arma::all(arma::abs(roots) < 1);
}
