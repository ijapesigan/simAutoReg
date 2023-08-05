// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".YXCpp")]]
Rcpp::List YXCpp(const arma::mat& data, int p) {
  int t = data.n_rows;  // Number of observations
  int k = data.n_cols;  // Number of variables

  // Create matrices to store lagged variables and the dependent variable
  arma::mat X(t - p, k * p + 1, arma::fill::zeros);  // Add 1 column for the
                                                     // constant
  arma::mat Y(t - p, k, arma::fill::zeros);

  // Populate the matrices X and Y with lagged data
  for (int i = 0; i < (t - p); i++) {
    X(i, 0) = 1;  // Set the first column to 1 for the constant term
    int index = 1;
    // Arrange predictors from smallest lag to biggest
    for (int lag = p - 1; lag >= 0; lag--) {
      X.row(i).subvec(index, index + k - 1) = data.row(i + lag);
      index += k;
    }
    Y.row(i) = data.row(i + p);
  }

  // Create a list to store X, Y
  Rcpp::List result;
  result["X"] = X;
  result["Y"] = Y;

  return result;
}

// Dependencies
// simAutoReg-sim-var.cpp
