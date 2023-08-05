// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".YXExoCpp")]]
Rcpp::List YXExoCpp(const arma::mat& data, int p, const arma::mat& exo_mat) {
  int t = data.n_rows;     // Number of observations
  int k = data.n_cols;     // Number of variables
  int m = exo_mat.n_cols;  // Number of exogenous variables

  // Create matrices to store lagged variables and exogenous variables
  arma::mat X(t - p, k * p + m + 1,
              arma::fill::ones);  // Add m columns for the exogenous variables
                                  // and 1 column for the constant
  arma::mat Y(t - p, k, arma::fill::zeros);

  // Populate the matrices X and Y with lagged data and exogenous data
  for (int i = 0; i < (t - p); i++) {
    int index = 1;
    // Arrange predictors from smallest lag to biggest
    for (int lag = p - 1; lag >= 0; lag--) {
      X(i, arma::span(index, index + k - 1)) = data.row(i + lag);
      index += k;
    }
    // Append the exogenous variables to X
    X(i, arma::span(index, index + m - 1)) = exo_mat.row(i + p);
    Y.row(i) = data.row(i + p);
  }

  // Create a list to store Y and X
  Rcpp::List result;
  result["Y"] = Y;
  result["X"] = X;

  return result;
}
