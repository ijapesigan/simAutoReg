// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Create Y and X Matrices with Exogenous Variables
//'
//' This function creates the dependent variable (Y)
//' and predictor variable (X) matrices.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous variables with dimensions `t` by `m`.
//'
//' @return List containing the dependent variable (Y)
//' and predictor variable (X) matrices.
//' Note that the resulting matrices will have `t - p` rows.
//'
//' @details
//' The [YX()] function creates the `Y` and `X` matrices
//' required for fitting a Vector Autoregressive (VAR) model.
//' Given the input `data` matrix with dimensions `t` by `k`,
//' where `t` is the number of observations and `k` is the number of variables,
//' and the order of the VAR model `p` (number of lags),
//' the function constructs lagged predictor matrix `X`
//' and the dependent variable matrix `Y`.
//'
//' The steps involved in creating the `Y` and `X` matrices are as follows:
//'
//' - Determine the number of observations `t` and the number of variables `k`
//'   from the input data matrix.
//' - Create matrices `X` and `Y` to store lagged variables
//'   and the dependent variable, respectively.
//' - Populate the matrices `X` and `Y` with the appropriate lagged data.
//'   The predictors matrix `X` contains a column of ones
//'   and the lagged values of the dependent variables,
//'   while the dependent variable matrix `Y` contains the original values
//'   of the dependent variables.
//' - The function returns a list containing the `Y` and `X` matrices,
//'   which can be used for further analysis and estimation
//'   of the VAR model parameters.
//'
//' @seealso
//' The [SimVAR()] function for simulating time series data
//' from a VAR model.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat) {
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
