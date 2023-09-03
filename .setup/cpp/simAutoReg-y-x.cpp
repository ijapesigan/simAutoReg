// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Create Y and X Matrices
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
//'
//' @return List containing the dependent variable (Y)
//' and predictor variable (X) matrices.
//' Note that the resulting matrices will have `t - p` rows.
//'
//' @examples
//' set.seed(42)
//' time <- 50L
//' burn_in <- 10L
//' k <- 3
//' p <- 2
//' constant <- c(1, 1, 1)
//' coef <- matrix(
//'   data = c(
//'     0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
//'     0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
//'     0.0, 0.0, 0.6, 0.0, 0.0, 0.3
//'   ),
//'   nrow = k,
//'   byrow = TRUE
//' )
//' chol_cov <- chol(diag(3))
//' y <- SimVAR(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov
//' )
//' yx <- YX(data = y, p = 2)
//' str(yx)
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
Rcpp::List YX(const arma::mat& data, int p) {
  // Step 1: Calculate the dimensions of the 'data' matrix
  // Number of time steps (rows)
  int time = data.n_rows;
  // Number of outcome variables (columns)
  int num_outcome_vars = data.n_cols;

  // Step 2: Create matrices 'X' and 'Y'
  //         to store transformed data 'X' matrix with ones
  arma::mat X(time - p, num_outcome_vars * p + 1, arma::fill::ones);
  // 'Y' matrix with zeros
  arma::mat Y(time - p, num_outcome_vars, arma::fill::zeros);

  // Step 3: Loop through the data and populate 'X' and 'Y'
  for (int time_index = 0; time_index < (time - p); time_index++) {
    // Initialize the column index for 'X'
    int index = 1;

    // Nested loop to populate 'X' with lagged values
    for (int lag = p - 1; lag >= 0; lag--) {
      // Update 'X' by assigning a subvector of 'data' to a subvector of 'X'
      X.row(time_index).subvec(index, index + num_outcome_vars - 1) =
          data.row(time_index + lag);
      // Move to the next set of columns in 'X'
      index += num_outcome_vars;
    }

    // Update 'Y' with the target values
    Y.row(time_index) = data.row(time_index + p);
  }

  // Step 4: Create an Rcpp List 'result' and assign 'X' and 'Y' matrices to it
  Rcpp::List result;
  result["X"] = X;
  result["Y"] = Y;

  // Step 5: Return the 'result' List containing the transformed data
  return result;
}

// Dependencies
// simAutoReg-sim-var.cpp
