#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

//' Create Y and X Matrices
//'
//' This function creates the Y and X matrices.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions n x k,
//'   where n is the number of observations and k is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//'
//' @return List containing the Y and X matrices.
//'
//' @examples
//' set.seed(42)
//' time <- 100000L
//' burn_in <- 200
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
//' chol_cov <- chol(
//'   matrix(
//'     data = c(
//'       0.1, 0.0, 0.0,
//'       0.0, 0.1, 0.0,
//'       0.0, 0.0, 0.1
//'     ),
//'     nrow = k,
//'     byrow = TRUE
//'   )
//' )
//' y <- SimVAR(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov
//' )
//' yx <- YX(data = y, p = p)
//' str(yx)
//'
//' @details
//' The \code{YX} function creates the Y and X matrices required for fitting a Vector Autoregressive (VAR) model.
//' Given the input \code{data} matrix with dimensions n x k, where n is the number of observations and k is the number of variables,
//' and the order of the VAR model \code{p} (number of lags), the function constructs lagged predictor matrix X and the dependent variable matrix Y.
//' The matrices X and Y are used as inputs for estimating the VAR model parameters.
//'
//' The steps involved in creating the Y and X matrices are as follows:
//'
//' \itemize{
//'   \item Determine the number of observations \code{n} and the number of variables \code{k} from the input data matrix.
//'   \item Create matrices X and Y to store lagged variables and the dependent variable, respectively.
//'   \item Populate the matrices X and Y with the appropriate lagged data. The predictors matrix X contains the lagged values of the dependent variables,
//'     while the dependent variable matrix Y contains the original values of the dependent variables.
//' }
//'
//' The function returns a list containing the Y and X matrices, which can be used for further analysis and estimation of the VAR model parameters.
//'
//' @seealso
//' The \code{SimVAR} function for simulating time series data from a VAR model.
//'
//' @importFrom Rcpp sourceCpp
//'
//' @export
// [[Rcpp::export]]
List YX(arma::mat data, int p) {
  int n = data.n_rows; // Number of observations
  int k = data.n_cols; // Number of variables

  // Create matrices to store lagged variables and the dependent variable
  arma::mat X(n - p, k * p, arma::fill::zeros);
  arma::mat Y(n - p, k, arma::fill::zeros);

  // Populate the matrices X and Y with lagged data
  for (int i = 0; i < (n - p); i++) {
    int index = 0;
    // Arrange predictors from smallest lag to biggest
    for (int lag = p - 1; lag >= 0; lag--) {
      X.row(i).subvec(index, index + k - 1) = data.row(i + lag);
      index += k;
    }
    Y.row(i) = data.row(i + p);
  }

  // Create a list to store X, Y
  List result;
  result["X"] = X;
  result["Y"] = Y;

  return result;
}

/*** R
set.seed(42)
time <- 100000L
burn_in <- 200
k <- 3
p <- 2
constant <- c(1, 1, 1)
coef <- matrix(
  data = c(
    0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
    0.0, 0.0, 0.6, 0.0, 0.0, 0.3
  ),
  nrow = k,
  byrow = TRUE
)
chol_cov <- chol(
  matrix(
    data = c(
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1
    ),
    nrow = k,
    byrow = TRUE
  )
)
y <- SimVAR(
  time = time,
  burn_in = burn_in,
  constant = constant,
  coef = coef,
  chol_cov = chol_cov
)
yx <- YX(data = y, p = p)
str(yx)
*/
