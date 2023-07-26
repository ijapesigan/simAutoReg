#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters using OLS
//'
//' This function estimates the parameters of a VAR model
//' using the Ordinary Least Squares (OLS) method.
//' The OLS method is used to estimate the autoregressive
//' and cross-regression coefficients.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of lagged predictors (X).
//'
//' @return Matrix of estimated autoregressive and cross-regression coefficients.
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
//' yx <- YX(y, p)
//' FitVAROLS(Y = yx$Y, X = yx$X)
//'
//' @details
//' The \code{FitVAROLS} function estimates the parameters of a Vector Autoregressive (VAR) model using the Ordinary Least Squares (OLS) method.
//' Given the input matrices \code{Y} and \code{X}, where \code{Y} is the matrix of dependent variables, and \code{X} is the matrix of lagged predictors,
//' the function computes the autoregressive and cross-regression coefficients of the VAR model.
//'
//' The steps involved in estimating the VAR model parameters using OLS are as follows:
//'
//' \itemize{
//'   \item Compute the QR decomposition of the lagged predictor matrix \code{X} using the \code{qr_econ} function from the Armadillo library.
//'   \item Extract the \code{Q} and \code{R} matrices from the QR decomposition.
//'   \item Solve the linear system \code{R * coef = Q.t() * Y} to estimate the VAR model coefficients \code{coef}.
//' }
//'
//' The function returns a matrix containing the estimated autoregressive and cross-regression coefficients of the VAR model.
//'
//' @seealso
//' The \code{qr_econ} function from the Armadillo library for QR decomposition.
//'
//' @importFrom Rcpp sourceCpp
//'
//' @export
// [[Rcpp::export]]
arma::mat FitVAROLS(arma::mat Y, arma::mat X) {
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr_econ(Q, R, X);

  // Solve the linear system R * coef = Q.t() * Y
  arma::mat coef = arma::solve(R, Q.t() * Y);

  return coef;
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
yx <- YX(y, p)
FitVAROLS(Y = yx$Y, X = yx$X)
*/
