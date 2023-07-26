#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

//' Simulate Data from a Vector Autoregressive (VAR) Model
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param time Integer.
//'   Number of time points to simulate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results.
//' @param constant Numeric vector.
//'   The constant term vector of length k, where k is the number of variables.
//' @param coef Numeric matrix.
//'   Coefficient matrix with dimensions k x (k * p).
//'   Each k x k block corresponds to the coefficient matrix
//'   for a particular lag.
//' @param chol_cov Numeric matrix.
//'   The Cholesky decomposition of the covariance matrix
//'   of the multivariate normal noise.
//'   It should have dimensions k x k.
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions k x (time - burn_in),
//'   where k is the number of variables and time is the number of observations.
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
//' head(y)
//'
//' @details
//' The \code{SimVAR} function generates synthetic time series data from a Vector Autoregressive (VAR) model.
//' The VAR model is defined by the constant term \code{constant}, the coefficient matrix \code{coef},
//' and the Cholesky decomposition of the covariance matrix of the multivariate normal noise \code{chol_cov}.
//' The generated time series data follows a VAR(p) process, where \code{p} is the number of lags specified by the size of \code{coef}.
//' The generated data includes a burn-in period, which is excluded before returning the results.
//'
//' The steps involved in generating the VAR time series data are as follows:
//'
//' \itemize{
//'   \item Extract the number of variables \code{k} and the number of lags \code{p} from the input.
//'   \item Create a matrix \code{data} of size \code{k} x (\code{time + burn_in}) to store the generated VAR time series data.
//'   \item Set the initial values of the matrix \code{data} using the constant term \code{constant}.
//'   \item For each time point starting from the \code{p}-th time point to \code{time + burn_in - 1}:
//'   \item Generate a vector of random noise from a multivariate normal distribution with mean 0 and covariance matrix \code{chol_cov}.
//'   \item Generate the VAR time series values for each variable \code{j} at time \code{i} using the formula:
//'   \deqn{Y_{ij} = constant_j + \sum_{l=1}^{p} \sum_{m=1}^{k} (coef_{jm} * Y_{im}) + \text{noise}_{j}}
//'   where \eqn{Y_{ij}} is the value of variable \code{j} at time \code{i},
//'   \code{constant_j} is the constant term for variable \code{j},
//'   \code{coef_{jm}} are the coefficients for variable \code{j} from lagged variables up to order \code{p},
//'   \eqn{Y_{im}} are the lagged values of variable \code{m} up to order \code{p} at time \code{i},
//'   and \code{noise_{j}} is the element \code{j} from the generated vector of random noise.
//'   \item Transpose the matrix \code{data} and return only the required time period after the burn-in period, which is from column \code{burn_in} to column \code{time + burn_in - 1}.
//' }
//'
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, arma::vec constant, arma::mat coef, arma::mat chol_cov) {
  int k = constant.n_elem; // Number of variables
  int coef_dim = coef.n_cols; // Dimension of the coefficient matrix
  int p = coef_dim / k; // Order of the VAR model (number of lags)

  // Matrix to store the generated VAR time series data
  arma::mat data(k, time + burn_in);

  // Set initial values using the constant term
  data.each_col() = constant; // Fill each column with the constant vector

  // Generate the VAR time series
  for (int i = p; i < time + burn_in; i++) {
    // Generate noise from a multivariate normal distribution
    arma::vec noise = arma::randn(k);
    arma::vec mult_noise = chol_cov * noise;

    // Generate eta_t vector
    for (int j = 0; j < k; j++) {
      for (int lag = 0; lag < p; lag++) {
        for (int l = 0; l < k; l++) {
          data(j, i) += coef(j, lag * k + l) * data(l, i - lag - 1);
        }
      }
      data(j, i) += mult_noise(j);
    }
  }

  // Transpose the data matrix and return only the required time period after burn-in
  return data.cols(burn_in, time + burn_in - 1).t();
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
head(y)
*/
