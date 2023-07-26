#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from an Autoregressive Model with Constant Term
//'
//' This function generates synthetic time series data
//' from an autoregressive (AR) model.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param time Integer.
//'   Number of time points to simulate.
//' @param burn_in Integer.
//'   Number of burn-in periods before recording data.
//' @param constant Numeric.
//'   The constant term of the AR model.
//' @param coef Numeric vector.
//'   Autoregressive coefficients.
//' @param sd Numeric.
//'   The standard deviation of the random noise.
//'
//' @return Numeric vector containing the simulated time series data.
//'
//' @examples
//' set.seed(42)
//' SimAR(time = 10, burn_in = 5, constant = 2, coef = c(0.5, -0.3), sd = 0.1)
//'
//' @details
//' The \code{SimAR} function generates synthetic time series data from an autoregressive (AR) model.
//' The generated data follows the AR(p) model, where \code{p} is the number of coefficients specified in \code{coef}.
//' The generated time series data includes a constant term and autoregressive terms based on the provided coefficients.
//' Random noise, sampled from a normal distribution with mean 0 and standard deviation \code{sd}, is added to the time series.
//' Additionally, a burn-in period can be specified to exclude initial data points from the output. 
//'
//' The steps in generating the autoregressive time series with burn-in are as follows:
//'
//' \itemize{
//'   \item Set the order of the AR model to \code{p}.
//'   \item Generate random noise from a normal distribution with mean 0 and standard deviation \code{sd}.
//'   \item Generate the autoregressive time series with burn-in using the formula:
//'   \deqn{Y_t = constant + \sum_{i=1}^{p} (coef[i] * Y_{t-i}) + noise_t}
//'   where \eqn{Y_t} is the time series data at time \eqn{t}, \eqn{constant} is the constant term, 
//'   \eqn{coef[i]} are the autoregressive coefficients, \eqn{Y_{t-i}} are the lagged data points up to order \code{p},
//'   and \eqn{noise_t} is the random noise at time \eqn{t}.
//'   \item Optionally, remove the burn-in period from the generated time series data.
//' }
//'
//' @export
// [[Rcpp::export]]
arma::vec SimAR(int time, int burn_in, double constant, arma::vec coef, double sd) {
  // Order of the AR model
  int p = coef.size();

  // Vector to store the generated time series data
  arma::vec data(time);

  // Generate random noise from a normal distribution
  arma::vec noise(time);
  for (int i = 0; i < time; i++) {
    noise(i) = R::rnorm(0, sd);
  }
  
  // Generate the autoregressive time series with burn-in
  for (int i = 0; i < time; i++) {
    data(i) = constant;
    for (int lag = 0; lag < p; lag++) {
      if (i - lag - 1 >= 0) {
        data(i) += coef(lag) * data(i - lag - 1) + noise(i);
      }
    }
  }
  
  // Remove the burn-in period
  if (burn_in > 0) {
    data = data(arma::span(burn_in, time - 1));
  }

  return data;
}

/*** R
SimAR(time = 10, burn_in = 5, constant = 2, coef = c(0.5, -0.3), sd = 0.1)
*/
