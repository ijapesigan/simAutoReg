// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-ar.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

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
//'   The standard deviation of the random process noise.
//'
//' @return Numeric vector (column matrix) containing the simulated
//'   time series data.
//'
//' @examples
//' set.seed(42)
//' SimAR(time = 10, burn_in = 5, constant = 2, coef = c(0.5, -0.3), sd = 0.1)
//'
//' @details
//' The [SimAR()] function generates synthetic time series data
//' from an autoregressive (AR) model.
//' The generated data follows the AR(p) model,
//' where `p` is the number of coefficients specified in `coef`.
//' The generated time series data includes a constant term
//' and autoregressive terms based on the provided coefficients.
//' Random noise, sampled from a normal distribution with mean 0
//' and standard deviation `sd`, is added to the time series.
//' A burn-in period is specified to exclude initial data points
//' from the output.
//'
//' The steps in generating the autoregressive time series with burn-in
//' are as follows:
//'
//' - Set the order of the AR model to `p` based on the length of `coef`.
//' - Create a vector data of size `time + burn_in`
//'   to store the generated AR time series data.
//' - Create a vector data of size `time + burn_in` of random process noise
//'   from a normal distribution with mean 0
//'   and standard deviation `sd`.
//' - Generate the autoregressive time series with burn-in using the formula:
//'   \deqn{
//'     Y_t = \mathrm{constant} +
//'     \sum_{i = 1}^{p} \left( \mathrm{coef}_i * Y_{t - i} \right) +
//'     \mathrm{noise}_t
//'   }
//'   where \eqn{Y_t} is the time series data at time \eqn{t},
//'   \eqn{\mathrm{constant}} is the constant term,
//'   \eqn{\mathrm{coef}_i} are the autoregressive coefficients,
//'   \eqn{Y_{t - i}} are the lagged data points up to order `p`,
//'   and \eqn{\mathrm{noise}_t} is the random noise at time \eqn{t}.
//' - Remove the burn-in period from the generated time series data.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data ar
//' @export
// [[Rcpp::export]]
arma::vec SimAR(int time, int burn_in, const double& constant,
                const arma::vec& coef, const double& sd) {
  // Step 1: Determine the number of lags and total time
  // Number of lags in the autoregressive model
  int num_lags = coef.size();
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a vector to store simulated autoregressive data
  // Initialize with ones to represent a constant term
  arma::vec data(total_time, arma::fill::ones);

  // Step 3: Set the initial values of the data vector using the constant term
  data *= constant;

  // Step 4: Generate a vector of random noise
  arma::vec noise = arma::randn(total_time);

  // Step 5: Simulate autoregressive data using a loop
  for (int time_index = 0; time_index < total_time; time_index++) {
    // Step 5.1: Iterate over lags and apply the autoregressive formula
    for (int lag = 0; lag < num_lags; lag++) {
      if (time_index - lag - 1 >= 0) {
        data(time_index) +=
            coef(lag) * data(time_index - lag - 1) + noise(time_index);
      }
    }
  }

  // Step 6: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data(arma::span(burn_in, total_time - 1));
  }

  // Step 7: Return the simulated autoregressive data
  return data;
}

// Dependencies
