// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var-zip-exo.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive Zero-Inflated Poisson (VARZIP)
//' Model with Exogenous Variables
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive Zero-Inflated Poisson (VARZIP) model
//' with exogenous variables.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param time Integer.
//'   Number of time points to simulate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results.
//' @param constant Numeric vector.
//'   The constant term vector of length `k`,
//'   where `k` is the number of variables.
//' @param coef Numeric matrix.
//'   Coefficient matrix with dimensions `k` by `(k * p)`.
//'   Each `k` by `k` block corresponds to the coefficient matrix
//'   for a particular lag.
//' @param chol_cov Numeric matrix.
//'   The Cholesky decomposition of the covariance matrix
//'   of the multivariate normal noise.
//'   It should have dimensions `k` by `k`.
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous covariates with dimensions `time + burn_in` by `x`.
//'   Each column corresponds to a different exogenous variable.
//' @param exo_coef Numeric vector.
//'   Coefficient matrix with dimensions `k` by `x`
//'   associated with the exogenous covariates.
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables
//'   and `time` is the number of observations.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVARZIPExo2(int time, int burn_in, const arma::vec& constant,
                        const arma::mat& coef, const arma::mat& chol_cov,
                        const arma::mat& exo_mat, const arma::mat& exo_coef) {
  int k = constant.n_elem;  // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  int total_time = burn_in + time;

  // Matrix to store the generated VAR time series data
  arma::mat data(k, total_time);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Transpose exo_mat
  arma::mat exo_mat_t = exo_mat.t();

  // Generate the VAR time series
  for (int t = p; t < total_time; t++) {
    // Generate noise from a multivariate normal distribution
    arma::vec noise = arma::randn(k);
    arma::vec mult_noise = chol_cov * noise;

    // Generate eta_t vector
    for (int j = 0; j < k; j++) {
      // Compute autoregressive terms using matrix operations
      for (int lag = 0; lag < p; lag++) {
        data.row(j).subvec(t, t) +=
            coef.submat(j, lag * k, j, lag * k + k - 1) * data.col(t - lag - 1);
      }

      // Add exogenous variables' impact on autoregression variables
      data.row(j).subvec(t, t) += exo_mat_t * exo_coef.row(j).t();

      // Add noise
      data(j, t) += mult_noise(j);
    }

    // Use ZIP model for the first variable
    double intensity = std::exp(data(0, t));
    if (R::runif(0, 1) < intensity / (1 + intensity)) {
      // Sample from the point mass at zero (inflation)
      data(0, t) = 0;
    } else {
      // Sample from the Poisson distribution (count process)
      data(0, t) = R::rpois(intensity);
    }
  }

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}

// Dependencies
