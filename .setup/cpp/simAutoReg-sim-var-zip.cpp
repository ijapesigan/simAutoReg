// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var-zip.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".SimVARZIPCpp")]]
arma::mat SimVARZIPCpp(int time, const arma::vec& constant,
                       const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  // Matrix to store the generated VAR time series data
  arma::mat data(k, time);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < time; t++) {
    // Generate noise from a multivariate normal distribution
    arma::vec noise = arma::randn(k);
    arma::vec mult_noise = chol_cov * noise;

    // Generate eta_t vector
    for (int j = 0; j < k; j++) {
      for (int lag = 0; lag < p; lag++) {
        for (int l = 0; l < k; l++) {
          data(j, t) += coef(j, lag * k + l) * data(l, t - lag - 1);
        }
      }

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

  return data.t();
}

// Dependencies
