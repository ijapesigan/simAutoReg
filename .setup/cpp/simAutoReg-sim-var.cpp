// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".SimVARCpp")]]
arma::mat SimVARCpp(int time, const arma::vec& constant, const arma::mat& coef,
                    const arma::mat& chol_cov) {
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
  }

  return data.t();
}

// Dependencies