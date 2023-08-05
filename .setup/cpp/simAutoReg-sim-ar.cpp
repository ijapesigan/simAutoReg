// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-ar.cpp
// -----------------------------------------------------------------------------

#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export(name = ".SimARCpp")]]
arma::vec SimARCpp(int time, const double& constant, const arma::vec& coef,
                   const double& sd) {
  // Order of the AR model
  int p = coef.size();

  // Vector to store the generated time series data
  arma::vec data(time);

  // Generate random noise from a normal distribution
  arma::vec noise(time);
  for (int i = 0; i < time; i++) {
    noise(i) = R::rnorm(0, sd);
  }

  // Generate the autoregressive time series
  for (int i = 0; i < time; i++) {
    data(i) = constant;
    for (int lag = 0; lag < p; lag++) {
      if (i - lag - 1 >= 0) {
        data(i) += coef(lag) * data(i - lag - 1) + noise(i);
      }
    }
  }

  return data;
}

// Dependencies
