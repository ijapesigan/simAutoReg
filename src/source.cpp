#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat SimMVN(int n, const arma::vec& location, const arma::mat& chol_scale);

arma::mat SimMVNFixed(int n, const arma::vec& location, const arma::mat& scale);

arma::vec SimAR(int time, int burn_in, const double& constant,
                const arma::vec& coef, const double& sd);

arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov);

arma::mat SimVARZIP(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov);

arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov,
                    const arma::mat& exo_mat, const arma::mat& exo_coef);

arma::mat SimVARZIPExo(int time, int burn_in, const arma::vec& constant,
                       const arma::mat& coef, const arma::mat& chol_cov,
                       const arma::mat& exo_mat, const arma::mat& exo_coef);

arma::mat SimVariance(int n, const arma::vec& location,
                      const arma::mat& chol_scale);

Rcpp::List YX(const arma::mat& data, int p);

Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat);
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-ar.cpp
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
//' The [simAutoReg::SimAR()] function generates synthetic time series data
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
//'   \deqn{Y_t = constant + \sum_{i=1}^{p} (coef[i] * Y_{t-i}) + noise_t}
//'   where \eqn{Y_t} is the time series data at time \eqn{t}, \eqn{constant}
//'   is the constant term,
//'   \eqn{coef[i]} are the autoregressive coefficients,
//'   \eqn{Y_{t-i}} are the lagged data points up to order `p`,
//'   and \eqn{noise_t} is the random noise at time \eqn{t}.
//' - Remove the burn-in period from the generated time series data.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::vec SimAR(int time, int burn_in, const double& constant,
                const arma::vec& coef, const double& sd) {
  // Order of the AR model
  int p = coef.size();
  int total_time = burn_in + time;

  // Vector to store the generated time series data
  arma::vec data(total_time);

  // Generate random noise from a normal distribution
  arma::vec noise(total_time);
  for (int i = 0; i < total_time; i++) {
    noise(i) = R::rnorm(0, sd);
  }

  // Generate the autoregressive time series
  for (int i = 0; i < total_time; i++) {
    data(i) = constant;
    for (int lag = 0; lag < p; lag++) {
      if (i - lag - 1 >= 0) {
        data(i) += coef(lag) * data(i - lag - 1) + noise(i);
      }
    }
  }

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data(arma::span(burn_in, total_time - 1));
  }

  return data;
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-mvn-fixed.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Multivariate Normal Random Numbers with Optional Fixed Values
//'
//' This function generates multivariate normal random numbers
//' with optinal fixed values where the variance is zero.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param n Integer.
//'   Number of samples to generate.
//' @param location Numeric vector.
//'   Mean vector of length `k`, where `k` is the number of variables.
//' @param scale Numeric matrix.
//'   Covariance matrix of dimensions `k` by `k`.
//'   Values for variables with variance of `0` will be fixed
//'   to the corresponding value in `location`.
//'
//' @return Matrix containing the simulated multivariate normal random numbers,
//'   with dimensions `n` by `k`, where `n` is the number of samples
//'   and `k` is the number of variables.
//'
//' @examples
//' set.seed(42)
//' n <- 1000L
//' location <- c(0.5, -0.2, 0.1)
//' scale <- matrix(
//'   data = c(0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
//'   nrow = 3,
//'   byrow = TRUE
//' )
//' y <- SimMVNFixed(n = n, location = location, scale = scale)
//' colMeans(y)
//' var(y)
//'
//' @details
//' The [simAutoReg::SimMVNFixed()] function first identifies the indices
//' of non-constant variables (i.e., variables with variance not equal to 0)
//' in the covariance matrix.
//' It then extracts the non-constant elements from the mean vector
//' and the covariance matrix.
//' A Cholesky decomposition is performed on the covariance matrix
//' of non-constant variables.
//' Random samples are generated for the non-constant variables
//' using the Cholesky factor.
//' The generated data matrix is constructed
//' by setting the non-constant variables
//' and constant variables to their corresponding values.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimMVNFixed(int n, const arma::vec& location,
                      const arma::mat& scale) {
  // Get the number of variables
  int k = location.n_elem;

  // Find indices of non-constant variables (variance not equal to 0)
  arma::uvec non_constant_indices = arma::find(arma::diagvec(scale) != 0);

  // Extract non-constant elements from location and scale matrix
  arma::vec non_constant_location = location(non_constant_indices);
  arma::mat non_constant_scale =
      scale(non_constant_indices, non_constant_indices);

  // Perform Cholesky decomposition on the scale matrix of non-constant
  // variables
  arma::mat L = arma::chol(non_constant_scale);

  // Generate random samples for the non-constant variables
  arma::mat samples = arma::randn(n, non_constant_indices.n_elem) * L.t() +
                      arma::repmat(non_constant_location.t(), n, 1);

  // Create a matrix to store the generated data
  arma::mat data(n, k, arma::fill::zeros);

  // Set the non-constant variables in the data matrix
  for (arma::uword i = 0; i < non_constant_indices.n_elem; i++) {
    data.col(non_constant_indices(i)) = samples.col(i);
  }

  // Set constant variables to their corresponding values
  for (int i = 0; i < k; i++) {
    if (arma::as_scalar(scale(i, i)) == 0) {
      data.col(i).fill(arma::as_scalar(location(i)));
    }
  }

  return data;
}
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-mvn.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Multivariate Normal Random Numbers
//'
//' This function generates multivariate normal random numbers.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param n Integer.
//'   Number of samples to generate.
//' @param location Numeric vector.
//'   Mean vector of length `k`, where `k` is the number of variables.
//' @param chol_scale Numeric matrix.
//'   Cholesky decomposition of the covariance matrix of dimensions `k` by `k`.
//'
//' @return Matrix containing the simulated multivariate normal random numbers,
//'   with dimensions `n` by `k`, where `n` is the number of samples
//'   and `k` is the number of variables.
//'
//' @examples
//' set.seed(42)
//' n <- 1000L
//' location <- c(0.5, -0.2, 0.1)
//' scale <- matrix(
//'   data = c(1.0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
//'   nrow = 3,
//'   byrow = TRUE
//' )
//' chol_scale <- chol(scale)
//' y <- SimMVN(n = n, location = location, chol_scale = chol_scale)
//' colMeans(y)
//' var(y)
//'
//' @details
//' The [simAutoReg::SimMVN()] function generates
//' multivariate normal random numbers
//' using the Cholesky decomposition method.
//' Given the number of samples `n`, the mean vector `location` of length `k`
//' (where `k` is the number of variables),
//' and the Cholesky decomposition `chol_scale` of the covariance matrix
//' of dimensions `k` by `k`,
//' the function produces a matrix of multivariate normal random numbers.
//'
//' The steps involved in generating multivariate normal random numbers
//' are as follows:
//'
//' - Determine the number of variables `k` from the length of the mean vector.
//' - Generate random data from a standard multivariate normal distribution,
//'   resulting in an `n` by `k` matrix of random numbers.
//' - Transform the standard normal random data
//'   into multivariate normal random data
//'   using the Cholesky decomposition `chol_scale`.
//' - Add the mean vector `location` to the transformed data
//'   to obtain the final simulated multivariate normal random numbers.
//' - The function returns a matrix of simulated
//'   multivariate normal random numbers
//'   with dimensions `n` by `k`,
//'   where `n` is the number of samples and `k` is the number of variables.
//'   This matrix can be used for various statistical analyses and simulations.
//'
//' @seealso
//' The [chol()] function in R to obtain the Cholesky decomposition
//' of a covariance matrix.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimMVN(int n, const arma::vec& location,
                 const arma::mat& chol_scale) {
  int k = location.n_elem;

  // Generate multivariate normal random numbers
  arma::mat random_data = arma::randn(n, k);
  arma::mat data = random_data * chol_scale + arma::repmat(location.t(), n, 1);

  return data;
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var-exo.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive (VAR) Model with Exogenous
// Variables
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model with exogenous variables.
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
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant,
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
      for (int lag = 0; lag < p; lag++) {
        for (int l = 0; l < k; l++) {
          data(j, t) += coef(j, lag * k + l) * data(l, t - lag - 1);
        }
      }

      // Add exogenous variables' impact on autoregression variables
      for (arma::uword x = 0; x < exo_mat_t.n_rows; x++) {
        data(j, t) += exo_mat_t(x, t) * exo_coef(j, x);
      }

      data(j, t) += mult_noise(j);
    }
  }

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}
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
arma::mat SimVARZIPExo(int time, int burn_in, const arma::vec& constant,
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
      for (int lag = 0; lag < p; lag++) {
        for (int l = 0; l < k; l++) {
          data(j, t) += coef(j, lag * k + l) * data(l, t - lag - 1);
        }
      }

      // Add exogenous variables' impact on autoregression variables
      for (arma::uword x = 0; x < exo_mat_t.n_rows; x++) {
        data(j, t) += exo_mat_t(x, t) * exo_coef(j, x);
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

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var-zip.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive Zero-Inflated Poisson (VARZIP)
//' Model
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive Zero-Inflated Poisson (VARZIP) model.
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
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables
//'   and `time` is the number of observations.
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
//' y <- SimVARZIP(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov
//' )
//' head(y)
//'
//' @details
//' The [simAutoReg::SimVARZIP()] function generates synthetic time series data
//' from a Vector Autoregressive (VAR)
//' with Zero-Inflated Poisson (ZIP) model for the first observed variable.
//' See [simAutoReg::SimVAR()] for more details on generating data for VAR(p).
//' The `SimVARZIP` function goes further by using the generated values
//' for the first variable to generate data from the ZIP model.
//' The exponential of the values from the first variable
//' from the original VAR(p) model
//' are used as the `intensity` parameter in the Poisson distribution
//' in the ZIP model.
//' Data from the ZIP model are used to replace the original values
//' for the first variable.
//' Values for the rest of the variables are unchanged.
//' The generated data includes a burn-in period,
//' which is excluded before returning the results.
//'
//' The steps involved in generating the time series data are as follows:
//'
//' - Extract the number of variables `k`
//'   and the number of lags `p` from the input.
//' - Create a matrix `data` of size `k` x (`time + burn_in`)
//'   to store the generated data.
//' - Set the initial values of the matrix `data`
//'   using the constant term `constant`.
//' - For each time point starting from the `p`-th time point
//'   to `time + burn_in - 1`:
//'   * Generate a vector of random process noise
//'     from a multivariate normal distribution
//'     with mean 0 and covariance matrix `chol_cov`.
//'   * Generate the VAR time series values for each variable `j`
//'     at time `t` by applying the autoregressive terms
//'     for each lag `lag` and each variable `l`.
//'     - Add the generated noise to the VAR time series values.
//'     - For the first variable,
//'       apply the Zero-Inflated Poisson (ZIP) model:
//'       * Compute the intensity `intensity`
//'         as the exponential of the first variable's value at time `t`.
//'       * Sample a random value `u`
//'         from a uniform distribution on \[0, 1\].
//'       * If `u` is less than `intensity / (1 + intensity)`,
//'         set the first variable's value to zero (inflation).
//'       * Otherwise, sample the first variable's value
//'         from a Poisson distribution
//'         with mean `intensity` (count process).
//' - Transpose the data matrix `data` and return only
//'   the required time period after burn-in as a numeric matrix.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVARZIP(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  int total_time = burn_in + time;

  // Matrix to store the generated VAR time series data
  arma::mat data(k, total_time);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < total_time; t++) {
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

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

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
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
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
//' head(y)
//'
//' @details
//' The [simAutoReg::SimVAR()] function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model.
//' The VAR model is defined by the constant term `constant`,
//' the coefficient matrix `coef`,
//' and the Cholesky decomposition of the covariance matrix
//' of the multivariate normal process noise `chol_cov`.
//' The generated time series data follows a VAR(p) process,
//' where `p` is the number of lags specified by the size of `coef`.
//' The generated data includes a burn-in period,
//' which is excluded before returning the results.
//'
//' The steps involved in generating the VAR time series data are as follows:
//'
//' - Extract the number of variables `k` and the number of lags `p`
//'   from the input.
//' - Create a matrix `data` of size `k` by (`time + burn_in`)
//'   to store the generated VAR time series data.
//' - Set the initial values of the matrix `data`
//'   using the constant term `constant`.
//' - For each time point starting from the `p`-th time point
//'   to `time + burn_in - 1`:
//'   * Generate a vector of random noise
//'     from a multivariate normal distribution
//'     with mean 0 and covariance matrix `chol_cov`.
//'   * Generate the VAR time series values for each variable `j` at time `t`
//'     using the formula:
//'     \deqn{Y_{tj} = constant_j +
//'     \sum_{l = 1}^{p} \sum_{m = 1}^{k} (coef_{jm} * Y_{im}) +
//'     \text{noise}_{j}}
//'     where \eqn{Y_{tj}} is the value of variable `j` at time `t`,
//'     \eqn{constant_j} is the constant term for variable `j`,
//'     \eqn{coef_{jm}} are the coefficients for variable `j`
//'     from lagged variables up to order `p`,
//'     \eqn{Y_{tm}} are the lagged values of variable `m`
//'     up to order `p` at time `t`,
//'     and \eqn{noise_{j}} is the element `j`
//'     from the generated vector of random process noise.
//' - Transpose the matrix `data` and return only
//'   the required time period after the burn-in period,
//'   which is from column `burn_in` to column `time + burn_in - 1`.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  int total_time = burn_in + time;

  // Matrix to store the generated VAR time series data
  arma::mat data(k, total_time);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < total_time; t++) {
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

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-variance.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Generate Random Data for the Variance Vector
//'
//' This function generates random data for the variance vector given by
//' \deqn{
//'   \boldsymbol{\sigma^{2}} =
//'   \exp \left( \boldsymbol{\mu} + \boldsymbol{\varepsilon} \right)
//'   \quad
//'   \text{with}
//'   \quad
//'   \boldsymbol{\varepsilon} \sim
//'   \mathcal{N} \left( \boldsymbol{0}, \boldsymbol{\Sigma} \right)
//' }.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param n Integer.
//'   Number of samples to generate.
//' @param location Numeric vector.
//'   The constant term \eqn{\boldsymbol{\mu}}.
//' @param chol_scale Numeric matrix.
//'   Cholesky decomposition of the covariance matrix \eqn{\boldsymbol{\Sigma}}
//'   for the multivariate normal random error \eqn{\boldsymbol{\varepsilon}}.
//'
//' @return Matrix with each row containing the simulated variance vector
//'   for each sample.
//'
//' @details
//' The [simAutoReg::SimVariance()] function generates random data
//' for the variance vector
//' based on the exponential of a multivariate normal distribution.
//' Given the number of samples `n`,
//' the constant term \eqn{\boldsymbol{\mu}} represented
//' by the `location` vector,
//' and the Cholesky decomposition matrix \eqn{\boldsymbol{\Sigma}}
//' for the multivariate normal random error \eqn{\boldsymbol{\varepsilon}},
//' the function simulates \eqn{n} independent samples
//' of the variance vector \eqn{\boldsymbol{\sigma^{2}}}.
//' Each sample of the variance vector \eqn{\boldsymbol{\sigma^{2}}}
//' is obtained by
//' calculating the exponential of random variations
//' to the mean vector \eqn{\boldsymbol{\mu}}.
//' The random variations are generated using the Cholesky decomposition
//' of the covariance matrix \eqn{\boldsymbol{\Sigma}}.
//' Finally, the function returns a matrix with each column
//' containing the simulated
//' variance vector for each sample.
//'
//' @examples
//' set.seed(42)
//' n <- 100
//' location <- c(0.5, -0.2, 0.1)
//' chol_scale <- chol(
//'   matrix(
//'     data = c(1.0, 0.3, 0.3, 0.3, 1.0, 0.2, 0.3, 0.2, 1.0),
//'     nrow = 3,
//'     byrow = TRUE
//'   )
//' )
//' SimVariance(n = n, location = location, chol_scale = chol_scale)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVariance(int n, const arma::vec& location,
                      const arma::mat& chol_scale) {
  // Generate multivariate normal random numbers
  arma::mat mvn = SimMVN(n, location, chol_scale);

  // Compute the variance vector for each sample
  arma::mat variance = arma::exp(mvn);

  return variance;
}

// Dependencies
// simAutoReg-sim-mvn.cpp
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Create Y and X Matrices with Exogenous Variables
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
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous variables with dimensions `t` by `m`.
//'
//' @return List containing the dependent variable (Y)
//' and predictor variable (X) matrices.
//' Note that the resulting matrices will have `t - p` rows.
//'
//' @details
//' The [simAutoReg::YX()] function creates the `Y` and `X` matrices
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
//' The [simAutoReg::SimVAR()] function for simulating time series data
//' from a VAR model.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat) {
  int t = data.n_rows;     // Number of observations
  int k = data.n_cols;     // Number of variables
  int m = exo_mat.n_cols;  // Number of exogenous variables

  // Create matrices to store lagged variables and exogenous variables
  arma::mat X(t - p, k * p + m + 1,
              arma::fill::ones);  // Add m columns for the exogenous variables
                                  // and 1 column for the constant
  arma::mat Y(t - p, k, arma::fill::zeros);

  // Populate the matrices X and Y with lagged data and exogenous data
  for (int i = 0; i < (t - p); i++) {
    int index = 1;
    // Arrange predictors from smallest lag to biggest
    for (int lag = p - 1; lag >= 0; lag--) {
      X(i, arma::span(index, index + k - 1)) = data.row(i + lag);
      index += k;
    }
    // Append the exogenous variables to X
    X(i, arma::span(index, index + m - 1)) = exo_mat.row(i + p);
    Y.row(i) = data.row(i + p);
  }

  // Create a list to store Y and X
  Rcpp::List result;
  result["Y"] = Y;
  result["X"] = X;

  return result;
}
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
//' The [simAutoReg::YX()] function creates the `Y` and `X` matrices
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
//' The [simAutoReg::SimVAR()] function for simulating time series data
//' from a VAR model.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
Rcpp::List YX(const arma::mat& data, int p) {
  int t = data.n_rows;  // Number of observations
  int k = data.n_cols;  // Number of variables

  // Create matrices to store lagged variables and the dependent variable
  arma::mat X(t - p, k * p + 1, arma::fill::zeros);  // Add 1 column for the
                                                     // constant
  arma::mat Y(t - p, k, arma::fill::zeros);

  // Populate the matrices X and Y with lagged data
  for (int i = 0; i < (t - p); i++) {
    X(i, 0) = 1;  // Set the first column to 1 for the constant term
    int index = 1;
    // Arrange predictors from smallest lag to biggest
    for (int lag = p - 1; lag >= 0; lag--) {
      X.row(i).subvec(index, index + k - 1) = data.row(i + lag);
      index += k;
    }
    Y.row(i) = data.row(i + p);
  }

  // Create a list to store X, Y
  Rcpp::List result;
  result["X"] = X;
  result["Y"] = Y;

  return result;
}

// Dependencies
// simAutoReg-sim-var.cpp
