#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/001-simAutoReg-sim-ar.cpp
// -----------------------------------------------------------------------------

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
//' @return Numeric vector containing the simulated time series data.
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
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::vec SimAR(int time, int burn_in, const double& constant,
                const arma::vec& coef, const double& sd) {
  // Order of the AR model
  int p = coef.size();
  int total_time = time + burn_in;

  // Vector to store the generated time series data
  arma::vec data(total_time);

  // Generate random noise from a normal distribution
  arma::vec noise(total_time);
  for (int i = 0; i < total_time; i++) {
    noise(i) = R::rnorm(0, sd);
  }

  // Generate the autoregressive time series with burn-in
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
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/002-simAutoReg-sim-var.cpp
// -----------------------------------------------------------------------------

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
//'   with dimensions `k` by `(time - burn_in)`,
//'   where `k` is the number of variables and
//'   time is the number of observations.
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
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int q = coef.n_cols;      // Dimension of the coefficient matrix
  int p = q / k;            // Order of the VAR model (number of lags)

  // Matrix to store the generated VAR time series data
  arma::mat data(k, time + burn_in);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < time + burn_in; t++) {
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

  // Transpose the data matrix and
  // return only the required time period after burn-in
  return data.cols(burn_in, time + burn_in - 1).t();
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/003-simAutoReg-sim-var-zip.cpp
// -----------------------------------------------------------------------------

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
//'   with dimensions `k` by `(time - burn_in)`,
//'   where `k` is the number of variables
//'   and time is the number of observations.
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
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVARZIP(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int q = coef.n_cols;      // Dimension of the coefficient matrix
  int p = q / k;            // Order of the VAR model (number of lags)

  // Matrix to store the generated VAR time series data
  arma::mat data(k, time + burn_in);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < time + burn_in; t++) {
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

  // Transpose the data matrix and
  // return only the required time period after burn-in
  return data.cols(burn_in, time + burn_in - 1).t();
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/004-simAutoReg-sim-mvn.cpp
// -----------------------------------------------------------------------------

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
//' @importFrom Rcpp sourceCpp
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
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/005-simAutoReg-sim-variance.cpp
// -----------------------------------------------------------------------------

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
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVariance(int n, const arma::vec& location,
                      const arma::mat& chol_scale) {
  int k = location.n_elem;  // Number of variables

  // Generate multivariate normal random vectors epsilon
  arma::mat epsilon = chol_scale * arma::randn(k, n);

  // Add the location vector to each column of epsilon
  epsilon.each_col() += location;

  // Compute the variance vector for each sample
  arma::mat variance = arma::exp(epsilon);

  return variance;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/006-simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

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
//' yx <- YX(data = VAR, p = 2)
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
//' @importFrom Rcpp sourceCpp
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
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/007-simAutoReg-std-mat.cpp
// -----------------------------------------------------------------------------

//' Standardize Matrix
//'
//' This function standardizes the given matrix by centering the columns
//' and scaling them to have unit variance.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param X Numeric matrix.
//'   The matrix to be standardized.
//'
//' @return Numeric matrix with standardized values.
//'
//' @examples
//' std <- StdMat(VAR)
//' colMeans(std)
//' var(std)
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat StdMat(const arma::mat& X) {
  int q = X.n_cols;  // Number of predictors
  int n = X.n_rows;  // Number of observations

  arma::mat X_std(n, q, arma::fill::zeros);  // Initialize the standardized
                                             // matrix

  // Calculate column means
  arma::vec col_means(q, arma::fill::zeros);
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      col_means(j) += X(i, j);
    }
    col_means(j) /= n;
  }

  // Calculate column standard deviations
  arma::vec col_stddevs(q, arma::fill::zeros);
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      col_stddevs(j) += std::pow(X(i, j) - col_means(j), 2);
    }
    col_stddevs(j) = std::sqrt(col_stddevs(j) / (n - 1));
  }

  // Standardize the matrix
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      X_std(i, j) = (X(i, j) - col_means(j)) / col_stddevs(j);
    }
  }

  return X_std;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/008-simAutoReg-orig-scale.cpp
// -----------------------------------------------------------------------------

//' Return Standardized Estimates to the Original Scale
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef_std Numeric matrix.
//'   Standardized estimates of the autoregression
//'   and cross regression coefficients.
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//'
//' @examples
//' coef_std <- FitVAROLS(Y = StdMat(VAR_YX$Y), X = StdMat(VAR_YX$X[, -1]))
//' OrigScale(coef_std = coef_std, Y = VAR_YX$Y, X = VAR_YX$X[, -1])
//' FitVAROLS(Y = VAR_YX$Y, X = VAR_YX$X[, -1])
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat OrigScale(const arma::mat& coef_std, const arma::mat& Y,
                    const arma::mat& X) {
  int k = coef_std.n_rows;  // Number of outcomes
  int q = coef_std.n_cols;  // Number of predictors

  arma::vec sd_Y(k);
  arma::vec sd_X(q);

  // Calculate standard deviations of Y and X columns
  for (int l = 0; l < k; l++) {
    sd_Y(l) = arma::as_scalar(arma::stddev(Y.col(l), 0, 0));
  }
  for (int j = 0; j < q; j++) {
    sd_X(j) = arma::as_scalar(arma::stddev(X.col(j), 0, 0));
  }

  arma::mat coef_orig(k, q);
  for (int l = 0; l < k; l++) {
    for (int j = 0; j < q; j++) {
      double orig_coeff = coef_std(l, j) * sd_Y(l) / sd_X(j);
      coef_orig(l, j) = orig_coeff;
    }
  }

  return coef_orig;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/009-simAutoReg-fit-var-ols.cpp
// -----------------------------------------------------------------------------

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
//'   Matrix of predictors (X).
//'
//' @return Matrix of estimated autoregressive
//' and cross-regression coefficients.
//'
//' @examples
//' FitVAROLS(Y = VAR_YX$Y, X = VAR_YX$X)
//'
//' @details
//' The [simAutoReg::FitVAROLS()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Ordinary Least Squares (OLS) method.
//' Given the input matrices `Y` and `X`,
//' where `Y` is the matrix of dependent variables,
//' and `X` is the matrix of predictors,
//' the function computes the autoregressive and cross-regression coefficients
//' of the VAR model.
//' Note that if the first column of `X` is a vector of ones,
//' the constant vector is also estimated.
//'
//' The steps involved in estimating the VAR model parameters
//' using OLS are as follows:
//'
//' - Compute the QR decomposition of the lagged predictor matrix `X`
//'   using the `qr` function from the Armadillo library.
//' - Extract the `Q` and `R` matrices from the QR decomposition.
//' - Solve the linear system `R * coef = Q.t() * Y`
//'   to estimate the VAR model coefficients `coef`.
//' - The function returns a matrix containing the estimated
//'   autoregressive and cross-regression coefficients of the VAR model.
//'
//' @seealso
//' The `qr` function from the Armadillo library for QR decomposition.
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVAROLS(const arma::mat& Y, const arma::mat& X) {
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr(Q, R, X);

  // Solve the linear system R * coef = Q.t() * Y
  arma::mat coef = arma::solve(R, Q.t() * Y);

  return coef.t();
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/010-simAutoReg-p-boot-var-ols-rep.cpp
// -----------------------------------------------------------------------------

// Generate Data and Fit Model
arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant,
                         const arma::mat& coef, const arma::mat& chol_cov) {
  // Indices
  int k = constant.n_elem;  // Number of variables
  int q = coef.n_cols;      // Dimension of the coefficient matrix
  int p = q / k;            // Order of the VAR model (number of lags)

  // Simulate data
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // OLS
  arma::mat pb_coef = FitVAROLS(Y, X);

  return arma::vectorise(pb_coef);
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/011-simAutoReg-p-boot-var-ols-sim.cpp
// -----------------------------------------------------------------------------

// Function to generate VAR time series data and fit VAR model B times
arma::mat PBootVAROLSSim(int B, int time, int burn_in,
                         const arma::vec& constant, const arma::mat& coef,
                         const arma::mat& chol_cov) {
  int num_coef = constant.n_elem + coef.n_elem;
  arma::mat result(B, num_coef, arma::fill::zeros);

  for (int i = 0; i < B; i++) {
    arma::vec coef_est =
        PBootVAROLSRep(time, burn_in, constant, coef, chol_cov);
    result.row(i) = arma::trans(coef_est);
  }

  return result;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/012-simAutoReg-p-boot-var-ols.cpp
// -----------------------------------------------------------------------------

//' Parametric Bootstrap for the Vector Autoregressive Model
//' Using Ordinary Least Squares
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results
//'   in the simulation step.
//'
//' @return List containing the estimates (`est`)
//' and bootstrap estimates (`boot`).
//'
//' @examples
//' pb <- PBootVAROLS(data = VAR, p = 2, B = 100)
//' str(pb)
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B = 1000,
                       int burn_in = 200) {
  // Indices
  int t = data.n_rows;  // Number of observations

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Set parameters
  arma::vec const_vec = coef.col(0);
  arma::mat coef_mat = coef.cols(1, coef.n_cols - 1);

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();

  // Calculate the covariance of residuals
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Result matrix
  arma::mat sim = PBootVAROLSSim(B, t, burn_in, const_vec, coef_mat, chol_cov);

  // Create a list to store the results
  Rcpp::List result;

  // Add coef as the first element
  result["est"] = coef;

  // Add sim as the second element
  result["boot"] = sim;

  // Return the list
  return result;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/013-simAutoReg-fit-var-lasso.cpp
// -----------------------------------------------------------------------------

//' Fit Vector Autoregressive (VAR) Model Parameters using Lasso Regularization
//'
//' This function estimates the parameters of a VAR model
//' using the Lasso regularization method with cyclical coordinate descent.
//' The Lasso method is used to estimate the autoregressive
//' and cross-regression coefficients with sparsity.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y_std Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param X_std Numeric matrix.
//'   Matrix of standardized predictors (X).
//' @param lambda Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm.
//'   Default is 10000.
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance. Default is 1e-6.
//'
//' @return Matrix of estimated autoregressive and
//' cross-regression coefficients.
//'
//' @examples
//' Y_std <- StdMat(VAR_YX$Y)
//' X_std <- StdMat(VAR_YX$X[, -1])
//' lambda <- 73.90722
//' FitVARLasso(Y_std = Y_std, X_std = X_std, lambda = lambda)
//'
//' @details
//' The [simAutoReg::FitVARLasso()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Lasso regularization method.
//' Given the input matrices `Y_std` and `X_std`,
//' where `Y_std` is the matrix of standardized dependent variables,
//' and `X_std` is the matrix of standardized predictors,
//' the function computes the autoregressive and cross-regression coefficients
//' of the VAR model with sparsity induced by the Lasso regularization.
//'
//' The steps involved in estimating the VAR model parameters
//' using Lasso are as follows:
//'
//' - **Initialization**: The function initializes the coefficient matrix
//'   `beta` with OLS estimates.
//'   The `beta` matrix will store the estimated autoregressive and
//'   cross-regression coefficients.
//' - **Coordinate Descent Loop**: The function performs
//'   the cyclical coordinate descent algorithm
//'   to estimate the coefficients iteratively.
//'   The loop iterates `max_iter` times (default is 10000),
//'   or until convergence is achieved.
//'   The outer loop iterates over the predictor variables
//'   (columns of `X_std`),
//'   while the inner loop iterates over the outcome variables
//'   (columns of `Y_std`).
//' - **Coefficient Update**: For each predictor variable (column of `X_std`),
//'   the function iteratively updates the corresponding column of `beta`
//'   using the coordinate descent algorithm with L1 norm regularization
//'   (Lasso).
//'   The update involves calculating the soft-thresholded value `c`,
//'   which encourages sparsity in the coefficients.
//'   The algorithm continues until the change in coefficients
//'   between iterations is below the specified tolerance `tol`
//'   or when the maximum number of iterations is reached.
//' - **Convergence Check**: The function checks for convergence
//'   by comparing the current `beta`
//'   matrix with the previous iteration's `beta_old`.
//'   If the maximum absolute difference between `beta` and `beta_old`
//'   is below the tolerance `tol`,
//'   the algorithm is considered converged, and the loop exits.
//'
//' @seealso
//' The [simAutoReg::FitVAROLS()] function for estimating VAR model parameters
//' using OLS.
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLasso(const arma::mat& Y_std, const arma::mat& X_std,
                      const double& lambda, int max_iter = 10000,
                      double tol = 1e-5) {
  int q =
      X_std.n_cols;  // Number of predictors (excluding the intercept column)
  int k = Y_std.n_cols;  // Number of outcomes

  // OLS starting values
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr(Q, R, X_std);
  // Solve the linear system R * beta = Q.t() * Y_std
  arma::mat beta = arma::solve(R, Q.t() * Y_std);

  // Coordinate Descent Loop
  for (int iter = 0; iter < max_iter; iter++) {
    arma::mat beta_old = beta;  // Initialize beta_old
                                // with the current value of beta

    // Create a copy of Y_std to use for updating Y_l
    arma::mat Y_copy = Y_std;

    // Update each coefficient for each predictor
    // using cyclical coordinate descent
    for (int j = 0; j < q; j++) {
      arma::vec Xj = X_std.col(j);
      for (int l = 0; l < k; l++) {
        arma::vec Y_l = Y_copy.col(l);
        double rho = dot(Xj, Y_l - X_std * beta.col(l) + beta(j, l) * Xj);
        double z = dot(Xj, Xj);
        double c = 0;

        if (rho < -lambda / 2) {
          c = (rho + lambda / 2) / z;
        } else if (rho > lambda / 2) {
          c = (rho - lambda / 2) / z;
        } else {
          c = 0;
        }
        beta(j, l) = c;

        // Update Y_l for the next iteration
        Y_l = Y_l - (Xj * (beta(j, l) - beta_old(j, l)));
      }
    }

    // Check convergence
    if (iter > 0) {
      if (arma::all(arma::vectorise(arma::abs(beta - beta_old)) < tol)) {
        break;  // Converged, exit the loop
      }
    }

    // If the loop reaches the last iteration and has not broken
    // (not converged),
    // emit a warning
    if (iter == max_iter - 1) {
      Rcpp::warning(
          "The algorithm did not converge within the specified maximum number "
          "of iterations.");
    }
  }

  return beta.t();
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/014-simAutoReg-fit-var-lasso-search.cpp
// -----------------------------------------------------------------------------

//' Fit Vector Autoregressive (VAR) Model Parameters using Lasso Regularization
//' with Lambda Search
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y_std Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param X_std Numeric matrix.
//'   Matrix of standardized predictors (X).
//' @param lambdas Numeric vector.
//'   Vector of lambda hyperparameters for Lasso regularization.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm.
//'   Default is 10000.
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance. Default is 1e-5.
//' @param crit Character string.
//'   Information criteria to use.
//'   Valid values include `"aic"`, `"bic"`, and `"ebic"`.
//'
//' @return Matrix of estimated autoregressive
//' and cross-regression coefficients.
//'
//' @examples
//' Y_std <- StdMat(VAR_YX$Y)
//' X_std <- StdMat(VAR_YX$X[, -1])
//' lambdas <- LambdaSeq(Y = Y_std, X = X_std, n_lambdas = 100)
//' FitVARLassoSearch(Y_std = Y_std, X_std = X_std, lambdas = lambdas)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLassoSearch(const arma::mat& Y_std, const arma::mat& X_std,
                            const arma::vec& lambdas,
                            const std::string& crit = "ebic",
                            int max_iter = 10000, double tol = 1e-5) {
  int n = X_std.n_rows;  // Number of observations (rows in X)
  int q = X_std.n_cols;  // Number of columns in X (predictors)

  // Variables to track the minimum criterion value
  double min_criterion = std::numeric_limits<double>::infinity();
  arma::mat beta_min_criterion;

  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Fit the VAR model using Lasso regularization
    arma::mat beta = FitVARLasso(Y_std, X_std, lambda, max_iter, tol);

    // Calculate the residuals
    arma::mat residuals = Y_std - X_std * beta.t();

    // Compute the residual sum of squares (RSS)
    double rss = arma::accu(residuals % residuals);

    // Compute the degrees of freedom for each parameter
    int num_params = arma::sum(arma::vectorise(beta != 0));

    // Compute the AIC, BIC, and EBIC criteria
    double aic = n * std::log(rss / n) + 2.0 * num_params;
    double bic = n * std::log(rss / n) + num_params * std::log(n);
    double ebic =
        n * std::log(rss / n) + 2.0 * num_params * std::log(n / double(q));

    // Update the minimum criterion and its index if necessary
    double current_criterion = 0.0;
    if (crit == "aic") {
      current_criterion = aic;
    } else if (crit == "bic") {
      current_criterion = bic;
    } else if (crit == "ebic") {
      current_criterion = ebic;
    }

    if (current_criterion < min_criterion) {
      min_criterion = current_criterion;
      beta_min_criterion = beta;
    }
  }

  return beta_min_criterion;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/015-simAutoReg-lambda-seq.cpp
// -----------------------------------------------------------------------------

//' Function to generate the sequence of lambdas
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//' @param n_lambdas Integer.
//'   Number of lambdas to generate.
//'
//' @return Returns a vector of lambdas.
//'
//' @examples
//' Y_std <- StdMat(VAR_YX$Y)
//' X_std <- StdMat(VAR_YX$X[, -1])
//' LambdaSeq(Y = Y_std, X = X_std, n_lambdas = 100)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg fit
//' @export
// [[Rcpp::export]]
arma::vec LambdaSeq(const arma::mat& Y, const arma::mat& X,
                    int n_lambdas = 100) {
  int k = Y.n_cols;  // Number of variables

  arma::mat XtX = trans(X) * X;
  double lambda_max = arma::max(diagvec(XtX)) / (k * 2);

  // Generate the sequence of lambdas
  double log_lambda_max = std::log10(lambda_max);
  arma::vec lambda_seq(n_lambdas);
  double log_lambda_step =
      (std::log10(lambda_max / 1000) - log_lambda_max) / (n_lambdas - 1);

  for (int i = 0; i < n_lambdas; ++i) {
    double log_lambda = log_lambda_max + i * log_lambda_step;
    lambda_seq(i) = std::pow(10, log_lambda);
  }

  return lambda_seq;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// edit .setup/cpp/016-simAutoReg-search-var-lasso.cpp
// -----------------------------------------------------------------------------

//' Compute AIC, BIC, and EBIC for Lasso Regularization
//'
//' This function computes the Akaike Information Criterion (AIC),
//' Bayesian Information Criterion (BIC),
//' and Extended Bayesian Information Criterion (EBIC)
//' for a given matrix of predictors `X`, a matrix of outcomes `Y`,
//' and a vector of lambda hyperparameters for Lasso regularization.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y_std Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param X_std Numeric matrix.
//'   Matrix of standardized predictors (X).
//' @param lambdas Numeric vector.
//'   Vector of lambda hyperparameters for Lasso regularization.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm.
//'   Default is 10000.
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance. Default is 1e-5.
//'
//' @return List containing two elements:
//'   - Element 1: Matrix with columns for
//'     lambda, AIC, BIC, and EBIC values.
//'   - Element 2: List of matrices containing
//'     the estimated autoregressive
//'     and cross-regression coefficients for each lambda.
//'
//' @examples
//' Y_std <- StdMat(VAR_YX$Y)
//' X_std <- StdMat(VAR_YX$X[, -1])
//' lambdas <- 10^seq(-5, 5, length.out = 100)
//' search <- SearchVARLasso(Y_std = Y_std, X_std = X_std, lambdas = lambdas)
//' plot(x = 1:nrow(search$criteria), y = search$criteria[, 4],
//'   type = "b", xlab = "lambda", ylab = "EBIC")
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg fit
//' @export
// [[Rcpp::export]]
Rcpp::List SearchVARLasso(const arma::mat& Y_std, const arma::mat& X_std,
                          const arma::vec& lambdas, int max_iter = 10000,
                          double tol = 1e-5) {
  int n = X_std.n_rows;  // Number of observations (rows in X)
  int q = X_std.n_cols;  // Number of columns in X (predictors)

  // Armadillo matrix to store the lambda, AIC, BIC, and EBIC values
  arma::mat results(lambdas.n_elem, 4, arma::fill::zeros);

  // List to store the output of FitVARLasso for each lambda
  Rcpp::List fit_list(lambdas.n_elem);

  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Fit the VAR model using Lasso regularization
    arma::mat beta = FitVARLasso(Y_std, X_std, lambda, max_iter, tol);

    // Calculate the residuals
    arma::mat residuals = Y_std - X_std * beta.t();

    // Compute the residual sum of squares (RSS)
    double rss = arma::accu(residuals % residuals);

    // Compute the degrees of freedom for each parameter
    int num_params = arma::sum(arma::vectorise(beta != 0));

    // Compute the AIC, BIC, and EBIC criteria
    double aic = n * std::log(rss / n) + 2.0 * num_params;
    double bic = n * std::log(rss / n) + num_params * std::log(n);
    double ebic =
        n * std::log(rss / n) + 2.0 * num_params * std::log(n / double(q));

    // Store the lambda, AIC, BIC, and EBIC values in the results matrix
    results(i, 0) = lambda;
    results(i, 1) = aic;
    results(i, 2) = bic;
    results(i, 3) = ebic;

    // Store the output of FitVARLasso for this lambda in the fit_list
    fit_list[i] = beta;
  }

  return Rcpp::List::create(Rcpp::Named("criteria") = results,
                            Rcpp::Named("fit") = fit_list);
}
