// -----------------------------------------------------------------------------
// edit .setup/cpp/000-forward-declarations.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat SimPD(int p);

arma::mat SimMVN(int n, const arma::vec& location, const arma::mat& chol_scale);

bool CheckARCoef(const arma::vec& coef);

arma::vec SimARCoef(int p);

arma::vec SimAR(int time, int burn_in, const double& constant,
                const arma::vec& coef, const double& sd);

bool CheckVARCoef(const arma::mat& coef);

arma::mat SimVARCoef(int k, int p);

arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov);

arma::mat SimVARZIP(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov);

bool CheckVARExoCoef(const arma::mat& coef);

arma::mat SimVARExoCoef(int k, int p, int m);

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
// edit .setup/cpp/simAutoReg-check-ar-coef.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Check AR(p) Coefficients for Stationarity
//'
//' This function checks for stationarity of the AR(p) coefficients.
//' Stationarity is determined based on the roots
//' of the autoregressive polynomial.
//' For a stationary AR(p) process,
//' all the roots of this autoregressive polynomial
//' must lie inside the unit circle in the complex plane.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef Numeric vector.
//'   Autoregressive coefficients.
//'
//' @examples
//' set.seed(42)
//' (coef <- SimARCoef(p = 2))
//' CheckARCoef(coef = coef)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg check ar
//' @export
// [[Rcpp::export]]
bool CheckARCoef(const arma::vec& coef) {
  // Step 1: Compute the roots of the characteristic polynomial
  arma::cx_vec roots = arma::roots(arma::join_cols(arma::vec{1}, -coef));

  // Step 2: Check if all roots have magnitudes less than 1
  //         (stability condition)
  return arma::all(arma::abs(roots) < 1);
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-check-var-coef.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Check VAR(p) Coefficients for Stationarity
//'
//' The function checks if all the eigenvalues have moduli
//' (absolute values) less than 1.
//' If all eigenvalues have moduli less than 1,
//' it indicates that the VAR process is stable and, therefore, stationary.
//' If any eigenvalue has a modulus greater than or equal to 1,
//' it indicates that the VAR process is not stable and,
//' therefore, non-stationary.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef Numeric matrix.
//'   Coefficient matrix with dimensions `k` by `(k * p)`.
//'   Each `k` by `k` block corresponds to the coefficient matrix
//'   for a particular lag.
//'
//' @examples
//' set.seed(42)
//' (coef <- SimVARCoef(k = 3, p = 2))
//' CheckVARCoef(coef = coef)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg check var
//' @export
// [[Rcpp::export]]
bool CheckVARCoef(const arma::mat& coef) {
  // Step 1: Determine the number of outcome variables and lags
  // Number of outcome variables
  int num_outcome_vars = coef.n_rows;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;

  // Step 2: Create a companion matrix for the VAR coefficients
  arma::mat companion_matrix(num_outcome_vars * num_lags,
                             num_outcome_vars * num_lags, arma::fill::zeros);

  // Step 3: Fill the companion matrix using VAR coefficients
  for (int i = 0; i < num_lags; i++) {
    // Step 3.1: Fill the diagonal block of the companion matrix
    //           with VAR coefficients
    companion_matrix.submat(i * num_outcome_vars, i * num_outcome_vars,
                            (i + 1) * num_outcome_vars - 1,
                            (i + 1) * num_outcome_vars - 1) =
        coef.cols(i * num_outcome_vars, (i + 1) * num_outcome_vars - 1);

    // Step 3.2: Fill the sub-diagonal block with identity matrices (lags > 0)
    if (i > 0) {
      companion_matrix.submat(i * num_outcome_vars, (i - 1) * num_outcome_vars,
                              (i + 1) * num_outcome_vars - 1,
                              i * num_outcome_vars - 1) =
          arma::eye(num_outcome_vars, num_outcome_vars);
    }
  }

  // Step 4: Compute the eigenvalues of the companion matrix
  arma::cx_vec eigenvalues = arma::eig_gen(companion_matrix);

  // Step 5: Check if all eigenvalues have magnitudes less than 1
  //         (stability condition)
  return arma::all(arma::abs(eigenvalues) < 1);
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-ar-coef.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Random Autoregressive Coefficients for a Stationary AR(p) Model
//'
//' This function generates autoregressive coefficients
//' for a stationary AR(p) model.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param p Positive integer. Number of lags.
//'
//' @examples
//' set.seed(42)
//' SimARCoef(p = 2)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim coef ar
//' @export
// [[Rcpp::export]]
arma::vec SimARCoef(int p) {
  // Step 1: Initialize a vector to store the generated stable
  //         autoregressive coefficients
  arma::vec coefs(p);

  // Step 2: Enter an infinite loop for coefficient generation
  //         and stability check
  while (true) {
    // Step 2.1: Generate random coefficients between -0.9 and 0.9
    coefs = -0.9 + 1.8 * arma::randu<arma::vec>(p);

    // Step 2.2: Compute the roots of the characteristic polynomial
    //           of the autoregressive model
    arma::cx_vec roots = arma::roots(arma::join_cols(arma::vec{1}, -coefs));

    // Step 2.3: Check if all roots have magnitudes less than 1
    //           (stability condition)
    if (arma::all(arma::abs(roots) < 1)) {
      // Step 2.4: If the coefficients lead to a stable autoregressive model,
      //           exit the loop
      break;
    }
  }

  // Step 3: Return the generated stable autoregressive coefficients
  return coefs;
}
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
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-mvn.cpp
// Ivan Jacob Agaloos Pesigan
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
//' The [SimMVN()] function generates
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
//' @keywords simAutoReg sim data mvn
//' @export
// [[Rcpp::export]]
arma::mat SimMVN(int n, const arma::vec& location,
                 const arma::mat& chol_scale) {
  // Step 1: Determine the number of variables
  int num_variables = location.n_elem;

  // Step 2: Generate a matrix of random standard normal variates
  arma::mat data = arma::randn(n, num_variables);

  // Step 3: Transform the random values to follow
  //         a multivariate normal distribution
  //         by scaling with the Cholesky decomposition
  //         and adding the location vector
  data = data * chol_scale + arma::repmat(location.t(), n, 1);

  // Step 4: Return the simulated multivariate normal data
  return data;
}

// Dependencies
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-pd.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Random Symmetric Positive Definite Matrix
//'
//' This function generates a random positive definite matrix.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param p Positive integer.
//'   Dimension of the `p` by `p` matrix.
//'
//' @examples
//' set.seed(42)
//' SimPD(p = 3)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim cov
//' @export
// [[Rcpp::export]]
arma::mat SimPD(int p) {
  // Step 1: Generate a p x p matrix filled with random values
  arma::mat data(p, p, arma::fill::randn);

  // Step 2: Make the matrix symmetric by multiplying it with its transpose
  data = data * data.t();

  // Step 3: Add a small positive diagonal to ensure positive definiteness
  data += 0.001 * arma::eye<arma::mat>(p, p);

  // Step 4: Return the positive definite matrix
  return data;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var-coef.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Random Vector Autoregressive Coefficients
//' for a Stationary VAR(p) Model
//'
//' This function generates stationary VAR(P) coefficients.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param k Positive integer.
//'   Number of autoregressive variables.
//' @param p Positive integer.
//'   Number of lags.
//'
//' @examples
//' set.seed(42)
//' SimVARCoef(k = 3, p = 2)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim coef var
//' @export
// [[Rcpp::export]]
arma::mat SimVARCoef(int k, int p) {
  // Step 1: Create a matrix to store VAR coefficients
  arma::mat coefs(k, k * p);

  // Step 2: Generate random coefficients between -0.9 and 0.9
  while (true) {
    // Generate random values in [0, 1]
    coefs.randu();
    // Scale and shift values to [-0.9, 0.9]
    coefs = -0.9 + 1.8 * coefs;

    // Step 3: Create a companion matrix from VAR coefficients
    arma::mat companion_matrix(k * p, k * p, arma::fill::zeros);

    // Step 4: Fill the companion matrix using VAR coefficients
    for (int i = 0; i < p; i++) {
      // Fill the diagonal block of the companion matrix with VAR coefficients
      companion_matrix.submat(i * k, i * k, (i + 1) * k - 1, (i + 1) * k - 1) =
          coefs.cols(i * k, (i + 1) * k - 1);

      // Fill the sub-diagonal block of the companion matrix
      // with identity matrices
      if (i > 0) {
        companion_matrix.submat(i * k, (i - 1) * k, (i + 1) * k - 1,
                                i * k - 1) = arma::eye(k, k);
      }
    }

    // Step 5: Compute the eigenvalues of the companion matrix
    arma::cx_vec eigenvalues = arma::eig_gen(companion_matrix);

    // Step 6: Check if all eigenvalues are inside the unit circle
    if (arma::all(arma::abs(eigenvalues) < 1)) {
      // Exit the loop if all eigenvalues satisfy the condition
      break;
    }
  }

  // Step 7: Return the generated VAR coefficients
  return coefs;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var-exo.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive (VAR) Model with Exogenous
//' Variables
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
//' @examples
//' set.seed(42)
//' time <- 1000L
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
//' chol_cov <- chol(diag(3))
//' exo_mat <- MASS::mvrnorm(
//'   n = time + burn_in,
//'   mu = c(0, 0, 0),
//'   Sigma = diag(3)
//' )
//' exo_coef <- matrix(
//'   data = c(
//'     0.5, 0.0, 0.0,
//'     0.0, 0.5, 0.0,
//'     0.0, 0.0, 0.5
//'   ),
//'   nrow = 3
//' )
//' y <- SimVARExo(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov,
//'   exo_mat = exo_mat,
//'   exo_coef = exo_coef
//' )
//' str(y)
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov,
                    const arma::mat& exo_mat, const arma::mat& exo_coef) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values
  //         for each outcome variable
  data.each_col() = constant;

  // Step 4: Transpose the exogenous matrix for efficient column access
  arma::mat exo_mat_t = exo_mat.t();

  // Step 5: Simulate VAR-Exo data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 5.1: Generate random noise vector
    arma::vec noise = arma::randn(num_outcome_vars);

    // Step 5.2: Multiply the noise vector by the Cholesky decomposition
    //           of the covariance matrix
    arma::vec mult_noise = chol_cov * noise;

    // Step 5.3: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 5.4: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 5.5: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 5.6: Iterate over exogenous variables
      for (arma::uword x = 0; x < exo_mat_t.n_rows; x++) {
        // Update data with exogenous variables and their coefficients
        data(j, t) += exo_mat_t(x, t) * exo_coef(j, x);
      }

      // Step 5.7: Add the corresponding element from the noise vector
      data(j, t) += mult_noise(j);
    }
  }

  // Step 6: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 7: Return the transposed data matrix
  return data.t();
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var-zip-exo.cpp
// Ivan Jacob Agaloos Pesigan
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
//' @examples
//' set.seed(42)
//' time <- 1000L
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
//' chol_cov <- chol(diag(3))
//' exo_mat <- MASS::mvrnorm(
//'   n = time + burn_in,
//'   mu = c(0, 0, 0),
//'   Sigma = diag(3)
//' )
//' exo_coef <- matrix(
//'   data = c(
//'     0.5, 0.0, 0.0,
//'     0.0, 0.5, 0.0,
//'     0.0, 0.0, 0.5
//'   ),
//'   nrow = 3
//' )
//' y <- SimVARZIPExo(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov,
//'   exo_mat = exo_mat,
//'   exo_coef = exo_coef
//' )
//' str(y)
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVARZIPExo(int time, int burn_in, const arma::vec& constant,
                       const arma::mat& coef, const arma::mat& chol_cov,
                       const arma::mat& exo_mat, const arma::mat& exo_coef) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values
  //         for each outcome variable
  data.each_col() = constant;

  // Step 4: Transpose the exogenous matrix for efficient column access
  arma::mat exo_mat_t = exo_mat.t();

  // Step 5: Simulate VAR-ZIP-Exo data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 5.1: Generate random noise vector
    arma::vec noise = arma::randn(num_outcome_vars);

    // Step 5.2: Multiply the noise vector by the Cholesky decomposition
    //           of the covariance matrix
    arma::vec mult_noise = chol_cov * noise;

    // Step 5.3: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 5.4: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 5.5: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 5.6: Iterate over exogenous variables
      for (arma::uword x = 0; x < exo_mat_t.n_rows; x++) {
        // Update data with exogenous variables and their coefficients
        data(j, t) += exo_mat_t(x, t) * exo_coef(j, x);
      }

      // Step 5.7: Add the corresponding element from the noise vector
      data(j, t) += mult_noise(j);

      // Step 5.8: Calculate the intensity for the zero-inflated Poisson
      // distribution
      double intensity = std::exp(data(0, t));

      // Step 5.9: Simulate a zero-inflated Poisson random variable
      if (R::runif(0, 1) < intensity / (1 + intensity)) {
        // Set to zero with probability 1 - intensity
        data(0, t) = 0;
      } else {
        // Simulate Poisson count with intensity
        data(0, t) = R::rpois(intensity);
      }
    }
  }

  // Step 6: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 7: Return the transposed data matrix
  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var-zip.cpp
// Ivan Jacob Agaloos Pesigan
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
//' The [SimVARZIP()] function generates synthetic time series data
//' from a Vector Autoregressive (VAR)
//' with Zero-Inflated Poisson (ZIP) model for the first observed variable.
//' See [SimVAR()] for more details on generating data for VAR(p).
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
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVARZIP(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values
  //         for each outcome variable
  data.each_col() = constant;

  // Step 4: Simulate VAR-ZIP data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 4.1: Generate random noise vector
    arma::vec noise = arma::randn(num_outcome_vars);

    // Step 4.2: Multiply the noise vector
    //           by the Cholesky decomposition of the covariance matrix
    arma::vec mult_noise = chol_cov * noise;

    // Step 4.3: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 4.4: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 4.5: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 4.6: Add the corresponding element from the noise vector
      data(j, t) += mult_noise(j);

      // Step 4.7: Calculate the intensity
      //           for the zero-inflated Poisson distribution
      double intensity = std::exp(data(0, t));

      // Step 4.8: Simulate a zero-inflated Poisson random variable
      if (R::runif(0, 1) < intensity / (1 + intensity)) {
        // Set to zero with probability 1 - intensity
        data(0, t) = 0;
      } else {
        // Simulate Poisson count with intensity
        data(0, t) = R::rpois(intensity);
      }
    }
  }

  // Step 5: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 6: Return the transposed data matrix
  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var.cpp
// Ivan Jacob Agaloos Pesigan
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
//' The [SimVAR()] function generates synthetic time series data
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
//'     \deqn{
//'       Y_{tj} = \mathrm{constant}_j +
//'       \sum_{l = 1}^{p} \sum_{m = 1}^{k} (\mathrm{coef}_{jm} * Y_{im}) +
//'       \mathrm{noise}_{j}
//'     }
//'     where \eqn{Y_{tj}} is the value of variable `j` at time `t`,
//'     \eqn{\mathrm{constant}_j} is the constant term for variable `j`,
//'     \eqn{\mathrm{coef}_{jm}} are the coefficients for variable `j`
//'     from lagged variables up to order `p`,
//'     \eqn{Y_{tm}} are the lagged values of variable `m`
//'     up to order `p` at time `t`,
//'     and \eqn{\mathrm{noise}_{j}} is the element `j`
//'     from the generated vector of random process noise.
//' - Transpose the matrix `data` and return only
//'   the required time period after the burn-in period,
//'   which is from column `burn_in` to column `time + burn_in - 1`.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values
  //         for each outcome variable
  data.each_col() = constant;

  // Step 4: Simulate VAR data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 4.1: Generate random noise vector
    arma::vec noise = arma::randn(num_outcome_vars);

    // Step 4.2: Multiply the noise vector
    //           by the Cholesky decomposition of the covariance matrix
    arma::vec mult_noise = chol_cov * noise;

    // Step 4.3: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 4.4: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 4.5: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 4.6: Add the corresponding element from the noise vector
      data(j, t) += mult_noise(j);
    }
  }

  // Step 5: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 6: Return the transposed data matrix
  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-variance.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Generate Random Data for the Variance Vector
//'
//' This function generates random data for the variance vector given by
//' \deqn{
//'   \boldsymbol{\sigma}^{2} =
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
//' The [SimVariance()] function generates random data
//' for the variance vector
//' based on the exponential of a multivariate normal distribution.
//' Given the number of samples `n`,
//' the constant term \eqn{\boldsymbol{\mu}} represented
//' by the `location` vector,
//' and the Cholesky decomposition matrix \eqn{\boldsymbol{\Sigma}}
//' for the multivariate normal random error \eqn{\boldsymbol{\varepsilon}},
//' the function simulates \eqn{n} independent samples
//' of the variance vector \eqn{\boldsymbol{\sigma}^{2}}.
//' Each sample of the variance vector \eqn{\boldsymbol{\sigma}^{2}}
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
//' n <- 10L
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
//' @keywords simAutoReg sim variance
//' @export
// [[Rcpp::export]]
arma::mat SimVariance(int n, const arma::vec& location,
                      const arma::mat& chol_scale) {
  // Step 1: Simulate multivariate normal data
  arma::mat data = SimMVN(n, location, chol_scale);

  // Step 2: Transform the simulated data
  //         by taking the exponential of each element
  data = arma::exp(data);

  // Step 3: Return the transformed data
  return data;
}

// Dependencies
// simAutoReg-sim-mvn.cpp
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-y-x.cpp
// Ivan Jacob Agaloos Pesigan
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
//' @examples
//' set.seed(42)
//' time <- 1000L
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
//' chol_cov <- chol(diag(3))
//' exo_mat <- MASS::mvrnorm(
//'   n = time + burn_in,
//'   mu = c(0, 0, 0),
//'   Sigma = diag(3)
//' )
//' exo_coef <- matrix(
//'   data = c(
//'     0.5, 0.0, 0.0,
//'     0.0, 0.5, 0.0,
//'     0.0, 0.0, 0.5
//'   ),
//'   nrow = 3
//' )
//' y <- SimVARExo(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov,
//'   exo_mat = exo_mat,
//'   exo_coef = exo_coef
//' )
//' yx <- YXExo(
//'   data = y,
//'   p = 2,
//'   exo_mat = exo_mat[(burn_in + 1):(time + burn_in), ]
//' )
//' str(yx)
//'
//' @details
//' The [YX()] function creates the `Y` and `X` matrices
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
//' The [SimVAR()] function for simulating time series data
//' from a VAR model.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat) {
  // Step 1: Calculate the dimensions of the 'data' and 'exo_mat' matrices
  // Number of time steps (rows) in 'data'
  int time = data.n_rows;
  // Number of outcome variables (columns) in 'data'
  int num_outcome_vars = data.n_cols;
  // Number of exogenous variables (columns) in 'exo_mat'
  int num_exo_vars = exo_mat.n_cols;

  // Step 2: Create matrices 'X' and 'Y'
  //         to store transformed data 'X' matrix with ones
  arma::mat X(time - p, num_outcome_vars * p + num_exo_vars + 1,
              arma::fill::ones);
  // 'Y' matrix with zeros
  arma::mat Y(time - p, num_outcome_vars, arma::fill::zeros);

  // Step 3: Loop through the data and populate 'X' and 'Y'
  for (int time_index = 0; time_index < (time - p); time_index++) {
    // Initialize the column index for 'X'
    int index = 1;

    // Nested loop to populate 'X' with lagged values
    for (int lag = p - 1; lag >= 0; lag--) {
      // Update 'X' by assigning a subvector of 'data' to a subvector of 'X'
      X.row(time_index).subvec(index, index + num_outcome_vars - 1) =
          data.row(time_index + lag);
      // Move to the next set of columns in 'X'
      index += num_outcome_vars;
    }

    // Update 'X' with the exogenous variables
    X.row(time_index).subvec(index, index + num_exo_vars - 1) =
        exo_mat.row(time_index + p);

    // Update 'Y' with the target values
    Y.row(time_index) = data.row(time_index + p);
  }

  // Step 4: Create an Rcpp List 'result' and assign 'Y' and 'X' matrices to it
  Rcpp::List result;
  result["Y"] = Y;
  result["X"] = X;

  // Step 5: Return the 'result' List containing the transformed data
  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-y-x.cpp
// Ivan Jacob Agaloos Pesigan
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
//' The [YX()] function creates the `Y` and `X` matrices
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
//' The [SimVAR()] function for simulating time series data
//' from a VAR model.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
Rcpp::List YX(const arma::mat& data, int p) {
  // Step 1: Calculate the dimensions of the 'data' matrix
  // Number of time steps (rows)
  int time = data.n_rows;
  // Number of outcome variables (columns)
  int num_outcome_vars = data.n_cols;

  // Step 2: Create matrices 'X' and 'Y'
  //         to store transformed data 'X' matrix with ones
  arma::mat X(time - p, num_outcome_vars * p + 1, arma::fill::ones);
  // 'Y' matrix with zeros
  arma::mat Y(time - p, num_outcome_vars, arma::fill::zeros);

  // Step 3: Loop through the data and populate 'X' and 'Y'
  for (int time_index = 0; time_index < (time - p); time_index++) {
    // Initialize the column index for 'X'
    int index = 1;

    // Nested loop to populate 'X' with lagged values
    for (int lag = p - 1; lag >= 0; lag--) {
      // Update 'X' by assigning a subvector of 'data' to a subvector of 'X'
      X.row(time_index).subvec(index, index + num_outcome_vars - 1) =
          data.row(time_index + lag);
      // Move to the next set of columns in 'X'
      index += num_outcome_vars;
    }

    // Update 'Y' with the target values
    Y.row(time_index) = data.row(time_index + p);
  }

  // Step 4: Create an Rcpp List 'result' and assign 'X' and 'Y' matrices to it
  Rcpp::List result;
  result["X"] = X;
  result["Y"] = Y;

  // Step 5: Return the 'result' List containing the transformed data
  return result;
}

// Dependencies
// simAutoReg-sim-var.cpp
