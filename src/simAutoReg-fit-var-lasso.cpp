#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

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
//' lambda <- 58.57
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
arma::mat FitVARLasso(const arma::mat& Y_std,
                      const arma::mat& X_std,
                      const double& lambda,
                      int max_iter = 10000,
                      double tol = 1e-5)
{
  int q = X_std.n_cols; // Number of predictors (excluding the intercept column)
  int k = Y_std.n_cols; // Number of outcomes

  // OLS starting values
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr(Q, R, X_std);
  // Solve the linear system R * beta = Q.t() * Y_std
  arma::mat beta = arma::solve(R, Q.t() * Y_std);

  // Coordinate Descent Loop
  for (int iter = 0; iter < max_iter; iter++)
  {
    arma::mat beta_old = beta; // Initialize beta_old
                               // with the current value of beta

    // Create a copy of Y_std to use for updating Y_l
    arma::mat Y_copy = Y_std;

    // Update each coefficient for each predictor
    // using cyclical coordinate descent
    for (int j = 0; j < q; j++)
    {
      arma::vec Xj = X_std.col(j);
      for (int l = 0; l < k; l++)
      {
        arma::vec Y_l = Y_copy.col(l);
        double rho = dot(Xj, Y_l - X_std * beta.col(l) + beta(j, l) * Xj);
        double z = dot(Xj, Xj);
        double c = 0;

        if (rho < -lambda / 2)
        {
          c = (rho + lambda / 2) / z;
        }
        else if (rho > lambda / 2)
        {
          c = (rho - lambda / 2) / z;
        }
        else
        {
          c = 0;
        }
        beta(j, l) = c;

        // Update Y_l for the next iteration
        Y_l = Y_l - (Xj * (beta(j, l) - beta_old(j, l)));
      }
    }

    // Check convergence
    if (iter > 0)
    {
      if (arma::all(arma::vectorise(arma::abs(beta - beta_old)) < tol))
      {
        break; // Converged, exit the loop
      }
    }

    // If the loop reaches the last iteration and has not broken
    // (not converged),
    // emit a warning
    if (iter == max_iter - 1)
    {
      Rcpp::warning(
        "The algorithm did not converge within the specified maximum number of iterations."
      );
    }
  }

  return beta.t();
}

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
Rcpp::List SearchVARLasso(const arma::mat& Y_std,
                          const arma::mat& X_std,
                          const arma::vec& lambdas,
                          int max_iter = 10000,
                          double tol = 1e-5) {
  int n = X_std.n_rows; // Number of observations (rows in X)
  int q = X_std.n_cols; // Number of columns in X (predictors)

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
    double ebic = n * std::log(rss / n) + 2.0 * num_params * std::log(n / double(q));

    // Store the lambda, AIC, BIC, and EBIC values in the results matrix
    results(i, 0) = lambda;
    results(i, 1) = aic;
    results(i, 2) = bic;
    results(i, 3) = ebic;

    // Store the output of FitVARLasso for this lambda in the fit_list
    fit_list[i] = beta;
  }

  return Rcpp::List::create(Rcpp::Named("criteria") = results, Rcpp::Named("fit") = fit_list);
}
