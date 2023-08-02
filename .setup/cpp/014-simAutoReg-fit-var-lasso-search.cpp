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
