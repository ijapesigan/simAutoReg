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
