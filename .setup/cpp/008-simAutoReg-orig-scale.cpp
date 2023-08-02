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
//' coef_std <- FitVAROLS(Y = StdMat(vark3p2yx$Y), X = StdMat(vark3p2yx$X[, -1]))
//' OrigScale(coef_std = coef_std, Y = vark3p2yx$Y, X = vark3p2yx$X[, -1])
//' FitVAROLS(Y = vark3p2yx$Y, X = vark3p2yx$X[, -1])
//'
//' @importFrom Rcpp sourceCpp
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat OrigScale(const arma::mat& coef_std,
                    const arma::mat& Y,
                    const arma::mat& X)
{
  int k = coef_std.n_rows; // Number of outcomes
  int q = coef_std.n_cols; // Number of predictors

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
