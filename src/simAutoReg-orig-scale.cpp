#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

//' Return Standardized Estimates to the Original Scale
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef_std Numeric matrix.
//'   Standardized estimates of the autoregression and cross regression coefficients.
//' @param sd_Y Numeric vector.
//'   Standard deviations of the `Y` matrix.
//' @param sd_X Numeric vector.
//'   Standard deviations of the `X` matrix.
//' @param mean_Y Numeric vector.
//'   Means of the `Y` matrix.
//' @param mean_X Numeric vector.
//'   Means of the `X` matrix.
//'
//' @examples
//' coef_std <- FitVAROLS(Y = StdMat(VAR_YX$Y), X = StdMat(VAR_YX$X[, -1]))
//' sd_Y <- sqrt(diag(var(VAR_YX$Y)))
//' sd_X <- sqrt(diag(var(VAR_YX$X[, -1])))
//' mean_Y <- colMeans(VAR_YX$Y)
//' mean_X <- colMeans(VAR_YX$X[, -1])
//' OrigScale(coef_std = coef_std, sd_Y = sd_Y, sd_X = sd_X, mean_Y = mean_Y, mean_X = mean_X)
//' FitVAROLS(Y = VAR_YX$Y, X = VAR_YX$X[, -1])
//'
//' @importFrom Rcpp sourceCpp
//'
//' @export
// [[Rcpp::export]]
arma::mat OrigScale(const arma::mat& coef_std,
                    const arma::vec& sd_Y,
                    const arma::vec& sd_X,
                    const arma::vec& mean_Y,
                    const arma::vec& mean_X) {
  int k = coef_std.n_rows; // Number of outcomes
  int q = coef_std.n_cols; // Number of predictors

  arma::mat coef_orig(k, q);
  for (int l = 0; l < k; l++) {
    for (int j = 0; j < q; j++) {
      double orig_coeff = coef_std(l, j) * sd_Y(l) / sd_X(j) + mean_Y(l) - (mean_X(j) * coef_std(l, j));
      coef_orig(l, j) = orig_coeff;
    }
  }

  return coef_orig;
}
