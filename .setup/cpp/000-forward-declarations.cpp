#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat SimMVN(int n, const arma::vec& location, const arma::mat& chol_scale);

arma::mat SimMVNFixed(int n, const arma::vec& location, const arma::mat& scale);

arma::vec SimAR(int time, const double& constant, const arma::vec& coef,
                const double& sd);

arma::mat SimVAR(int time, const arma::vec& constant, const arma::mat& coef,
                 const arma::mat& chol_cov);

arma::mat SimVARZIP(int time, const arma::vec& constant, const arma::mat& coef,
                    const arma::mat& chol_cov);

arma::mat SimVARExo(int time, const arma::vec& constant, const arma::mat& coef,
                    const arma::mat& chol_cov, const arma::mat& exo_mat,
                    const arma::mat& exo_coef);

arma::mat SimVARZIPExo(int time, const arma::vec& constant,
                       const arma::mat& coef, const arma::mat& chol_cov,
                       const arma::mat& exo_mat, const arma::mat& exo_coef);

arma::mat SimVariance(int n, const arma::vec& location,
                      const arma::mat& chol_scale);

Rcpp::List YX(const arma::mat& data, int p);

Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat);
