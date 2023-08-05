#include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat SimMVNCpp(int n, const arma::vec& location,
                    const arma::mat& chol_scale);

arma::mat SimMVNFixedCpp(int n, const arma::vec& location,
                         const arma::mat& scale);

arma::vec SimARCpp(int time, const double& constant, const arma::vec& coef,
                   const double& sd);

arma::mat SimVARCpp(int time, const arma::vec& constant, const arma::mat& coef,
                    const arma::mat& chol_cov);

arma::mat SimVARZIPCpp(int time, const arma::vec& constant,
                       const arma::mat& coef, const arma::mat& chol_cov);

arma::mat SimVARExoCpp(int time, const arma::vec& constant,
                       const arma::mat& coef, const arma::mat& chol_cov,
                       const arma::mat& exo_mat, const arma::mat& exo_coef);

arma::mat SimVARZIPExoCpp(int time, const arma::vec& constant,
                          const arma::mat& coef, const arma::mat& chol_cov,
                          const arma::mat& exo_mat, const arma::mat& exo_coef);

arma::mat SimVarianceCpp(int n, const arma::vec& location,
                         const arma::mat& chol_scale);

Rcpp::List YXCpp(const arma::mat& data, int p);

Rcpp::List YXExoCpp(const arma::mat& data, int p, const arma::mat& exo_mat);
