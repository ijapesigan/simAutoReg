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
