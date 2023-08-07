// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// SimARCoef
arma::vec SimARCoef(int p);
RcppExport SEXP _simAutoReg_SimARCoef(SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(SimARCoef(p));
    return rcpp_result_gen;
END_RCPP
}
// SimAR
arma::vec SimAR(int time, int burn_in, const double& constant, const arma::vec& coef, const double& sd);
RcppExport SEXP _simAutoReg_SimAR(SEXP timeSEXP, SEXP burn_inSEXP, SEXP constantSEXP, SEXP coefSEXP, SEXP sdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< const double& >::type constant(constantSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type coef(coefSEXP);
    Rcpp::traits::input_parameter< const double& >::type sd(sdSEXP);
    rcpp_result_gen = Rcpp::wrap(SimAR(time, burn_in, constant, coef, sd));
    return rcpp_result_gen;
END_RCPP
}
// SimMVN
arma::mat SimMVN(int n, const arma::vec& location, const arma::mat& chol_scale);
RcppExport SEXP _simAutoReg_SimMVN(SEXP nSEXP, SEXP locationSEXP, SEXP chol_scaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type location(locationSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type chol_scale(chol_scaleSEXP);
    rcpp_result_gen = Rcpp::wrap(SimMVN(n, location, chol_scale));
    return rcpp_result_gen;
END_RCPP
}
// SimPD
arma::mat SimPD(int p);
RcppExport SEXP _simAutoReg_SimPD(SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(SimPD(p));
    return rcpp_result_gen;
END_RCPP
}
// SimVARCoef
arma::mat SimVARCoef(int k, int p);
RcppExport SEXP _simAutoReg_SimVARCoef(SEXP kSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(SimVARCoef(k, p));
    return rcpp_result_gen;
END_RCPP
}
// SimVARExo
arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant, const arma::mat& coef, const arma::mat& chol_cov, const arma::mat& exo_mat, const arma::mat& exo_coef);
RcppExport SEXP _simAutoReg_SimVARExo(SEXP timeSEXP, SEXP burn_inSEXP, SEXP constantSEXP, SEXP coefSEXP, SEXP chol_covSEXP, SEXP exo_matSEXP, SEXP exo_coefSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type constant(constantSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coef(coefSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type chol_cov(chol_covSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type exo_mat(exo_matSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type exo_coef(exo_coefSEXP);
    rcpp_result_gen = Rcpp::wrap(SimVARExo(time, burn_in, constant, coef, chol_cov, exo_mat, exo_coef));
    return rcpp_result_gen;
END_RCPP
}
// SimVARZIPExo
arma::mat SimVARZIPExo(int time, int burn_in, const arma::vec& constant, const arma::mat& coef, const arma::mat& chol_cov, const arma::mat& exo_mat, const arma::mat& exo_coef);
RcppExport SEXP _simAutoReg_SimVARZIPExo(SEXP timeSEXP, SEXP burn_inSEXP, SEXP constantSEXP, SEXP coefSEXP, SEXP chol_covSEXP, SEXP exo_matSEXP, SEXP exo_coefSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type constant(constantSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coef(coefSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type chol_cov(chol_covSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type exo_mat(exo_matSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type exo_coef(exo_coefSEXP);
    rcpp_result_gen = Rcpp::wrap(SimVARZIPExo(time, burn_in, constant, coef, chol_cov, exo_mat, exo_coef));
    return rcpp_result_gen;
END_RCPP
}
// SimVARZIP
arma::mat SimVARZIP(int time, int burn_in, const arma::vec& constant, const arma::mat& coef, const arma::mat& chol_cov);
RcppExport SEXP _simAutoReg_SimVARZIP(SEXP timeSEXP, SEXP burn_inSEXP, SEXP constantSEXP, SEXP coefSEXP, SEXP chol_covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type constant(constantSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coef(coefSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type chol_cov(chol_covSEXP);
    rcpp_result_gen = Rcpp::wrap(SimVARZIP(time, burn_in, constant, coef, chol_cov));
    return rcpp_result_gen;
END_RCPP
}
// SimVAR
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant, const arma::mat& coef, const arma::mat& chol_cov);
RcppExport SEXP _simAutoReg_SimVAR(SEXP timeSEXP, SEXP burn_inSEXP, SEXP constantSEXP, SEXP coefSEXP, SEXP chol_covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type constant(constantSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coef(coefSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type chol_cov(chol_covSEXP);
    rcpp_result_gen = Rcpp::wrap(SimVAR(time, burn_in, constant, coef, chol_cov));
    return rcpp_result_gen;
END_RCPP
}
// SimVariance
arma::mat SimVariance(int n, const arma::vec& location, const arma::mat& chol_scale);
RcppExport SEXP _simAutoReg_SimVariance(SEXP nSEXP, SEXP locationSEXP, SEXP chol_scaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type location(locationSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type chol_scale(chol_scaleSEXP);
    rcpp_result_gen = Rcpp::wrap(SimVariance(n, location, chol_scale));
    return rcpp_result_gen;
END_RCPP
}
// YXExo
Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat);
RcppExport SEXP _simAutoReg_YXExo(SEXP dataSEXP, SEXP pSEXP, SEXP exo_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type exo_mat(exo_matSEXP);
    rcpp_result_gen = Rcpp::wrap(YXExo(data, p, exo_mat));
    return rcpp_result_gen;
END_RCPP
}
// YX
Rcpp::List YX(const arma::mat& data, int p);
RcppExport SEXP _simAutoReg_YX(SEXP dataSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(YX(data, p));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_simAutoReg_SimARCoef", (DL_FUNC) &_simAutoReg_SimARCoef, 1},
    {"_simAutoReg_SimAR", (DL_FUNC) &_simAutoReg_SimAR, 5},
    {"_simAutoReg_SimMVN", (DL_FUNC) &_simAutoReg_SimMVN, 3},
    {"_simAutoReg_SimPD", (DL_FUNC) &_simAutoReg_SimPD, 1},
    {"_simAutoReg_SimVARCoef", (DL_FUNC) &_simAutoReg_SimVARCoef, 2},
    {"_simAutoReg_SimVARExo", (DL_FUNC) &_simAutoReg_SimVARExo, 7},
    {"_simAutoReg_SimVARZIPExo", (DL_FUNC) &_simAutoReg_SimVARZIPExo, 7},
    {"_simAutoReg_SimVARZIP", (DL_FUNC) &_simAutoReg_SimVARZIP, 5},
    {"_simAutoReg_SimVAR", (DL_FUNC) &_simAutoReg_SimVAR, 5},
    {"_simAutoReg_SimVariance", (DL_FUNC) &_simAutoReg_SimVariance, 3},
    {"_simAutoReg_YXExo", (DL_FUNC) &_simAutoReg_YXExo, 3},
    {"_simAutoReg_YX", (DL_FUNC) &_simAutoReg_YX, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_simAutoReg(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
