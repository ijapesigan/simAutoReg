.FitVAROLS <- function(Y, X) {
  QR <- qr(X)
  R <- qr.R(QR)
  coef <- backsolve(
    R,
    qr.qty(QR, Y)
  )
  return(t(coef))
}
