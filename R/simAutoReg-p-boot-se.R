#' Parametric Bootstrap Standard Errors
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param x Numeric matrix.
#'   Output of [simAutoReg::PBootVAROLS()].
#'
#' @return A matrix of standard error.
#'
#' @examples
#' pb <- PBootVAROLS(data = vark3p2, p = 2, B = 100)
#' PBootSE(pb)
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords simAutoReg pb
#' @export
PBootSE <- function(x) {
  dims <- dim(x$est)
  return(
    matrix(
      data = sqrt(diag(stats::var(x$boot))),
      nrow = dims[1],
      ncol = dims[2]
    )
  )
}
