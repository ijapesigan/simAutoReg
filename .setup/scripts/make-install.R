#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
path <- as.character(args[1])
if (
  file.exists(
    file.path(
      path,
      "DESCRIPTION"
    )
  )
) {
  deps <- list.files(
    path = file.path(
      path,
      ".setup",
      "r-dependencies"
    ),
    pattern = ".*\\.R",
    full.names = TRUE, recursive = TRUE
  )
  if (length(deps) > 0) {
    file.copy(
      from = deps,
      to = file.path(path, "R")
    )
  }
  cpp <- list.files(
    path = file.path(path, "src"),
    pattern = "^.*\\.cpp"
  )
  if (length(cpp) > 0) {
    namespace <- file.path(
      path,
      "NAMESPACE"
    )
    if(!file.exists(namespace)) {
      file.create(namespace)
    }
    Rcpp::compileAttributes(pkgdir = path)
    roxygen2::roxygenize(
      package.dir = path,
      roclets = "rd"
    )
    unlink(namespace)
  }
  devtools::document(path)
  devtools::install(path, dependencies = FALSE)
}
warnings()
