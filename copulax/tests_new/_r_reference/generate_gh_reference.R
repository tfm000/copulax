# Generate R ghyp reference data for multivariate GH and skewed-t tests.
#
# Run from project root:
#   Rscript copulax/tests_new/_r_reference/generate_gh_reference.R \
#     > copulax/tests_new/_r_reference/gh_reference_data.py
#
# The ghyp package uses the McNeil et al. (2005) (lambda, chi, psi, mu,
# sigma, gamma) parametrisation via its ghyp() constructor -- identical
# to CopulAX mvt_gh and (with lambda=-nu/2, chi=nu, psi=0) mvt_skewed_t.
#
# This script is the source of truth -- regenerate the .py file if
# parameter sets or test points change.

suppressPackageStartupMessages(library(ghyp))

# ----- Fixed test x-points (d=2 uses first 2 columns, d=3 uses all) ----
X_POINTS <- rbind(
  c( 0.0,  0.0,  0.0),
  c( 0.5, -1.0,  0.3),
  c(-1.0,  1.0,  2.0),
  c( 2.0,  0.5, -0.5),
  c(-2.0, -0.5,  1.0),
  c( 0.1,  0.2, -0.1),
  c( 1.5, -1.5,  0.5),
  c(-0.5,  1.5, -2.0)
)

# ----- mvt_gh parameter cases -----
GH_CASES <- list(
  list(
    name  = "gh_d3_symmetric",
    d     = 3,
    lamb  = 0.5,
    chi   = 1.5,
    psi   = 1.5,
    mu    = c(0.0, 0.0, 0.0),
    gamma = c(0.0, 0.0, 0.0),
    sigma = matrix(c(1.0, 0.5, 0.3,
                     0.5, 1.0, 0.4,
                     0.3, 0.4, 1.0), nrow = 3)
  ),
  list(
    name  = "gh_d3_skewed",
    d     = 3,
    lamb  = 0.5,
    chi   = 1.5,
    psi   = 1.5,
    mu    = c(0.1, -0.2, 0.3),
    gamma = c(0.3, -0.2, 0.1),
    sigma = matrix(c(1.0, 0.5, 0.3,
                     0.5, 1.0, 0.4,
                     0.3, 0.4, 1.0), nrow = 3)
  ),
  list(
    name  = "gh_d2_skewed",
    d     = 2,
    lamb  = -0.5,
    chi   = 1.0,
    psi   = 2.0,
    mu    = c(0.0, 0.0),
    gamma = c(0.5, -0.3),
    sigma = matrix(c(1.0, 0.6,
                     0.6, 1.0), nrow = 2)
  )
)

# ----- mvt_skewed_t parameter cases (lambda=-nu/2, chi=nu, psi=0) -----
SKEWT_CASES <- list(
  list(
    name  = "skewt_d3_symmetric",
    d     = 3,
    nu    = 5.0,
    mu    = c(0.0, 0.0, 0.0),
    gamma = c(0.0, 0.0, 0.0),
    sigma = matrix(c(1.0, 0.5, 0.3,
                     0.5, 1.0, 0.4,
                     0.3, 0.4, 1.0), nrow = 3)
  ),
  list(
    name  = "skewt_d3_skewed",
    d     = 3,
    nu    = 5.0,
    mu    = c(0.1, -0.2, 0.3),
    gamma = c(0.3, -0.2, 0.1),
    sigma = matrix(c(1.0, 0.5, 0.3,
                     0.5, 1.0, 0.4,
                     0.3, 0.4, 1.0), nrow = 3)
  ),
  list(
    name  = "skewt_d2_skewed",
    d     = 2,
    nu    = 4.0,
    mu    = c(0.0, 0.0),
    gamma = c(0.5, -0.3),
    sigma = matrix(c(1.0, 0.6,
                     0.6, 1.0), nrow = 2)
  )
)

# ----- helpers -----
fmt_vec <- function(v) paste(sprintf("%.16e", v), collapse = ", ")
fmt_mat <- function(m) {
  rows <- apply(m, 1, function(r) paste0("[", fmt_vec(r), "]"))
  paste0("[", paste(rows, collapse = ", "), "]")
}

write_case_header <- function(name, d) {
  cat(sprintf('    "%s": {\n', name))
  cat(sprintf('        "d": %d,\n', d))
}

write_case_params_gh <- function(case) {
  cat(sprintf('        "lamb": %.16e,\n', case$lamb))
  cat(sprintf('        "chi":  %.16e,\n', case$chi))
  cat(sprintf('        "psi":  %.16e,\n', case$psi))
  cat(sprintf('        "mu":    np.array([%s]),\n', fmt_vec(case$mu)))
  cat(sprintf('        "gamma": np.array([%s]),\n', fmt_vec(case$gamma)))
  cat(sprintf('        "sigma": np.array(%s),\n', fmt_mat(case$sigma)))
}

write_case_params_skewt <- function(case) {
  cat(sprintf('        "nu":   %.16e,\n', case$nu))
  cat(sprintf('        "mu":    np.array([%s]),\n', fmt_vec(case$mu)))
  cat(sprintf('        "gamma": np.array([%s]),\n', fmt_vec(case$gamma)))
  cat(sprintf('        "sigma": np.array(%s),\n', fmt_mat(case$sigma)))
}

write_values <- function(x_mat, logpdf_vals, pdf_vals) {
  cat('        "x":      np.array([\n')
  for (i in 1:nrow(x_mat)) {
    cat(sprintf('            [%s],\n', fmt_vec(x_mat[i, ])))
  }
  cat('        ]),\n')
  cat('        "logpdf": np.array([\n')
  for (v in logpdf_vals) cat(sprintf('            %.16e,\n', v))
  cat('        ]),\n')
  cat('        "pdf":    np.array([\n')
  for (v in pdf_vals) cat(sprintf('            %.16e,\n', v))
  cat('        ]),\n    },\n')
}

# ----- header -----
cat('"""Auto-generated R ghyp reference data for mvt_gh and mvt_skewed_t tests.\n')
cat('\n')
cat('Source: copulax/tests_new/_r_reference/generate_gh_reference.R\n')
cat(sprintf('R ghyp package: v%s\n', as.character(packageVersion("ghyp"))))
cat('\n')
cat('Parametrisation: ghyp(lambda, chi, psi, mu, sigma, gamma) is the McNeil\n')
cat('et al. (2005) form identical to CopulAX. For skewed-t, lambda=-nu/2,\n')
cat('chi=nu, psi=0.\n')
cat('\n')
cat('Do NOT edit by hand -- regenerate by running the R script.\n')
cat('"""\n\n')
cat('import numpy as np\n\n')

# ----- GH cases -----
cat('GH_CASES = {\n')
for (case in GH_CASES) {
  obj <- ghyp(lambda = case$lamb, chi = case$chi, psi = case$psi,
              mu = case$mu, sigma = case$sigma, gamma = case$gamma)
  x_mat <- X_POINTS[, 1:case$d, drop = FALSE]
  logpdf_vals <- dghyp(x_mat, object = obj, logvalue = TRUE)
  pdf_vals <- dghyp(x_mat, object = obj, logvalue = FALSE)

  write_case_header(case$name, case$d)
  write_case_params_gh(case)
  write_values(x_mat, logpdf_vals, pdf_vals)
}
cat('}\n\n')

# ----- skewed-t cases -----
cat('SKEWT_CASES = {\n')
for (case in SKEWT_CASES) {
  obj <- ghyp(lambda = -case$nu / 2, chi = case$nu, psi = 0,
              mu = case$mu, sigma = case$sigma, gamma = case$gamma)
  x_mat <- X_POINTS[, 1:case$d, drop = FALSE]
  logpdf_vals <- dghyp(x_mat, object = obj, logvalue = TRUE)
  pdf_vals <- dghyp(x_mat, object = obj, logvalue = FALSE)

  write_case_header(case$name, case$d)
  write_case_params_skewt(case)
  write_values(x_mat, logpdf_vals, pdf_vals)
}
cat('}\n')
