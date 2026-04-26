# Generate R copula reference data for fitting and sampling tests.
#
# Run from project root:
#   Rscript copulax/tests/_r_reference/generate_fit_sample.R \
#     > copulax/tests/_r_reference/fit_sample_data.py
#
# The output is a Python module containing:
#   - FIT_DATA[name] = {theta_true, theta_hat_r, u}: data + R itau fit
#   - SAMPLE_STATS[name] = {theta, n, tau_emp, cdf_emp}: empirical stats
#     from a large R sample (used to validate copulax sampling).
#
# This script is the source of truth -- regenerate the .py file if
# parameters or test points change.

suppressPackageStartupMessages(library(copula))

# Mirror of U_POINTS in the test file
u1 <- c(0.30, 0.50, 0.10, 0.90, 0.05, 0.95, 0.01, 0.50)
u2 <- c(0.70, 0.50, 0.90, 0.10, 0.05, 0.95, 0.50, 0.99)
U_POINTS <- cbind(u1, u2)

CASES <- list(
  list(name = "Clayton", theta = 2.0, ctor = claytonCopula),
  list(name = "Frank",   theta = 5.0, ctor = frankCopula),
  list(name = "Gumbel",  theta = 3.0, ctor = gumbelCopula),
  list(name = "Joe",     theta = 3.0, ctor = joeCopula),
  list(name = "AMH",     theta = 0.5, ctor = amhCopula)
)

n_fit <- 100
n_sample <- 50000

# Empirical bivariate CDF at points u
emp_cdf <- function(samples, points) {
  apply(points, 1, function(p) {
    mean(samples[, 1] <= p[1] & samples[, 2] <= p[2])
  })
}

cat('"""Auto-generated R reference data for Archimedean fitting and sampling tests.\n')
cat('\n')
cat('Source: copulax/tests/_r_reference/generate_fit_sample.R\n')
cat(sprintf('R copula package: v%s\n', as.character(packageVersion("copula"))))
cat('\n')
cat('Do NOT edit by hand -- regenerate by running the R script.\n')
cat('"""\n\n')
cat('import numpy as np\n\n')

cat('FIT_DATA = {\n')
for (case in CASES) {
  set.seed(42)
  cop_true <- case$ctor(case$theta, dim = 2)
  u <- rCopula(n_fit, cop_true)
  # fitCopula prints progress to stderr/stdout; capture and discard
  fit <- suppressMessages(suppressWarnings(
    capture.output({
      .fit_result <- fitCopula(case$ctor(dim = 2), data = u, method = "itau")
    })
  ))
  theta_hat <- as.numeric(coef(.fit_result))

  cat(sprintf('    "%s": {\n', case$name))
  cat(sprintf('        "theta_true": %.1f,\n', case$theta))
  cat(sprintf('        "theta_hat_r": %.16e,\n', theta_hat))
  cat('        "u": np.array([\n')
  for (i in 1:n_fit) {
    cat(sprintf('            [%.16e, %.16e],\n', u[i, 1], u[i, 2]))
  }
  cat('        ]),\n')
  cat('    },\n')
}
cat('}\n\n')

cat('SAMPLE_STATS = {\n')
for (case in CASES) {
  set.seed(123)
  cop <- case$ctor(case$theta, dim = 2)
  s <- rCopula(n_sample, cop)
  tau_emp <- cor(s[, 1], s[, 2], method = "kendall")
  cdf_emp <- emp_cdf(s, U_POINTS)

  cat(sprintf('    "%s": {\n', case$name))
  cat(sprintf('        "theta": %.1f,\n', case$theta))
  cat(sprintf('        "n": %d,\n', n_sample))
  cat(sprintf('        "tau_emp_r": %.16e,\n', tau_emp))
  cat('        "cdf_emp_r": np.array([',
      paste(sprintf('%.16e', cdf_emp), collapse = ', '), ']),\n')
  cat('    },\n')
}
cat('}\n')
