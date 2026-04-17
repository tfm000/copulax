# Generate Archimedean copula CDF/PDF reference values from R copula package.
#
# Run from project root:
#   Rscript copulax/tests_new/generate_r_reference.R
#
# Paste the output into the R_REFERENCE_DATA dict in test_archimedean_reference.py.
# This script is reproducibility documentation -- not invoked by pytest.
#
# Generated against R copula package v1.1.2.

library(copula)

u1 <- c(0.3, 0.5, 0.1, 0.9, 0.05, 0.95, 0.01, 0.5)
u2 <- c(0.7, 0.5, 0.9, 0.1, 0.05, 0.95, 0.5, 0.99)
u_mat <- cbind(u1, u2)

cat("# R copula package version:", as.character(packageVersion("copula")), "\n")
cat("# U_POINTS = np.column_stack([u1, u2]) where:\n")
cat("# u1 =", u1, "\n# u2 =", u2, "\n\n")

print_block <- function(name, theta, cdf_vals, pdf_vals) {
  cat(sprintf("    (\"%s\", %.1f): {\n", name, theta))
  cat("        \"cdf\": np.array([",
      paste(sprintf("%.16e", cdf_vals), collapse=", "), "]),\n")
  cat("        \"pdf\": np.array([",
      paste(sprintf("%.16e", pdf_vals), collapse=", "), "]),\n")
  cat("    },\n")
}

cat("R_REFERENCE_DATA = {\n")

for (theta in c(0.5, 2.0, 8.0)) {
  cop <- claytonCopula(theta, dim = 2)
  print_block("Clayton", theta, pCopula(u_mat, cop), dCopula(u_mat, cop))
}
for (theta in c(1.0, 5.0, 15.0)) {
  cop <- frankCopula(theta, dim = 2)
  print_block("Frank", theta, pCopula(u_mat, cop), dCopula(u_mat, cop))
}
for (theta in c(1.5, 3.0, 8.0)) {
  cop <- gumbelCopula(theta, dim = 2)
  print_block("Gumbel", theta, pCopula(u_mat, cop), dCopula(u_mat, cop))
}
for (theta in c(1.5, 3.0, 8.0)) {
  cop <- joeCopula(theta, dim = 2)
  print_block("Joe", theta, pCopula(u_mat, cop), dCopula(u_mat, cop))
}
for (theta in c(-0.5, 0.3, 0.9)) {
  cop <- amhCopula(theta, dim = 2)
  print_block("AMH", theta, pCopula(u_mat, cop), dCopula(u_mat, cop))
}

cat("}\n")
