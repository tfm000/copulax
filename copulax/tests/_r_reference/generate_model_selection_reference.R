# Generate a COMMON-SERIES rugarch model-selection reference for the
# AIC/BIC ranking tests in
# copulax/tests/test_timeseries_arma_garch.py::TestModelSelectionConsistency.
#
# Run from project root:
#   Rscript copulax/tests/_r_reference/generate_model_selection_reference.R \
#     > copulax/tests/_r_reference/model_selection_reference_data.py
#
# Why this reference exists (J2 / HARD-05):
# ----------------------------------------------------------------------
# The general ARMA-GARCH reference (generate_arma_garch_reference.R)
# simulates a SEPARATE series for each variant label and fits rugarch on
# that variant's OWN series, so its committed AIC/BIC belong to four
# different data realizations (var(y): garch ~1.65, gjr ~1.80,
# egarch ~2.13, igarch ~18.14). A model-selection RANKING assertion needs
# all candidates evaluated on ONE shared series; comparing copulax's
# same-series ranking against those per-variant-own-series numbers is a
# scale artifact (the "igarch last" agreement comes from the igarch
# series' 11x larger variance, not from a like-for-like comparison).
#
# This script therefore:
#   1. Simulates ONE arma11_garch11_normal-style series (fixed seed).
#   2. Fits rugarch sGARCH(1,1), iGARCH(1,1), gjrGARCH(1,1), eGARCH(1,1)
#      -- ARMA(1,1) mean, normal residuals -- ALL on that SAME series.
#   3. Captures each fit's log-likelihood, AIC, and BIC.
#
# rugarch::infocriteria returns *per-observation* values; this script
# rescales to absolute (AIC_total = aic_per_obs * N), matching
# generate_arma_garch_reference.R so the numbers are directly comparable
# to copulax's absolute AIC/BIC.
#
# The simulation truth spec is the same arma11_garch11_normal spec and
# seed used by generate_arma_garch_reference.R (mu=0.05, ar1=0.5, ma1=0.3,
# omega=0.05, alpha1=0.10, beta1=0.85; seed=11), so the shared series is
# the familiar GARCH-Normal reference series.
#
# Variant map (copulax <-> rugarch):
#   GARCH     <-> sGARCH
#   IGARCH    <-> iGARCH
#   GJR_GARCH <-> gjrGARCH
#   EGARCH    <-> eGARCH

suppressPackageStartupMessages(library(rugarch))

options(digits = 17)

# ---------------------------------------------------------------------
# Python-repr helpers (same contract as generate_arma_garch_reference.R)
# ---------------------------------------------------------------------

py_repr_scalar <- function(x) {
  if (is.logical(x)) return(if (x) "True" else "False")
  if (is.na(x) || is.nan(x)) return("float('nan')")
  if (is.infinite(x)) return(if (x > 0) "float('inf')" else "float('-inf')")
  sprintf("%.17g", x)
}

py_repr_array <- function(x) {
  if (length(x) == 0) return("np.array([], dtype=float)")
  body <- paste(sapply(x, py_repr_scalar), collapse = ", ")
  sprintf("np.array([%s], dtype=float)", body)
}

cx_var_model_name <- function(rugarch_name) {
  switch(rugarch_name,
    "sGARCH"   = "GARCH",
    "iGARCH"   = "IGARCH",
    "gjrGARCH" = "GJR_GARCH",
    "eGARCH"   = "EGARCH",
    stop("cx_var_model_name: unsupported ", rugarch_name)
  )
}

# ---------------------------------------------------------------------
# 1) Simulate ONE shared arma11_garch11_normal-style series.
# ---------------------------------------------------------------------

MEAN_ORDER <- c(1, 1)
VAR_ORDER  <- c(1, 1)
RESIDUAL   <- "norm"
SIM_SEED   <- 11
N_SIM      <- 2000

spec_truth <- ugarchspec(
  mean.model = list(armaOrder = MEAN_ORDER, include.mean = TRUE),
  variance.model = list(model = "sGARCH", garchOrder = VAR_ORDER),
  distribution.model = RESIDUAL,
  fixed.pars = list(mu = 0.05, ar1 = 0.5, ma1 = 0.3,
                    omega = 0.05, alpha1 = 0.10, beta1 = 0.85)
)
set.seed(SIM_SEED)
sim <- ugarchpath(spec_truth, n.sim = N_SIM, m.sim = 1)
y <- as.numeric(fitted(sim))
N <- length(y)

# ---------------------------------------------------------------------
# 2) Fit each variant on that SAME series; capture LL / AIC / BIC.
# ---------------------------------------------------------------------

fit_variant <- function(rugarch_model) {
  spec <- ugarchspec(
    mean.model = list(armaOrder = MEAN_ORDER, include.mean = TRUE),
    variance.model = list(model = rugarch_model, garchOrder = VAR_ORDER),
    distribution.model = RESIDUAL
  )
  fit <- ugarchfit(spec = spec, data = y, solver = "hybrid")
  ic <- infocriteria(fit)
  list(
    var_model = cx_var_model_name(rugarch_model),
    loglikelihood = as.numeric(fit@fit$LLH),
    aic = as.numeric(ic[1, 1]) * N,   # per-obs -> absolute
    bic = as.numeric(ic[2, 1]) * N,
    n_coef = length(coef(fit))
  )
}

# Ordering here is documentation only; the test ranks by the IC value.
VARIANTS <- c("sGARCH", "iGARCH", "gjrGARCH", "eGARCH")
LABELS   <- c("garch", "igarch", "gjr", "egarch")

results <- lapply(VARIANTS, fit_variant)
names(results) <- LABELS

# ---------------------------------------------------------------------
# 3) Emit the Python reference module to stdout.
# ---------------------------------------------------------------------

cat("\"\"\"Auto-generated COMMON-SERIES rugarch model-selection reference.\n\n")
cat("rugarch fits sGARCH / iGARCH / gjrGARCH / eGARCH -- ARMA(1,1) mean,\n")
cat("normal residuals -- ALL on ONE shared arma11_garch11_normal-style\n")
cat("series (seed ", SIM_SEED, ", n=", N, "), so the AIC/BIC ranking is a\n", sep = "")
cat("common-series ranking (not the cross-dataset per-variant-own-series\n")
cat("numbers in arma_garch_reference_data.py). Consumed by\n")
cat("test_timeseries_arma_garch.py::TestModelSelectionConsistency.\n\n")
cat("Regenerate with:\n")
cat("    Rscript copulax/tests/_r_reference/generate_model_selection_reference.R \\\n")
cat("        > copulax/tests/_r_reference/model_selection_reference_data.py\n\n")
cat("infocriteria() is per-observation; AIC/BIC below are absolute\n")
cat("(per-obs * N), matching copulax's absolute AIC/BIC and\n")
cat("generate_arma_garch_reference.R.\n\n")
cat("rugarch ", as.character(packageVersion("rugarch")),
    " on R ", paste(R.Version()$major, R.Version()$minor, sep = "."),
    ".\n", sep = "")
cat("\"\"\"\n\n")
cat("import numpy as np\n\n")

cat(sprintf("MODEL_SELECTION_SEED = %d\n", SIM_SEED))
cat(sprintf("MODEL_SELECTION_N = %d\n\n", N))

# The shared series, so the test fits copulax on the identical y.
cat(sprintf("MODEL_SELECTION_Y = %s\n\n", py_repr_array(y)))

cat("MODEL_SELECTION_REFERENCE = {\n")
for (lbl in LABELS) {
  r <- results[[lbl]]
  cat(sprintf("    \"%s\": {\n", lbl))
  cat(sprintf("        \"var_model\":     \"%s\",\n", r$var_model))
  cat(sprintf("        \"mean_order\":    (1, 1),\n"))
  cat(sprintf("        \"var_order\":     (1, 1),\n"))
  cat(sprintf("        \"residual_dist\": \"normal\",\n"))
  cat(sprintf("        \"n_coef\":        %d,\n", r$n_coef))
  cat(sprintf("        \"loglikelihood\": %s,\n", py_repr_scalar(r$loglikelihood)))
  cat(sprintf("        \"aic\":           %s,\n", py_repr_scalar(r$aic)))
  cat(sprintf("        \"bic\":           %s,\n", py_repr_scalar(r$bic)))
  cat("    },\n")
}
cat("}\n")
