# Generate rugarch reference data for the STANDALONE variance-model
# Layer-1 formula tests in copulax/tests/test_timeseries_variance.py.
#
# Run from project root:
#   Rscript copulax/tests/_r_reference/generate_garch_standalone_reference.R \
#     > copulax/tests/_r_reference/garch_standalone_reference_data.py
#
# rugarch (Galanos) is the primary GARCH oracle. Each reference case:
#   1. Specifies a standalone variance spec (armaOrder=c(0,0),
#      include.mean=FALSE) with copulax truth parameters.
#   2. Simulates n=2000 observations from that spec.
#   3. Re-fits the same spec (free parameters) on the simulated y.
#   4. Captures fitted params, log-likelihood, AIC, BIC (rescaled
#      per-obs -> absolute via *N), standard errors, and the FULL
#      fitted conditional-variance path sigma^2_t (needed for the
#      two-sided Layer-1 recursion match).
#
# Layer-1 (formula) match convention
# ----------------------------------
# rugarch fixes the leading max(p, q) conditional variances at the
# unconditional-variance estimate mean(residuals^2) (its rec.init
# control, default "all": "uses all the values for the unconditional
# variance calculation") and starts the recursion proper at index
# max(p, q). copulax reproduces this exactly via its opt-in "squared"
# pre-sample initialisation mode, so the Layer-1 test evaluates
# copulax at rugarch's fitted params with init="squared" and matches
# the sigma^2 path and log-likelihood two-sided at rtol <= 1e-8.
#
# The output is a Python module assigning GARCH_STANDALONE_REFERENCE --
# already in copulax parameter convention. The conversion table
# (rugarch coef name -> copulax key), per variant:
#
#   sGARCH   (GARCH):    omega->omega, alpha1..->alpha (tuple),
#                        beta1..->beta (tuple).
#   iGARCH   (IGARCH):   omega->omega, alpha1..->alpha (tuple).
#                        rugarch drops the last beta as the constrained
#                        parameter of the alpha+beta=1 simplex; copulax
#                        stores beta = 1 - sum(alpha) explicitly (the
#                        IGARCH simplex). ONLY the free params carry an
#                        SE; the constrained beta SE is NA.
#   gjrGARCH (GJR_GARCH):omega->omega, alpha1..->alpha (tuple),
#                        gamma1..->gamma (tuple, leverage on
#                        eps^2 * 1{eps<0}), beta1..->beta (tuple).
#   eGARCH   (EGARCH):   omega->omega, alpha1..->alpha (tuple),
#                        gamma1..->gamma (tuple), beta1..->beta (tuple).
#                        copulax follows Nelson (1991): alpha = leverage
#                        (coefficient on z_{t-i}), gamma = size
#                        (coefficient on |z_{t-i}| - E|z|). rugarch
#                        reports the SAME labelling (alpha=leverage,
#                        gamma=size), so both keys map straight through
#                        with no swap -- verified against the fitted
#                        sigma^2 path to machine precision.
#
# Residual laws (copulax <-> rugarch):
#   normal    <-> norm
#   student_t <-> std   (shape -> nu)
#
# arch (Python, 8.0.0) is the second oracle for GARCH/GJR/EGARCH but
# uses an EWMA backcast pre-sample (decay 0.94) rather than rugarch's
# mean(residuals^2), so it does NOT reproduce rugarch's sigma^2 path at
# rtol <= 1e-8. arch therefore enters the suite as an independent
# fitted-parameter / log-likelihood oracle (recorded and checked in the
# Python test with pytest.importorskip("arch")), not as a Layer-1
# recursion oracle. arch does NOT fit IGARCH in this standalone form, so
# IGARCH is rugarch-only (annotated below).

suppressPackageStartupMessages(library(rugarch))

# ---------------------------------------------------------------------
# Python-repr helpers
# ---------------------------------------------------------------------
py_repr_scalar <- function(x) {
  if (is.logical(x)) return(if (x) "True" else "False")
  if (is.character(x)) return(sprintf("\"%s\"", x))
  if (is.na(x) || is.nan(x)) return("float('nan')")
  if (is.infinite(x)) return(if (x > 0) "float('inf')" else "float('-inf')")
  sprintf("%.17g", x)
}

py_repr_array <- function(x) {
  if (length(x) == 0) return("np.array([], dtype=float)")
  body <- paste(sapply(x, py_repr_scalar), collapse=", ")
  sprintf("np.array([%s], dtype=float)", body)
}

py_repr_tuple <- function(x) {
  if (length(x) == 0) return("()")
  body <- paste(sapply(x, py_repr_scalar), collapse=", ")
  if (length(x) == 1) sprintf("(%s,)", body) else sprintf("(%s)", body)
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

cx_residual_name <- function(rugarch_name) {
  switch(rugarch_name,
    "norm" = "normal",
    "std"  = "student_t",
    stop("cx_residual_name: unsupported ", rugarch_name)
  )
}

# ---------------------------------------------------------------------
# Per-case driver
# ---------------------------------------------------------------------
process_case <- function(label, var_model, var_order, residual_dist,
                         fixed_pars, residual_shape_truth,
                         n_sim = 2000, seed = 13) {
  pv <- var_order[1]; qv <- var_order[2]
  vlist <- list(model=var_model, garchOrder=var_order)
  spec_truth <- ugarchspec(
    mean.model=list(armaOrder=c(0,0), include.mean=FALSE),
    variance.model=vlist,
    distribution.model=residual_dist,
    fixed.pars=fixed_pars
  )
  sim <- ugarchpath(spec_truth, n.sim=n_sim, m.sim=1, rseed=seed)
  y <- as.numeric(fitted(sim))
  spec <- ugarchspec(
    mean.model=list(armaOrder=c(0,0), include.mean=FALSE),
    variance.model=vlist,
    distribution.model=residual_dist
  )
  fit <- ugarchfit(spec=spec, data=y, solver="hybrid")
  cf <- coef(fit)
  # matcoef keeps all rows including constrained ones (NA SE).
  mc <- fit@fit$matcoef
  se <- mc[, 2]; names(se) <- rownames(mc)

  ll <- as.numeric(fit@fit$LLH)
  N <- length(y)
  ic <- infocriteria(fit)
  aic_total <- as.numeric(ic[1, 1]) * N
  bic_total <- as.numeric(ic[2, 1]) * N
  sigma2 <- as.numeric(sigma(fit))^2
  # rugarch's reported UNCONDITIONAL variance -- the closed-form
  # long-run Var(eps) implied by the fitted coefficients (NOT a path
  # quantity). Captured here as the third-party oracle for copulax's
  # stats()["unconditional_variance"] accessor. Per-variant behaviour
  # (VERIFIED empirically, 01-MATH-REVIEW.md unconditional-variance
  # third-party section):
  #   sGARCH   -> omega/(1 - sum(alpha) - sum(beta))          [closed form]
  #   iGARCH   -> Inf (persistence == 1: variance does not exist)
  #   gjrGARCH -> omega/(1 - alpha - beta - 0.5*gamma)  (rugarch fixes
  #               the leverage expectation at kappa = E[z^2 1{z<0}] = 0.5
  #               for ALL residual laws; copulax computes kappa by
  #               quadrature, == 0.5 for symmetric laws)
  #   eGARCH   -> exp(omega/(1 - sum(beta)))  (Nelson geometric-mean
  #               convention; residual-law-independent)
  uncvar <- tryCatch(uncvariance(fit),
                     error = function(e) NA_real_)

  # --- convert coef vector -> copulax params dict (per variant) ---
  alpha_keys <- if (pv > 0) paste0("alpha", seq_len(pv)) else character(0)
  beta_keys  <- if (qv > 0) paste0("beta",  seq_len(qv)) else character(0)
  gamma_keys <- if (pv > 0) paste0("gamma", seq_len(pv)) else character(0)

  # rugarch reports every coefficient -- including IGARCH's pinned
  # beta -- directly in coef()/matcoef (verified: iGARCH(1,1) returns
  # omega, alpha1, beta1 with alpha1 + beta1 == 1 exactly, and the
  # pinned beta1's SE is NA in matcoef). copulax stores the same beta
  # tuple under its alpha+beta=1 simplex, so the value conversion is
  # uniform across variants; only the pinned beta's SE carries through
  # as NaN.
  to_cx <- function(vec) {
    out <- list()
    out[["omega"]] <- unname(vec["omega"])
    out[["alpha"]] <- if (pv > 0) unname(vec[alpha_keys]) else numeric(0)
    if (var_model %in% c("gjrGARCH", "eGARCH")) {
      out[["gamma"]] <- if (pv > 0) unname(vec[gamma_keys]) else numeric(0)
    }
    out[["beta"]] <- if (qv > 0) unname(vec[beta_keys]) else numeric(0)
    out
  }
  to_cx_se <- to_cx
  residual_out <- switch(residual_dist,
    "norm" = list(),
    "std"  = list(nu = unname(cf["shape"]))
  )
  residual_se <- switch(residual_dist,
    "norm" = list(),
    "std"  = list(nu = unname(se["shape"]))
  )

  list(
    label                = label,
    var_model            = cx_var_model_name(var_model),
    var_order            = var_order,
    residual_dist        = cx_residual_name(residual_dist),
    residual_shape_truth = residual_shape_truth,
    y                    = y,
    params               = to_cx(cf),
    residual             = residual_out,
    standard_errors      = to_cx_se(se),
    residual_se          = residual_se,
    loglikelihood        = ll,
    aic                  = aic_total,
    bic                  = bic_total,
    sigma2               = sigma2,
    uncvariance          = uncvar
  )
}

# ---------------------------------------------------------------------
# Curated reference set: {GARCH, IGARCH, GJR, EGARCH} x {norm, std}
# ---------------------------------------------------------------------
CASES <- list(
  list(label="garch11_normal",  var_model="sGARCH",   var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(), seed=11),
  list(label="garch11_studentt", var_model="sGARCH",  var_order=c(1,1),
       residual_dist="std",
       fixed_pars=list(omega=0.05, alpha1=0.10, beta1=0.85, shape=6.0),
       residual_shape_truth=list(nu=6.0), seed=21),

  # IGARCH: alpha + beta = 1 simplex. rugarch-only (arch has no
  # standalone IGARCH in this form).
  list(label="igarch11_normal", var_model="iGARCH",   var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(omega=0.02, alpha1=0.10),
       residual_shape_truth=list(), seed=12),
  list(label="igarch11_studentt", var_model="iGARCH", var_order=c(1,1),
       residual_dist="std",
       fixed_pars=list(omega=0.02, alpha1=0.10, shape=6.0),
       residual_shape_truth=list(nu=6.0), seed=22),

  list(label="gjr11_normal",    var_model="gjrGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(omega=0.05, alpha1=0.05, beta1=0.85, gamma1=0.10),
       residual_shape_truth=list(), seed=13),
  list(label="gjr11_studentt",  var_model="gjrGARCH", var_order=c(1,1),
       residual_dist="std",
       fixed_pars=list(omega=0.05, alpha1=0.05, beta1=0.85, gamma1=0.10,
                       shape=7.0),
       residual_shape_truth=list(nu=7.0), seed=23),

  list(label="egarch11_normal", var_model="eGARCH",   var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(omega=0.0, alpha1=-0.05, beta1=0.95, gamma1=0.10),
       residual_shape_truth=list(), seed=14),
  list(label="egarch11_studentt", var_model="eGARCH", var_order=c(1,1),
       residual_dist="std",
       fixed_pars=list(omega=0.0, alpha1=-0.05, beta1=0.95, gamma1=0.10,
                       shape=8.0),
       residual_shape_truth=list(nu=8.0), seed=24)
)

# ---------------------------------------------------------------------
# Emit Python output to stdout.
# ---------------------------------------------------------------------
cat("\"\"\"Auto-generated rugarch reference data for standalone variance-model tests.\n\n")
cat("Regenerate with:\n")
cat("    Rscript copulax/tests/_r_reference/generate_garch_standalone_reference.R \\\n")
cat("        > copulax/tests/_r_reference/garch_standalone_reference_data.py\n\n")
cat("rugarch ", as.character(packageVersion("rugarch")),
    " on R ", paste(R.Version()$major, R.Version()$minor, sep="."),
    ".\n\n", sep="")
cat("Layer-1 convention: rugarch fixes the leading max(p, q) conditional\n")
cat("variances at mean(residuals^2) (rec.init=\"all\") and recurses from\n")
cat("index max(p, q); copulax reproduces this with init=\"squared\". The\n")
cat("stored `sigma2` is the full fitted conditional-variance path and\n")
cat("`loglikelihood` the reported LLH -- both matched two-sided at\n")
cat("rtol <= 1e-8 in test_timeseries_variance.py::TestGarchStandaloneReference.\n")
cat("EGARCH uses the Nelson (1991) labelling (alpha=leverage, gamma=size),\n")
cat("identical to rugarch's, so no parameter swap is applied.\n\n")
cat("`uncvariance` is rugarch's reported unconditional variance (the\n")
cat("closed-form long-run Var(eps) implied by the fitted coefficients) --\n")
cat("the third-party oracle for copulax's stats()[\"unconditional_variance\"]\n")
cat("accessor (TestUnconditionalVarianceThirdParty). sGARCH -> omega/(1 -\n")
cat("sum alpha - sum beta); iGARCH -> inf (does not exist, persistence=1);\n")
cat("gjrGARCH -> omega/(1 - alpha - beta - 0.5*gamma) (rugarch fixes\n")
cat("kappa=E[z^2 1{z<0}]=0.5 for all laws); eGARCH -> exp(omega/(1 - sum\n")
cat("beta)) (Nelson geometric-mean convention).\n")
cat("\"\"\"\n\n")
cat("import numpy as np\n\n")
cat("GARCH_STANDALONE_REFERENCE = {\n")

for (cfg in CASES) {
  res <- process_case(
    label=cfg$label, var_model=cfg$var_model, var_order=cfg$var_order,
    residual_dist=cfg$residual_dist, fixed_pars=cfg$fixed_pars,
    residual_shape_truth=cfg$residual_shape_truth, seed=cfg$seed
  )

  emit_param_dict <- function(d) {
    for (k in names(d)) {
      v <- d[[k]]
      cat(sprintf("            \"%s\": %s,\n", k, py_repr_tuple(v)))
    }
  }
  emit_residual_dict <- function(d) {
    if (length(d) == 0) { cat("{}"); return(invisible(NULL)) }
    parts <- sapply(names(d), function(k) sprintf("\"%s\": %s", k, py_repr_scalar(d[[k]])))
    cat(sprintf("{%s}", paste(parts, collapse=", ")))
  }

  cat(sprintf("    \"%s\": {\n", res$label))
  cat(sprintf("        \"var_model\":            \"%s\",\n", res$var_model))
  cat(sprintf("        \"var_order\":            (%d, %d),\n",
              res$var_order[1], res$var_order[2]))
  cat(sprintf("        \"residual_dist\":        \"%s\",\n", res$residual_dist))
  cat(sprintf("        \"residual_shape_truth\": "))
  emit_residual_dict(res$residual_shape_truth); cat(",\n")
  cat(sprintf("        \"y\":                    %s,\n", py_repr_array(res$y)))
  cat("        \"params\": {\n"); emit_param_dict(res$params); cat("        },\n")
  cat(sprintf("        \"residual\":             "))
  emit_residual_dict(res$residual); cat(",\n")
  cat("        \"standard_errors\": {\n"); emit_param_dict(res$standard_errors); cat("        },\n")
  cat(sprintf("        \"residual_se\":          "))
  emit_residual_dict(res$residual_se); cat(",\n")
  cat(sprintf("        \"loglikelihood\":        %s,\n", py_repr_scalar(res$loglikelihood)))
  cat(sprintf("        \"aic\":                  %s,\n", py_repr_scalar(res$aic)))
  cat(sprintf("        \"bic\":                  %s,\n", py_repr_scalar(res$bic)))
  cat(sprintf("        \"sigma2\":               %s,\n", py_repr_array(res$sigma2)))
  cat(sprintf("        \"uncvariance\":          %s,\n", py_repr_scalar(res$uncvariance)))
  cat("    },\n")
}

cat("}\n")
