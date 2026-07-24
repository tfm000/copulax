# Generate rugarch GARCH-in-mean (GARCH-M) reference data for the
# Layer-1 formula tests in copulax/tests/test_timeseries_variance.py.
#
# Run from project root:
#   Rscript copulax/tests/_r_reference/generate_garch_m_reference.R \
#     > copulax/tests/_r_reference/garch_m_reference_data.py
#
# GARCH-M convention (RQ2, VERIFIED)
# ----------------------------------
# copulax GARCH_M uses the variance-in-mean form
#     y_t = mu + lambda_m * sigma^2_t + eps_t           (garch_m.py)
# i.e. the CONDITIONAL VARIANCE (sigma^2), not the standard deviation,
# enters the mean. rugarch's archm term adds archm * sigma^archpow to
# the mean; archpow=2 -> sigma^2, archpow=1 -> sigma. We therefore
# generate with archm=TRUE, archpow=2 so rugarch's `archm` coefficient
# maps DIRECTLY to copulax `lambda_m`. Using archpow=1 would compare
# sigma-in-mean (rugarch) against sigma^2-in-mean (copulax) -- a
# convention mismatch, so it is deliberately avoided.
#
# Conversion table (rugarch coef name -> copulax key):
#   mu     -> "mu"        (intercept of the mean equation)
#   archm  -> "lambda_m"  (variance-in-mean coefficient; archpow=2)
#   omega  -> "omega"
#   alpha1 -> "alpha" (tuple)
#   beta1  -> "beta"  (tuple)
#   shape  -> residual nu (student-t)
#
# Layer-1 (formula) match convention
# ----------------------------------
# For GARCH-M rugarch fixes the leading max(p, q) conditional variances
# at mean((y - mu)^2) -- residuals formed from the intercept ONLY,
# excluding the variance-in-mean term (the archm term depends on
# sigma^2, which is not available pre-sample). copulax reproduces this
# exactly via init="squared" (its GARCH-M warm-up level is
# mean((y - mu)^2) at the fitted mu). Verified against rugarch's
# reported sigma^2 path and LLH to machine precision.
#
# rugarch is the only oracle here: arch (Python) does not fit GARCH-M in
# this variance-in-mean form, so GARCH-M is rugarch-only.

suppressPackageStartupMessages(library(rugarch))

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

cx_residual_name <- function(rugarch_name) {
  switch(rugarch_name,
    "norm" = "normal",
    "std"  = "student_t",
    stop("cx_residual_name: unsupported ", rugarch_name)
  )
}

process_case <- function(label, var_order, residual_dist, fixed_pars,
                         residual_shape_truth, n_sim = 2000, seed = 51) {
  pv <- var_order[1]; qv <- var_order[2]
  # archm=TRUE, archpow=2 (RQ2): sigma^2-in-mean.
  mm <- list(armaOrder=c(0,0), include.mean=TRUE, archm=TRUE, archpow=2)
  vlist <- list(model="sGARCH", garchOrder=var_order)
  spec_truth <- ugarchspec(
    mean.model=mm, variance.model=vlist,
    distribution.model=residual_dist, fixed.pars=fixed_pars
  )
  sim <- ugarchpath(spec_truth, n.sim=n_sim, m.sim=1, rseed=seed)
  y <- as.numeric(fitted(sim))
  spec <- ugarchspec(mean.model=mm, variance.model=vlist,
                     distribution.model=residual_dist)
  fit <- ugarchfit(spec=spec, data=y, solver="hybrid")
  cf <- coef(fit)
  mc <- fit@fit$matcoef
  se <- mc[, 2]; names(se) <- rownames(mc)

  ll <- as.numeric(fit@fit$LLH)
  N <- length(y)
  ic <- infocriteria(fit)
  aic_total <- as.numeric(ic[1, 1]) * N
  bic_total <- as.numeric(ic[2, 1]) * N
  sigma2 <- as.numeric(sigma(fit))^2

  alpha_keys <- if (pv > 0) paste0("alpha", seq_len(pv)) else character(0)
  beta_keys  <- if (qv > 0) paste0("beta",  seq_len(qv)) else character(0)

  to_cx <- function(vec) {
    list(
      mu       = unname(vec["mu"]),
      lambda_m = unname(vec["archm"]),
      omega    = unname(vec["omega"]),
      alpha    = if (pv > 0) unname(vec[alpha_keys]) else numeric(0),
      beta     = if (qv > 0) unname(vec[beta_keys]) else numeric(0)
    )
  }
  residual_out <- switch(residual_dist,
    "norm" = list(), "std" = list(nu = unname(cf["shape"])))
  residual_se <- switch(residual_dist,
    "norm" = list(), "std" = list(nu = unname(se["shape"])))

  list(
    label                = label,
    var_order            = var_order,
    residual_dist        = cx_residual_name(residual_dist),
    residual_shape_truth = residual_shape_truth,
    y                    = y,
    params               = to_cx(cf),
    residual             = residual_out,
    standard_errors      = to_cx(se),
    residual_se          = residual_se,
    loglikelihood        = ll,
    aic                  = aic_total,
    bic                  = bic_total,
    sigma2               = sigma2
  )
}

CASES <- list(
  list(label="garchm11_normal", var_order=c(1,1), residual_dist="norm",
       fixed_pars=list(mu=0.02, archm=0.10, omega=0.05, alpha1=0.10,
                       beta1=0.85),
       residual_shape_truth=list(), seed=51),
  list(label="garchm11_studentt", var_order=c(1,1), residual_dist="std",
       fixed_pars=list(mu=0.02, archm=0.10, omega=0.05, alpha1=0.10,
                       beta1=0.85, shape=6.0),
       residual_shape_truth=list(nu=6.0), seed=52)
)

cat("\"\"\"Auto-generated rugarch GARCH-M reference data (archm=TRUE, archpow=2).\n\n")
cat("Regenerate with:\n")
cat("    Rscript copulax/tests/_r_reference/generate_garch_m_reference.R \\\n")
cat("        > copulax/tests/_r_reference/garch_m_reference_data.py\n\n")
cat("rugarch ", as.character(packageVersion("rugarch")),
    " on R ", paste(R.Version()$major, R.Version()$minor, sep="."),
    ".\n\n", sep="")
cat("GARCH-M is generated with archm=TRUE, archpow=2 so rugarch's archm\n")
cat("coefficient maps directly to copulax lambda_m (variance-in-mean,\n")
cat("y_t = mu + lambda_m * sigma^2_t + eps_t). Layer-1 uses copulax\n")
cat("init=\"squared\"; for GARCH-M the fixed pre-sample level is\n")
cat("mean((y - mu)^2) (intercept-only residuals), matched two-sided at\n")
cat("rtol <= 1e-8 in test_timeseries_variance.py::TestGarchMReference.\n")
cat("\"\"\"\n\n")
cat("import numpy as np\n\n")
cat("GARCH_M_REFERENCE = {\n")

for (cfg in CASES) {
  res <- process_case(
    label=cfg$label, var_order=cfg$var_order,
    residual_dist=cfg$residual_dist, fixed_pars=cfg$fixed_pars,
    residual_shape_truth=cfg$residual_shape_truth, seed=cfg$seed
  )
  emit_param_dict <- function(d) {
    for (k in names(d)) {
      v <- d[[k]]
      if (k %in% c("alpha", "beta")) {
        cat(sprintf("            \"%s\": %s,\n", k, py_repr_tuple(v)))
      } else {
        cat(sprintf("            \"%s\": %s,\n", k, py_repr_scalar(v)))
      }
    }
  }
  emit_residual_dict <- function(d) {
    if (length(d) == 0) { cat("{}"); return(invisible(NULL)) }
    parts <- sapply(names(d), function(k) sprintf("\"%s\": %s", k, py_repr_scalar(d[[k]])))
    cat(sprintf("{%s}", paste(parts, collapse=", ")))
  }

  cat(sprintf("    \"%s\": {\n", res$label))
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
  cat("    },\n")
}

cat("}\n")
