# Generate rugarch ARMA-GARCH reference data for the joint composite
# tests in copulax/tests/test_timeseries_arma_garch.py.
#
# Run from project root:
#   Rscript copulax/tests/_r_reference/generate_arma_garch_reference.R \
#     > copulax/tests/_r_reference/arma_garch_reference_data.py
#
# rugarch (Galanos 2024) is the only mainstream Python/R third party
# that natively fits joint ARMA(p,q)-GARCH(p',q') models with an
# arbitrary residual law. Each reference case here:
#   1. Specifies an ARMA-GARCH spec with copulax truth parameters.
#   2. Simulates n=2000 observations from that spec.
#   3. Re-fits the same spec (free parameters) on the simulated y.
#   4. Captures parameters, log-likelihood, AIC, BIC, standard errors,
#      h=1..20 forecast mean and variance, and standardised Ljung-Box /
#      Q-stat-on-squared-residuals statistics.
#
# The output is a Python module assigning RUGARCH_REFERENCE -- already
# in copulax parameter convention. The conversion table:
#
#   rugarch coef name           copulax key
#   -------------------------   -----------------------
#   mu                          "mu"   (centred-form unconditional mean)
#   ar1, ar2, ...               "phi" (tuple)
#   ma1, ma2, ...               "theta" (tuple)
#   omega                       "omega"
#   alpha1, alpha2, ...         "alpha" (tuple)
#   beta1, beta2, ...           "beta" (tuple)
#   gamma1, gamma2, ...         "gamma" (tuple, for GJR / EGARCH)
#   shape                       residual shape param (nu / beta / alpha)
#   skew                        residual skew param (NIG beta)
#
# rugarch::infocriteria returns *per-observation* values; this script
# rescales to absolute (AIC_total = aic_per_obs * N).
#
# Variants supported here (copulax <-> rugarch):
#   GARCH      <-> sGARCH
#   IGARCH     <-> iGARCH
#   GJR_GARCH  <-> gjrGARCH
#   EGARCH     <-> eGARCH
# TGARCH and QGARCH have no equivalent rugarch model class
# (different parameterisation); those matrix configs are sourced
# separately in the test file.
#
# Residual laws supported here (copulax <-> rugarch):
#   normal     <-> norm
#   student_t  <-> std       (shape -> nu)
#   gen_normal <-> ged       (shape -> beta)
#   nig        <-> nig       (shape -> alpha, skew -> beta)
# gh and skewed_t use different parameterisations than rugarch's
# ghyp / sstd; their matrix configs are sourced separately.

suppressPackageStartupMessages(library(rugarch))

# ---------------------------------------------------------------------
# Helpers
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

py_repr_dict <- function(d) {
  if (length(d) == 0) return("{}")
  parts <- character(0)
  for (k in names(d)) {
    v <- d[[k]]
    rhs <- if (is.numeric(v) && length(v) == 1) py_repr_scalar(v)
           else if (is.numeric(v) && length(v) > 1) py_repr_tuple(v)
           else if (is.character(v) && length(v) == 1) py_repr_scalar(v)
           else stop("py_repr_dict: unsupported value type for key ", k)
    parts <- c(parts, sprintf("\"%s\": %s", k, rhs))
  }
  sprintf("{%s}", paste(parts, collapse=", "))
}

# Map rugarch coef vector -> copulax params dict. copulax now matches
# rugarch's parametrisation directly: ARMA mean uses the centred form
# (mu = unconditional mean), and EGARCH follows Nelson 1991 (alpha =
# leverage, gamma = size). Both keys map straight through with no
# reparametrisation.
to_copulax_params <- function(coef_vec, mean_order, var_model, var_order,
                              residual_dist) {
  p <- mean_order[1]; q <- mean_order[2]
  pv <- var_order[1]; qv <- var_order[2]
  out <- list()
  out[["mu"]] <- if ("mu" %in% names(coef_vec)) unname(coef_vec["mu"]) else 0.0
  ar_keys <- if (p > 0) paste0("ar", seq_len(p)) else character(0)
  ma_keys <- if (q > 0) paste0("ma", seq_len(q)) else character(0)
  out[["phi"]] <- if (p > 0) unname(coef_vec[ar_keys]) else numeric(0)
  out[["theta"]] <- if (q > 0) unname(coef_vec[ma_keys]) else numeric(0)
  out[["omega"]] <- unname(coef_vec["omega"])
  alpha_keys <- if (pv > 0) paste0("alpha", seq_len(pv)) else character(0)
  beta_keys  <- if (qv > 0) paste0("beta",  seq_len(qv)) else character(0)
  gamma_keys <- if (pv > 0) paste0("gamma", seq_len(pv)) else character(0)
  out[["alpha"]] <- if (pv > 0) unname(coef_vec[alpha_keys]) else numeric(0)
  if (var_model %in% c("gjrGARCH", "eGARCH")) {
    out[["gamma"]] <- if (pv > 0) unname(coef_vec[gamma_keys]) else numeric(0)
  }
  out[["beta"]] <- if (qv > 0) unname(coef_vec[beta_keys]) else numeric(0)
  out[["residual"]] <- residual_shape_to_copulax(coef_vec, residual_dist)
  out
}

residual_shape_to_copulax <- function(coef_vec, residual_dist) {
  switch(residual_dist,
    "norm" = list(),
    "std"  = list(nu = unname(coef_vec["shape"])),
    "ged"  = list(beta = unname(coef_vec["shape"])),
    "nig"  = list(alpha = unname(coef_vec["shape"]),
                  beta  = unname(coef_vec["skew"])),
    stop("residual_shape_to_copulax: unsupported dist ", residual_dist)
  )
}

cx_residual_name <- function(rugarch_name) {
  switch(rugarch_name,
    "norm" = "normal",
    "std"  = "student_t",
    "ged"  = "gen_normal",
    "nig"  = "nig",
    stop("cx_residual_name: unsupported ", rugarch_name)
  )
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
# Per-case driver
# ---------------------------------------------------------------------

process_case <- function(label, mean_order, var_model, var_order,
                         residual_dist, fixed_pars, residual_shape_truth,
                         n_sim = 2000, seed = 13) {
  vlist <- list(model=var_model, garchOrder=var_order)
  spec_truth <- ugarchspec(
    mean.model=list(armaOrder=mean_order, include.mean=TRUE),
    variance.model=vlist,
    distribution.model=residual_dist,
    fixed.pars=fixed_pars
  )
  set.seed(seed)
  sim <- ugarchpath(spec_truth, n.sim=n_sim, m.sim=1)
  y <- as.numeric(fitted(sim))
  spec <- ugarchspec(
    mean.model=list(armaOrder=mean_order, include.mean=TRUE),
    variance.model=vlist,
    distribution.model=residual_dist
  )
  fit <- ugarchfit(spec=spec, data=y, solver="hybrid")
  cf <- coef(fit)
  # se.coef aligns with FREE parameters only; iGARCH (and any constrained
  # variant) drops constrained params. Use matcoef which keeps all rows;
  # constrained rows have NA for SE.
  mc <- fit@fit$matcoef
  # rugarch emits column headers with leading spaces (" Estimate",
  # " Std. Error", " t value"); index by position to avoid that.
  se <- mc[, 2]; names(se) <- rownames(mc)

  # Mean-intercept convention: rugarch parameterises the ARMA mean
  # equation in centred form
  #   y_t = mu + phi (y_{t-1} - mu) + theta eps_{t-1} + eps_t
  # copulax now follows the same centred convention, so rugarch's
  # `mu` (the unconditional mean) maps directly to copulax's `mu`
  # with no reparametrisation.
  ll <- as.numeric(fit@fit$LLH)
  N <- length(y)
  ic <- infocriteria(fit)
  aic_total <- as.numeric(ic[1, 1]) * N
  bic_total <- as.numeric(ic[2, 1]) * N
  fc <- ugarchforecast(fit, n.ahead=20)
  fc_mean <- as.numeric(fitted(fc))
  fc_sd   <- as.numeric(sigma(fc))
  fc_var  <- fc_sd ^ 2
  z <- as.numeric(residuals(fit, standardize=TRUE))
  lb     <- Box.test(z,    lag=10, type="Ljung-Box")
  lb_sq  <- Box.test(z^2,  lag=10, type="Ljung-Box")

  params_cx <- to_copulax_params(cf, mean_order, var_model, var_order,
                                 residual_dist)
  se_cx     <- to_copulax_params(se, mean_order, var_model, var_order,
                                 residual_dist)

  list(
    label                 = label,
    mean_order            = mean_order,
    var_model             = cx_var_model_name(var_model),
    var_order             = var_order,
    residual_dist         = cx_residual_name(residual_dist),
    residual_shape_truth  = residual_shape_truth,
    y                     = y,
    params                = params_cx,
    standard_errors       = se_cx,
    loglikelihood         = ll,
    aic                   = aic_total,
    bic                   = bic_total,
    forecast_mean         = fc_mean,
    forecast_variance     = fc_var,
    ljung_box_statistic   = as.numeric(lb$statistic),
    ljung_box_pvalue      = as.numeric(lb$p.value),
    ljung_box_sq_statistic = as.numeric(lb_sq$statistic),
    ljung_box_sq_pvalue    = as.numeric(lb_sq$p.value)
  )
}

# ---------------------------------------------------------------------
# Curated reference set
# ---------------------------------------------------------------------

CASES <- list(
  # 1) Variant sweep at ARMA(1,1) x var_order(1,1) x norm.
  list(label="arma11_garch11_normal",
       mean_order=c(1,1), var_model="sGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(),
       seed=11),
  list(label="arma11_igarch11_normal",
       mean_order=c(1,1), var_model="iGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.02, alpha1=0.10),
       residual_shape_truth=list(),
       seed=12),
  list(label="arma11_gjr11_normal",
       mean_order=c(1,1), var_model="gjrGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.05, beta1=0.85, gamma1=0.10),
       residual_shape_truth=list(),
       seed=13),
  list(label="arma11_egarch11_normal",
       mean_order=c(1,1), var_model="eGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.0, alpha1=-0.05, beta1=0.95, gamma1=0.10),
       residual_shape_truth=list(),
       seed=14),

  # 2) Residual sweep at ARMA(1,1) x sGARCH(1,1).
  list(label="arma11_garch11_studentt",
       mean_order=c(1,1), var_model="sGARCH", var_order=c(1,1),
       residual_dist="std",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.10, beta1=0.85, shape=6.0),
       residual_shape_truth=list(nu=6.0),
       seed=21),
  list(label="arma11_garch11_gennormal",
       mean_order=c(1,1), var_model="sGARCH", var_order=c(1,1),
       residual_dist="ged",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.10, beta1=0.85, shape=1.5),
       residual_shape_truth=list(beta=1.5),
       seed=22),
  list(label="arma11_garch11_nig",
       mean_order=c(1,1), var_model="sGARCH", var_order=c(1,1),
       residual_dist="nig",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.10, beta1=0.85,
                       shape=2.0, skew=0.1),
       residual_shape_truth=list(alpha=2.0, beta=0.1),
       seed=23),

  # 3) Mean-order sweep at sGARCH(1,1) x norm.
  list(label="ar1_garch11_normal",
       mean_order=c(1,0), var_model="sGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5,
                       omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(),
       seed=31),
  list(label="ma1_garch11_normal",
       mean_order=c(0,1), var_model="sGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ma1=0.3,
                       omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(),
       seed=32),
  list(label="arma21_garch11_normal",
       mean_order=c(2,1), var_model="sGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.4, ar2=0.2, ma1=0.3,
                       omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(),
       seed=33),
  list(label="arma12_garch11_normal",
       mean_order=c(1,2), var_model="sGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3, ma2=0.1,
                       omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(),
       seed=34),
  list(label="arma22_garch11_normal",
       mean_order=c(2,2), var_model="sGARCH", var_order=c(1,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.4, ar2=0.2, ma1=0.3, ma2=0.1,
                       omega=0.05, alpha1=0.10, beta1=0.85),
       residual_shape_truth=list(),
       seed=35),

  # 4) Variance-order sweep at ARMA(1,1) x norm.
  list(label="arma11_garch12_normal",
       mean_order=c(1,1), var_model="sGARCH", var_order=c(1,2),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.05, beta1=0.45, beta2=0.45),
       residual_shape_truth=list(),
       seed=41),
  list(label="arma11_garch21_normal",
       mean_order=c(1,1), var_model="sGARCH", var_order=c(2,1),
       residual_dist="norm",
       fixed_pars=list(mu=0.05, ar1=0.5, ma1=0.3,
                       omega=0.05, alpha1=0.05, alpha2=0.05, beta1=0.85),
       residual_shape_truth=list(),
       seed=42)
)

# ---------------------------------------------------------------------
# Drive every case and emit Python output to stdout.
# ---------------------------------------------------------------------

cat("\"\"\"Auto-generated rugarch reference data for ARMA-GARCH joint composite tests.\n\n")
cat("Regenerate with:\n")
cat("    Rscript copulax/tests/_r_reference/generate_arma_garch_reference.R \\\n")
cat("        > copulax/tests/_r_reference/arma_garch_reference_data.py\n\n")
cat("rugarch ", as.character(packageVersion("rugarch")),
    " on R ", paste(R.Version()$major, R.Version()$minor, sep="."),
    ".\n", sep="")
cat("\"\"\"\n\n")
cat("import numpy as np\n\n")
cat("RUGARCH_REFERENCE = {\n")

for (cfg in CASES) {
  res <- do.call(process_case, c(
    list(label=cfg$label,
         mean_order=cfg$mean_order,
         var_model=cfg$var_model,
         var_order=cfg$var_order,
         residual_dist=cfg$residual_dist,
         fixed_pars=cfg$fixed_pars,
         residual_shape_truth=cfg$residual_shape_truth),
    if (!is.null(cfg$seed)) list(seed=cfg$seed) else list()
  ))

  cat(sprintf("    \"%s\": {\n", res$label))
  cat(sprintf("        \"mean_order\":           (%d, %d),\n",
              res$mean_order[1], res$mean_order[2]))
  cat(sprintf("        \"var_model\":            \"%s\",\n", res$var_model))
  cat(sprintf("        \"var_order\":            (%d, %d),\n",
              res$var_order[1], res$var_order[2]))
  cat(sprintf("        \"residual_dist\":        \"%s\",\n", res$residual_dist))
  cat(sprintf("        \"residual_shape_truth\": %s,\n",
              py_repr_dict(res$residual_shape_truth)))
  cat(sprintf("        \"y\":                    %s,\n",
              py_repr_array(res$y)))
  # Vector-shaped param keys: always emit as tuple (even length 1) to
  # match copulax's params dict convention.
  vec_keys <- c("phi", "theta", "alpha", "beta", "gamma")
  emit_param_dict <- function(d) {
    for (k in names(d)) {
      v <- d[[k]]
      if (is.list(v)) {
        cat(sprintf("            \"%s\": %s,\n", k, py_repr_dict(v)))
      } else if (k %in% vec_keys) {
        cat(sprintf("            \"%s\": %s,\n", k, py_repr_tuple(v)))
      } else if (length(v) == 1) {
        cat(sprintf("            \"%s\": %s,\n", k, py_repr_scalar(v)))
      } else {
        cat(sprintf("            \"%s\": %s,\n", k, py_repr_tuple(v)))
      }
    }
  }
  cat("        \"params\": {\n");           emit_param_dict(res$params);          cat("        },\n")
  cat("        \"standard_errors\": {\n");  emit_param_dict(res$standard_errors); cat("        },\n")
  cat(sprintf("        \"loglikelihood\":          %s,\n",
              py_repr_scalar(res$loglikelihood)))
  cat(sprintf("        \"aic\":                    %s,\n",
              py_repr_scalar(res$aic)))
  cat(sprintf("        \"bic\":                    %s,\n",
              py_repr_scalar(res$bic)))
  cat(sprintf("        \"forecast_mean\":          %s,\n",
              py_repr_array(res$forecast_mean)))
  cat(sprintf("        \"forecast_variance\":      %s,\n",
              py_repr_array(res$forecast_variance)))
  cat(sprintf("        \"ljung_box_statistic\":    %s,\n",
              py_repr_scalar(res$ljung_box_statistic)))
  cat(sprintf("        \"ljung_box_pvalue\":       %s,\n",
              py_repr_scalar(res$ljung_box_pvalue)))
  cat(sprintf("        \"ljung_box_sq_statistic\": %s,\n",
              py_repr_scalar(res$ljung_box_sq_statistic)))
  cat(sprintf("        \"ljung_box_sq_pvalue\":    %s,\n",
              py_repr_scalar(res$ljung_box_sq_pvalue)))
  cat("    },\n")
}

cat("}\n")
