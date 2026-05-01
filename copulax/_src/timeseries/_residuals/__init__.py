"""Internal residual-distribution machinery for time-series models.

Provides the whitelist of allowed innovation laws and the standardisation
wrapper that maps a base CopulAX univariate distribution to a (mean=0, var=1)
form suitable for use as a GARCH/ARMA innovation.
"""
