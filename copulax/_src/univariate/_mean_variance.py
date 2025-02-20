"""File containing useful functions for mean-variance mixture distributions."""
from jax import lax


def mean_variance_ldmle_params(stats: dict, gamma: float, sample_mean: float, sample_variance: float) -> tuple[float, float]:
    # obtaining mu and sigma estimates
    mu: float = sample_mean - stats['mean'] * gamma
    sigma_sq: float = (sample_variance - stats['variance'] * lax.pow(gamma, 2)) / stats['mean'] 
    sigma: float = lax.sqrt(lax.abs(sigma_sq))
    return mu, sigma


def mean_variance_stats(w_stats: dict, mu: float, sigma: float, gamma: float) -> dict:
    # obtaining the mean and variance of the mixture distribution
    mean: float = mu + lax.mul(w_stats['mean'], gamma)
    var: float = lax.mul(w_stats['mean'], lax.pow(sigma, 2)) + lax.mul(w_stats['variance'], lax.pow(gamma, 2))
    return {'mean': mean,'variance': var}