import pytest
import numpy as np

from copulax._src.multivariate._shape import _corr
from copulax.univariate import lognormal, student_t, gamma
from copulax.copulas import gaussian_copula, student_t_copula, gh_copula


def gen_corr_matrix(num_assets: int) -> np.ndarray:
    """Generate a correlation matrix for a given number of assets."""
    # generating a random correlation matrix
    random_uniform: np.ndarray = np.random.uniform(-1, 1, size=(num_assets, num_assets))
    lower_triangular: np.ndarray = np.tril(random_uniform)
    correlation: np.ndarray = lower_triangular + lower_triangular.T
    np.fill_diagonal(correlation, 1.0)

    R = _corr._rm_incomplete(correlation, 1e-5)  # 1e-11 too small precision for float32 -> only 6-7 digits of precision
    eigenvalues = np.linalg.eigvalsh(R)
    if np.any(eigenvalues <= 0):
        return gen_corr_matrix(num_assets)
    return R


def gen_gaussian_copula_params(num_assets: int) -> dict:
    """Generate parameters for a Gaussian copula."""
    mu = np.zeros(num_assets)
    sigma = gen_corr_matrix(num_assets)
    return {"copula": {"mu": mu, "sigma": sigma}}


def gen_student_t_copula_params(num_assets: int) -> dict:
    """Generate parameters for a Student's t copula."""
    nu = 3.0 + np.abs(np.random.normal(scale=5))
    mu = np.zeros(num_assets)
    sigma = gen_corr_matrix(num_assets)
    return {"copula": {"nu": nu, "mu": mu, "sigma": sigma}}


def gen_gh_copula_params(num_assets: int) -> dict:
    """Generate parameters for a Gaussian copula."""
    scalars = np.random.normal(size=3)
    lamb = scalars[0]
    chi, psi = np.exp(scalars[1:])
    mu = np.zeros(num_assets)
    gamma = np.random.normal(size=num_assets, scale = 2.5)
    sigma = gen_corr_matrix(num_assets)
    return {"copula": {"lamb": lamb, "chi": chi, "psi": psi, 
                       "mu": mu, "gamma": gamma, "sigma": sigma}}


NUM_ASSETS: int = 3
NUM_SAMPLES: int = 100

@pytest.fixture(scope='package', autouse=True)
def continuous_sample():
    """Generate a multivariate normal sample."""
    mu = np.random.uniform(-5, 5, NUM_ASSETS)
    R = gen_corr_matrix(NUM_ASSETS)
    sigma = np.random.uniform(0.5, 2.5, NUM_ASSETS)
    sigma_diag = np.diag(sigma)
    C = sigma_diag @ R @ sigma_diag
    sample = np.random.multivariate_normal(mu, C, NUM_SAMPLES)

    # giving data a non-normal profile
    sample[:, 0] = np.exp(sample[:, 0] / np.abs(sample[:, 0]).max()) # lognormal
    sample[:, -1] = np.abs(sample[:, -1]) # strictly positive
    return sample


@pytest.fixture(scope='package', autouse=True)
def u_sample():
    """Generate a sample of uniform marginals."""
    return np.random.uniform(size=(NUM_SAMPLES, NUM_ASSETS))

def get_all_args() -> dict:
    """Generate arguments for copula tests."""
    # generating marginals
    marginals = (lognormal, {'mu': 0.0, 'sigma': 1.0}), \
                (student_t, {'nu': 3.0, 'mu': 0.0, 'sigma': 1.0}), \
                (gamma, {'alpha': 2.0, 'beta': 1.0})
    
    all_args: dict = {}
    copula_names = ('gaussian', 'student_t', 'gh')
    for name in copula_names:
        # initialising
        obj = eval(f'{name}_copula')
        all_args[obj] = {}

        # generating parameters
        copula_params = eval(f'gen_{name}_copula_params')(NUM_ASSETS)
        all_args[obj]['params'] = {'marginals': marginals, **copula_params}
        
    return all_args
# TODO: the above has been set up to be an input in a similar way to the shape and multivariate tests. 
# i,e, with the paramertrize decorator. hence these funcs should be imported into the test file
