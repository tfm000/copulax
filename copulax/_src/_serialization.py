"""Save and load fitted copulAX distribution objects.

Distributions are saved as ``.cpx`` files — ZIP archives containing a
human-readable ``metadata.json`` and binary NumPy ``.npy`` arrays for
each parameter.  The format is cross-platform (Windows, macOS, Linux),
requires no additional dependencies, and avoids ``pickle``.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------
def _get_singleton(class_name: str):
    """Look up an unparameterized template singleton by its class name.

    Searches the univariate, multivariate, and copula registries in that
    order.

    Args:
        class_name: Python class name (e.g. ``"Normal"``, ``"MvtNormal"``).

    Returns:
        An unparameterized distribution instance.

    Raises:
        ValueError: If *class_name* is not found in any registry.
    """
    from copulax.univariate.distributions import _registry as uvt_registry
    for dist in uvt_registry:
        if type(dist).__name__ == class_name:
            return dist

    from copulax.multivariate.distributions import _registry as mvt_registry
    for dist in mvt_registry:
        if type(dist).__name__ == class_name:
            return dist

    from copulax.copulas.distributions import _registry as cop_registry
    for dist in cop_registry:
        if type(dist).__name__ == class_name:
            return dist

    raise ValueError(
        f"Unknown distribution class: {class_name!r}. "
        "Ensure the distribution is registered in copulax."
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _save_distribution(dist, path) -> None:
    """Serialize a fitted distribution to a ``.cpx`` file.

    Args:
        dist: A fitted ``Distribution`` instance (must have stored
            parameters).
        path: Destination file path.  The ``.cpx`` extension is appended
            automatically when missing.

    Raises:
        ValueError: If the distribution has no stored parameters.
    """
    path = Path(path)
    if path.suffix != ".cpx":
        path = path.with_suffix(path.suffix + ".cpx")

    params = dist._stored_params
    if params is None:
        raise ValueError(
            "Cannot save an unfitted distribution (no parameters set). "
            "Fit the distribution first via .fit()."
        )

    metadata: dict = {
        "dist_family": dist.dist_type,
        "dist_dtype": dist.dtype,
        "dist_class": type(dist).__name__,
        "dist_name": dist.name,
    }

    arrays: dict[str, np.ndarray] = {}

    if dist.dist_type in ("univariate", "multivariate"):
        metadata["params"] = {}
        for key, val in params.items():
            arr = np.asarray(val)
            arrays[key] = arr
            metadata["params"][key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

    elif dist.dist_type == "copula":
        from copulax._src.copulas._distributions import Copula

        is_elliptical = isinstance(dist, Copula)
        metadata["copula_type"] = "elliptical" if is_elliptical else "archimedean"

        # Copula parameters
        copula_params = params["copula"]
        metadata["copula_params"] = {}
        for key, val in copula_params.items():
            arr = np.asarray(val)
            arrays[f"copula__{key}"] = arr
            metadata["copula_params"][key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        # Marginal distributions
        marginals = params["marginals"]
        metadata["marginals"] = []
        for i, (marginal_dist, marginal_params) in enumerate(marginals):
            m_meta: dict = {
                "dist_class": type(marginal_dist).__name__,
                "params": {},
            }
            for key, val in marginal_params.items():
                arr = np.asarray(val)
                arrays[f"marginal_{i}__{key}"] = arr
                m_meta["params"][key] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
            metadata["marginals"].append(m_meta)

    else:
        raise ValueError(f"Unsupported dist_type: {dist.dist_type!r}")

    # Write ZIP archive
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(f"arrays/{name}.npy", buf.getvalue())


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load(path, name: str = None):
    """Load a fitted distribution from a ``.cpx`` file.

    Args:
        path: Path to the ``.cpx`` file.
        name: Optional name for the loaded instance.  When ``None`` the
            name saved in the file is used.

    Returns:
        A fitted ``Distribution`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains an unknown distribution class.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json"))

        def _read_array(array_name: str) -> jnp.ndarray:
            buf = io.BytesIO(zf.read(f"arrays/{array_name}.npy"))
            return jnp.asarray(np.load(buf))

        dist_family = metadata["dist_family"]
        dist_class = metadata["dist_class"]
        dist_name = name if name is not None else metadata["dist_name"]

        template = _get_singleton(dist_class)

        if dist_family in ("univariate", "multivariate"):
            params = {
                key: _read_array(key) for key in metadata["params"]
            }
            return template._fitted_instance(params, name=dist_name)

        elif dist_family == "copula":
            # Copula parameters
            copula_params = {
                key: _read_array(f"copula__{key}")
                for key in metadata["copula_params"]
            }

            # Marginal distributions
            marginals = []
            for i, m_meta in enumerate(metadata["marginals"]):
                m_template = _get_singleton(m_meta["dist_class"])
                m_params = {
                    key: _read_array(f"marginal_{i}__{key}")
                    for key in m_meta["params"]
                }
                marginals.append((m_template, m_params))

            params = {
                "marginals": tuple(marginals),
                "copula": copula_params,
            }
            return template._fitted_instance(params, name=dist_name)

        else:
            raise ValueError(
                f"Unknown dist_family in metadata: {dist_family!r}"
            )
