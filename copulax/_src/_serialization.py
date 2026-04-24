"""Save and load fitted copulAX distribution and preprocessing objects.

Fitted objects are saved as ``.cpx`` files — ZIP archives containing a
human-readable ``metadata.json`` and binary NumPy ``.npy`` arrays for
each parameter. The format is cross-platform (Windows, macOS, Linux),
requires no additional dependencies, and avoids ``pickle``.

Supported families:

* Univariate, multivariate, and copula distributions (legacy behaviour).
* ``copulax.preprocessing`` objects such as :class:`DataScaler`, which
  additionally may carry user-supplied callables on ``pre_fns`` /
  ``post_fns``. Those callables are serialised by import-path qualname —
  lambdas and locally-defined functions are rejected at save time.
"""

import importlib
import io
import json
import warnings
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
        from copulax._src.copulas._distributions import (
            EllipticalCopula,
            MeanVarianceCopula,
            MeanVarianceCopulaBase,
        )

        # Tag with the most specific known taxonomic category.  The
        # serialiser-side reader only needs to distinguish "is this a
        # mean-variance / elliptical-style copula" from "Archimedean",
        # but we record the finer grain too for forward compatibility.
        if isinstance(dist, EllipticalCopula):
            metadata["copula_type"] = "elliptical"
        elif isinstance(dist, MeanVarianceCopula):
            metadata["copula_type"] = "mean_variance"
        elif isinstance(dist, MeanVarianceCopulaBase):
            # Future MV-base subclass that's neither Elliptical nor
            # MeanVariance — fall back to the umbrella label.
            metadata["copula_type"] = "mean_variance_base"
        else:
            metadata["copula_type"] = "archimedean"

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
# Callable serialisation (for preprocessing objects with pre_fns/post_fns)
# ---------------------------------------------------------------------------
def _serialise_callable(fn):
    """Serialise a callable by its import path.

    Returns a ``{"module", "qualname"}`` dict, or ``None`` when *fn* is
    ``None``. Raises :class:`ValueError` when *fn* cannot be round-tripped
    via ``importlib`` (lambdas, nested / locally-defined functions, or
    callables whose qualname does not resolve back to the same object).

    A :class:`UserWarning` is emitted when *fn* lives in ``__main__`` —
    the save itself succeeds, but reloading in a different session
    requires an identically-named callable to be defined in that
    session's ``__main__`` too.
    """
    if fn is None:
        return None

    qn = getattr(fn, "__qualname__", None)
    mod = getattr(fn, "__module__", None)
    if qn is None or mod is None:
        raise ValueError(
            f"Cannot serialise callable {fn!r}: missing __module__ or "
            "__qualname__. Use a plain module-level function."
        )
    if "<lambda>" in qn:
        raise ValueError(
            f"Cannot serialise lambda {fn!r}: lambdas have no stable import "
            "path. Use a named module-level function instead, or save the "
            "scaler after removing pre_fns/post_fns."
        )
    if "<locals>" in qn:
        raise ValueError(
            f"Cannot serialise locally-defined callable {fn!r} (qualname "
            f"{qn!r}): nested / closure-defined functions cannot be "
            "round-tripped by qualname. Move the function to module scope."
        )

    try:
        module = importlib.import_module(mod)
    except ImportError as exc:
        raise ValueError(
            f"Cannot serialise callable {fn!r}: module {mod!r} is not "
            f"importable ({exc})."
        ) from exc

    resolved = module
    for part in qn.split("."):
        try:
            resolved = getattr(resolved, part)
        except AttributeError as exc:
            raise ValueError(
                f"Cannot serialise callable {fn!r}: qualname {qn!r} does "
                f"not resolve under module {mod!r} ({exc})."
            ) from exc

    if resolved is not fn:
        raise ValueError(
            f"Cannot serialise callable {fn!r}: qualname {mod}.{qn} "
            "resolves to a different object. This happens when a function "
            "has been monkey-patched, redefined, or wrapped after import."
        )

    if mod == "__main__":
        warnings.warn(
            f"Serialising callable {qn!r} from __main__. The save will "
            "succeed, but loading in a different session will only work "
            f"if an identically-named callable {qn!r} is defined in that "
            "session's __main__ (e.g. the same script is re-run). For "
            "portable loading, move the function to an importable module.",
            UserWarning,
            stacklevel=3,
        )

    return {"module": mod, "qualname": qn}


def _serialise_fn_pair(fns):
    """Serialise a ``(forward, inverse)`` tuple. Returns a list or ``None``."""
    if fns is None:
        return None
    return [_serialise_callable(fns[0]), _serialise_callable(fns[1])]


def _deserialise_callable(entry):
    """Reverse of :func:`_serialise_callable`. ``None`` passes through."""
    if entry is None:
        return None
    module = importlib.import_module(entry["module"])
    obj = module
    for part in entry["qualname"].split("."):
        obj = getattr(obj, part)
    return obj


def _deserialise_fn_pair(entry):
    """Reverse of :func:`_serialise_fn_pair`. Returns a tuple or ``None``."""
    if entry is None:
        return None
    return (_deserialise_callable(entry[0]), _deserialise_callable(entry[1]))


# ---------------------------------------------------------------------------
# Save — preprocessing objects (DataScaler)
# ---------------------------------------------------------------------------
def _save_scaler(scaler, path) -> None:
    """Serialise a fitted preprocessing object to a ``.cpx`` file.

    Currently only :class:`~copulax.preprocessing.DataScaler` is
    supported. The file contains a ``metadata.json`` describing the
    static configuration (method, quantile bounds, mode flags, function
    pair qualnames) plus ``arrays/offset.npy`` and ``arrays/scale.npy``
    holding the fitted parameters.

    Args:
        scaler: A fitted :class:`DataScaler` instance.
        path: Destination file path. The ``.cpx`` extension is appended
            automatically when missing.

    Raises:
        ValueError: If the scaler has not been fitted, or if any
            ``pre_fns`` / ``post_fns`` callable cannot be serialised by
            qualname (lambdas, closures, etc.).
    """
    if not scaler.is_fitted:
        raise ValueError(
            "Cannot save an unfitted DataScaler (offset/scale are None). "
            "Call .fit(x) first."
        )

    path = Path(path)
    if path.suffix != ".cpx":
        path = path.with_suffix(path.suffix + ".cpx")

    offset_arr = np.asarray(scaler.offset)
    scale_arr = np.asarray(scaler.scale)

    metadata = {
        "dist_family": "preprocessing",
        "scaler_class": type(scaler).__name__,
        "method": scaler.method,
        "q_low": scaler.q_low,
        "q_high": scaler.q_high,
        "offset_only": scaler.offset_only,
        "scale_only": scaler.scale_only,
        "pre_fns": _serialise_fn_pair(scaler.pre_fns),
        "post_fns": _serialise_fn_pair(scaler.post_fns),
        "arrays": {
            "offset": {"shape": list(offset_arr.shape), "dtype": str(offset_arr.dtype)},
            "scale": {"shape": list(scale_arr.shape), "dtype": str(scale_arr.dtype)},
        },
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        for name, arr in (("offset", offset_arr), ("scale", scale_arr)):
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

        if dist_family in ("univariate", "multivariate"):
            dist_class = metadata["dist_class"]
            dist_name = name if name is not None else metadata["dist_name"]
            template = _get_singleton(dist_class)
            params = {
                key: _read_array(key) for key in metadata["params"]
            }
            return template._fitted_instance(params, name=dist_name)

        elif dist_family == "copula":
            dist_class = metadata["dist_class"]
            dist_name = name if name is not None else metadata["dist_name"]
            template = _get_singleton(dist_class)
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

        elif dist_family == "preprocessing":
            # Local import to avoid a circular dependency at module import
            # time (preprocessing imports nothing from _serialization, but
            # _serialization must reference the class by name).
            from copulax.preprocessing import DataScaler

            scaler_class = metadata.get("scaler_class", "DataScaler")
            if scaler_class != "DataScaler":
                raise ValueError(
                    f"Unknown preprocessing scaler class: {scaler_class!r}."
                )

            offset = _read_array("offset")
            scale = _read_array("scale")
            return DataScaler(
                method=metadata["method"],
                q_low=metadata["q_low"],
                q_high=metadata["q_high"],
                offset_only=metadata["offset_only"],
                scale_only=metadata["scale_only"],
                pre_fns=_deserialise_fn_pair(metadata.get("pre_fns")),
                post_fns=_deserialise_fn_pair(metadata.get("post_fns")),
                offset=offset,
                scale=scale,
            )

        else:
            raise ValueError(
                f"Unknown dist_family in metadata: {dist_family!r}"
            )
