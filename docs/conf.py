# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "CopulAX"
copyright = "2024-2026, Tyler Mitchell"
author = "Tyler Mitchell"
release = "3.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# `.readthedocs.yaml` installs the package via `pip install .[docs]`, which
# pulls in jax, equinox, optax, quadax, interpax, matplotlib transitively, so
# autodoc can import the real modules. Mocking them out (as we previously did)
# made `=-jnp.inf` default arguments in module-level signatures resolve to
# string-y mock objects, which sphinx-autodoc-typehints then tried to negate
# during signature rendering -- producing 76 spurious "bad operand type for
# unary -: 'inf'" warnings on every build.
autodoc_mock_imports = []

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
