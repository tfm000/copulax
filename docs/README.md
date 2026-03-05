# Documentation

Sphinx-based documentation for CopulAX, hosted on Read the Docs.

## Building Locally

```bash
# Install doc dependencies
pip install copulax[docs]

# Build HTML documentation
cd docs
make html

# On Windows
.\make.bat html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in a browser.

## Structure

| File / Directory       | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| `conf.py`              | Sphinx configuration (extensions, theme, mock imports) |
| `index.rst`            | Documentation landing page                             |
| `getting_started.rst`  | Installation and quick-start guide                     |
| `api/`                 | API reference (auto-generated from docstrings)         |
| `api/univariate.rst`   | Univariate distribution API                            |
| `api/multivariate.rst` | Multivariate distribution API                          |
| `api/copulas.rst`      | Copula distribution API                                |
| `api/special.rst`      | Special functions API                                  |
| `api/utilities.rst`    | Utility functions API                                  |

## Configuration

- **Theme**: `sphinx_rtd_theme` (Read the Docs theme)
- **Extensions**: autodoc, autosummary, napoleon, viewcode, intersphinx
- **Mock imports**: JAX, equinox, optax, and other dependencies are mocked for doc builds
- **Docstring style**: Google-style (parsed by napoleon)
