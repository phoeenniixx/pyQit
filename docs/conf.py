"""Sphinx configuration for the PyQit documentation."""

from importlib.metadata import version as _get_version

project = "PyQit"
author = "Aryan Saini"
copyright = "2026, Aryan Saini"
release = _get_version("pyqit")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints", "tutorials/ckpts"]

# Docstrings follow the numpy convention (see [tool.ruff.lint.pydocstyle]).
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Signatures come from __init__, whose parameters skbase stores verbatim.
autoclass_content = "class"

# The tutorials are executed in CI by build_tools/run_examples.sh with outputs
# committed, so rebuilding them here would only re-derive what is already in the
# notebooks -- and would drag torch/lightning into the docs environment.
nbsphinx_execute = "never"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# .ico rather than .svg: it carries a simplified 16px glyph alongside the
# full mark, because the ket bracket blurs into the Q below ~24px.
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "github_url": "https://github.com/phoeenniixx/pyQit",
    "logo": {
        "image_light": "_static/pyqit-logo.svg",
        "image_dark": "_static/pyqit-logo-dark.svg",
        "text": "PyQit",
    },
}
