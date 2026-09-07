"""Sphinx configuration for the jigsaw documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "jigsaw"
copyright = "2026, Deifilia Kieckhefen"
author = "Deifilia Kieckhefen"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------
# Recursively generate stub pages for jigsaw and every submodule (see api.rst).
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# GPU/scientific-stack dependencies that aren't installed in the docs build
# environment. autodoc only needs to import jigsaw's modules far enough to
# read signatures and docstrings, so these are mocked instead of installed -
# see containers/README.md for why torch/transformer_engine are kept out of
# lightweight environments.
autodoc_mock_imports = [
    "torch",
    "einops",
    "transformer_engine",
    "transformer_engine_torch",
    "mpi4py",
    "zarr",
    "timm",
    "xarray",
    "matplotlib",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# ``autodoc_mock_imports`` resolves attribute access (e.g. ``torch.Tensor``)
# to an *instance* of a dynamically generated mock class, not the class
# itself (see sphinx.ext.autodoc.mock._MockObject.__getattr__). jigsaw uses
# PEP 604 unions like ``torch.Tensor | None`` in live (non-deferred) type
# annotations, which Python evaluates at import time via ``__or__`` - a
# method mock instances don't have, so import crashes without this patch.
# Real, unmocked modules are unaffected.
from sphinx.ext.autodoc.mock import _MockObject


def _mock_or(self: _MockObject, other: object) -> _MockObject:
    return self


_MockObject.__or__ = _mock_or
_MockObject.__ror__ = _mock_or

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
