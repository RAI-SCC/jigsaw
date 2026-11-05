"""Sphinx configuration for the jigsaw documentation."""

from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path

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
    # Members inherited from jigsaw's private bases (e.g. the conv and matmul
    # bases), but not those of torch.nn.Module or of builtins the public classes
    # extend (ProcessMesh is a dict, the enums are StrEnums).
    "inherited-members": "Module,dict,str,Enum",
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# GPU/scientific-stack dependencies that aren't installed in the docs build
# environment. autodoc only needs to import jigsaw's modules far enough to
# read signatures and docstrings, so these are mocked instead of installed.
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
# annotations, which Python evaluates at import time via ``__or__``: a
# method mock instances don't have, so import crashes without this patch.
# Real, unmocked modules are unaffected.
from sphinx.ext.autodoc.mock import _MockObject
from sphinx.util.inspect import stringify_signature


def _mock_or(self: _MockObject, other: object) -> _MockObject:
    return self


_MockObject.__or__ = _mock_or
_MockObject.__ror__ = _mock_or

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Markdown -----------------------------------------------------------------
# Anchors for headings down to ``####``, so in-page links like
# ``#replicating-your-own-layers`` (ddp.md) resolve.
myst_heading_anchors = 4

# The Markdown pages, and README.md (included by index.md), link to repo files
# that are not doc pages (examples/, containers/, scripts/, ...). Those links work on
# GitHub, so the Markdown keeps them; for the docs build they are pointed at the
# file on GitHub instead. Links into docs/ are doc pages and are left alone.
_REPO = Path(__file__).resolve().parent.parent
_GITHUB_BLOB = "https://github.com/RAI-SCC/jigsaw/blob/main/"
_LOCAL_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")


def _repo_links_to_github(text: str, base: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        target, _, anchor = match.group(1).partition("#")
        path = (base / target).resolve()
        if not path.exists() or not path.is_relative_to(_REPO):
            return match.group(0)
        rel = path.relative_to(_REPO)
        if rel.parts[0] == "docs":
            return match.group(0)
        return f"]({_GITHUB_BLOB}{rel.as_posix()}{'#' + anchor if anchor else ''})"

    return _LOCAL_LINK.sub(replace, text)


def _on_source_read(app: object, docname: str, source: list[str]) -> None:
    doc = Path(app.env.doc2path(docname))  # type: ignore[attr-defined]
    source[0] = _repo_links_to_github(source[0], doc.parent)


def _on_include_read(
    app: object, relative_path: Path, parent_docname: str, content: list[str]
) -> None:
    included = Path(app.srcdir) / relative_path  # type: ignore[attr-defined]
    content[0] = _repo_links_to_github(content[0], included.resolve().parent)


# -- Constructor docs of private bases -----------------------------------------
# Public classes that only pin a detail of a private base (DistributedConv1d/2d/3d
# set the spatial rank) leave the constructor to the base's class docstring. The
# subclass page shows the base's sections after the subclass's own summary.
def _inherit_base_sections(
    app: object,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    if what != "class" or "Parameters" in lines:
        return
    for base in getattr(obj, "__mro__", ())[1:]:
        if not (base.__name__.startswith("_") and base.__module__.startswith("jigsaw")):
            continue
        doc = inspect.cleandoc(base.__doc__ or "").splitlines()
        if "Parameters" in doc:
            lines += ["", *doc[doc.index("Parameters") :]]
            return


# -- Constructor signatures -----------------------------------------------------
# With torch mocked, torch.nn.Module is a mock class whose __new__(*args, **kwargs)
# autodoc finds before the subclass's __init__, so every Module subclass would show
# ``(*args, **kwargs)``. Such classes get their jigsaw __init__ signature instead,
# or ``()`` if only the mocked base defines one; the types are left out because
# the description lists them.
def _init_signature(
    app: object,
    what: str,
    name: str,
    obj: object,
    options: object,
    signature: str | None,
    return_annotation: str,
) -> tuple[str, str] | None:
    if what != "class" or signature != "(*args, **kwargs)":
        return None
    if not (isinstance(obj, type) and issubclass(obj, _MockObject)):
        return None
    init = getattr(obj, "__init__", None)
    if not (inspect.isfunction(init) and init.__module__.startswith("jigsaw")):
        return "()", return_annotation
    sig = inspect.signature(init)
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])  # drop self
    text = stringify_signature(sig, show_annotation=False, show_return_annotation=False)
    return text, return_annotation


def setup(app: object) -> None:
    """Register the repo-link rewrite, base-section inheritance and signatures."""
    app.connect("source-read", _on_source_read)  # type: ignore[attr-defined]
    app.connect("include-read", _on_include_read)  # type: ignore[attr-defined]
    # Before napoleon (priority 500) converts the numpydoc sections.
    app.connect(  # type: ignore[attr-defined]
        "autodoc-process-docstring", _inherit_base_sections, priority=400
    )
    # After autodoc's own handlers (typehints recording, priority 500): the event
    # stops at the first handler that returns a result.
    app.connect(  # type: ignore[attr-defined]
        "autodoc-process-signature", _init_signature, priority=600
    )

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
