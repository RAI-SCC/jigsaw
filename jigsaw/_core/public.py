"""Label re-exported names with the public module they are imported from."""

import types
from collections.abc import Iterable, Mapping


def set_module(namespace: Mapping[str, object], names: Iterable[str]) -> None:
    """Set ``__module__`` of the jigsaw classes and functions in ``names`` to ``namespace``'s module.

    A public module that re-exports names from internal modules calls
    ``set_module(globals(), __all__)`` after its imports. Reprs, error
    messages, the docs and pickles then name the public path
    (``jigsaw.modules.linear.DistributedLinear``) instead of the internal file
    (``jigsaw.modules.linear._linear``), so the internal layout can change
    without breaking them. A name exported from several modules keeps the
    label of the last one to run.

    Parameters
    ----------
    namespace : Mapping[str, object]
        The calling module's ``globals()``.
    names : Iterable[str]
        Names to relabel, usually the module's ``__all__``. Submodules and
        objects defined outside jigsaw are left alone.
    """
    module = namespace["__name__"]
    assert isinstance(module, str)
    for name in names:
        obj = namespace[name]
        if isinstance(obj, types.ModuleType):
            continue
        if getattr(obj, "__module__", "").startswith("jigsaw."):
            obj.__module__ = module
