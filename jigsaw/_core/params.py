"""Replacement of a layer's parameters by ``reshard_parameters``."""

import torch


def replace_parameter(module: torch.nn.Module, name: str, data: torch.Tensor) -> None:
    """Set ``module.<name>`` to a new parameter holding ``data``.

    The new parameter keeps the ``requires_grad`` of the one it replaces, so a
    frozen parameter stays frozen. It is a new object: an optimizer or hook
    holding the old parameter does not see it.

    Parameters
    ----------
    module : torch.nn.Module
        Module owning the parameter.
    name : str
        Attribute name of the parameter.
    data : torch.Tensor
        Values of the new parameter.
    """
    old = getattr(module, name)
    setattr(module, name, torch.nn.Parameter(data, requires_grad=old.requires_grad))
