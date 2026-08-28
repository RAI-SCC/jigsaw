"""Convert to jigsaw module."""

from typing import Any, Callable

import torch

from jigsaw.modules import (
    attention,
    conv,
    conv1d,
    conv_transpose,
    conv_transpose_1d,
    conv_transpose_3d,
    linear,
    roll,
)

mapped_modules: dict[type[torch.nn.Module], Any] = {
    torch.nn.Linear: linear.DistributedXWT,
    torch.nn.Conv1d: conv1d.DistributedConv1D,
    torch.nn.Conv2d: conv.DistributedConv2D,
    torch.nn.ConvTranspose1d: conv_transpose_1d.DistributedConvTranspose1D,
    torch.nn.ConvTranspose2d: conv_transpose.DistributedConvTranspose2D,
    torch.nn.ConvTranspose3d: conv_transpose_3d.DistributedConvTranspose3D,
    roll.Roll: roll.DistributedRoll,
}
container_converters: dict[type[torch.nn.Module], Callable[..., torch.nn.Module]] = {
    attention.Attention: attention.Attention.convert_to_jigsaw
}


def convert_to_jigsaw(
    model: torch.nn.Module,
    process_mesh_dict: dict,
    device: torch.device,
    row_kw: str = "row",
    col_kw: str = "col",
    model_kw: str = "model",
    **kwargs: bool,
) -> torch.nn.Module | Any:
    """
    Convert pytorch model into jigsaw distributed model.

    Note that this function changes the model in-place. Make a copy of the
    existing model if you want to retain it.

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model.
    process_mesh_dict: dict
        Process mesh dict that comtains row, column, and model process groups.
    device: torch.device
        Device that this model is on.
    row_kw: str
        Key for process mesh dict row the model rows.
    col_kw: str
        Key for process mesh dict row the model cols.
    model_kw: str
        Key for process mesh dict row the model (rows x columns).
    **kwargs:
        Additional keyword arguments passed to :func: `convert_to_jigsaw`

    Return
    ------
    Distributed model.
    """
    # 1. Container match: hand off entirely, do NOT recurse afterward
    container_fn = container_converters.get(type(model))
    if container_fn is not None:
        return container_fn(
            model,
            process_mesh_dict,
            device,
            row_kw=row_kw,
            col_kw=col_kw,
            model_kw=model_kw,
            **kwargs,
        )

    # 2. Leaf match: swap with Distributed equivalent
    replacement_cls = mapped_modules.get(type(model))
    if replacement_cls is not None:
        return replacement_cls.convert_to_jigsaw(
            model,
            process_mesh_dict=process_mesh_dict,
            device=device,
            row_kw=row_kw,
            col_kw=col_kw,
            model_kw=model_kw,
            **kwargs,
        )

    # 3. Generic container: recurse into children
    for name, child in model.named_children():
        new_child = convert_to_jigsaw(
            child,
            process_mesh_dict,
            device,
            row_kw=row_kw,
            col_kw=col_kw,
            model_kw=model_kw,
            **kwargs,
        )
        if new_child is not child:
            setattr(model, name, new_child)
    return model
