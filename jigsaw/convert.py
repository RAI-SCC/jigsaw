"""Conversion of PyTorch models to jigsaw modules."""

from typing import Any, Callable

import torch

from jigsaw.modules import (
    attention,
    convolution,
    linear,
    normalization,
    roll,
)

mapped_modules: dict[type[torch.nn.Module], Any] = {
    torch.nn.Linear: linear.DistributedXWT,
    torch.nn.Conv1d: convolution.DistributedConv1D,
    torch.nn.Conv2d: convolution.DistributedConv2D,
    torch.nn.Conv3d: convolution.DistributedConv3D,
    torch.nn.ConvTranspose1d: convolution.DistributedConv1DTranspose,
    torch.nn.ConvTranspose2d: convolution.DistributedConv2DTranspose,
    torch.nn.ConvTranspose3d: convolution.DistributedConv3DTranspose,
    roll.Roll: roll.DistributedRoll,
    torch.nn.LayerNorm: normalization.DistributedLayerNorm,
    torch.nn.GroupNorm: normalization.DistributedGroupNorm,
    torch.nn.Softmax: normalization.DistributedSoftMax,
    linear.MatMulXWT: linear.DistributedMatMulXWT,
    linear.MatMulXW: linear.DistributedMatMulXW,
    linear.MatMulXTW: linear.DistributedMatMulXTW,
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
    convert_layer_norm: bool = True,
    **kwargs: bool,
) -> torch.nn.Module | Any:
    """Convert a PyTorch model into a jigsaw distributed model in place.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    process_mesh_dict : dict
        Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
    device : torch.device
        Device of the model.
    row_kw : str
        Key of the row process group.
    col_kw : str
        Key of the column process group.
    model_kw : str
        Key of the model (rows x columns) process group.
    convert_layer_norm : bool
        Whether ``torch.nn.LayerNorm`` is swapped for
        :class:`~jigsaw.modules.normalization.DistributedLayerNorm`. Leave
        this at the default unless the normalized dimension is *not* sharded
        at the point where the layer norm sits (e.g. it is never split across
        ``col_kw``) -- a plain ``nn.LayerNorm`` is only correct there, and
        skipping the distributed version avoids its extra all-reduces. With a
        sharded normalized dimension, turning this off silently computes
        statistics on the local shard only, which is wrong.
    **kwargs : Any
        Forwarded to the ``convert_to_jigsaw`` of each converted module.

    Returns
    -------
    torch.nn.Module
        The converted model.
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
    if isinstance(model, torch.nn.LayerNorm) and not convert_layer_norm:
        replacement_cls = None
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
            convert_layer_norm=convert_layer_norm,
            **kwargs,
        )
        if new_child is not child:
            setattr(model, name, new_child)
    return model
