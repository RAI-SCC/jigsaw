"""Conversion of PyTorch models to jigsaw modules."""

import warnings
from typing import Any, Callable, Iterator, Literal

import torch

from jigsaw.modules import (
    activation,
    attention,
    convolution,
    linear,
    normalization,
    roll,
)

__all__ = ["convert_to_jigsaw"]

mapped_modules: dict[type[torch.nn.Module], Any] = {
    torch.nn.Linear: linear.DistributedLinear,
    torch.nn.Conv1d: convolution.DistributedConv1d,
    torch.nn.Conv2d: convolution.DistributedConv2d,
    torch.nn.Conv3d: convolution.DistributedConv3d,
    torch.nn.ConvTranspose1d: convolution.DistributedConvTranspose1d,
    torch.nn.ConvTranspose2d: convolution.DistributedConvTranspose2d,
    torch.nn.ConvTranspose3d: convolution.DistributedConvTranspose3d,
    roll.Roll: roll.DistributedRoll,
    torch.nn.LayerNorm: normalization.DistributedLayerNorm,
    torch.nn.GroupNorm: normalization.DistributedGroupNorm,
    torch.nn.Softmax: activation.DistributedSoftmax,
    linear.MatMulXWT: linear.DistributedMatMulXWT,
    linear.MatMulXW: linear.DistributedMatMulXW,
    linear.MatMulXTW: linear.DistributedMatMulXTW,
}
container_converters: dict[type[torch.nn.Module], Callable[..., torch.nn.Module]] = {
    attention.Attention: attention.Attention.convert_to_jigsaw
}
# Leaf modules that are correct on a sharded tensor as-is: elementwise, no
# parameters, no mixing along any dimension.
shard_safe_modules: tuple[type[torch.nn.Module], ...] = (
    torch.nn.Identity,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.LeakyReLU,
    torch.nn.ELU,
    torch.nn.SELU,
    torch.nn.CELU,
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.Mish,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
    torch.nn.Hardtanh,
    torch.nn.Hardsigmoid,
    torch.nn.Hardswish,
    torch.nn.Softplus,
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
)


def _unrecognized_modules(
    model: torch.nn.Module, prefix: str, convert_layer_norm: bool
) -> Iterator[tuple[str, str]]:
    """Yield ``(qualified name, reason)`` for modules the converter cannot handle.

    Walks the tree the way :func:`convert_to_jigsaw` does: converted containers
    and leaves are not descended into.
    """
    name = prefix or "<root>"
    if type(model) in container_converters or type(model) in mapped_modules:
        return
    if isinstance(model, torch.nn.LayerNorm) and not convert_layer_norm:
        return  # explicit opt-out, see ``convert_layer_norm``
    if type(model).__module__.startswith("jigsaw."):
        return  # already a jigsaw module
    children = list(model.named_children())
    if not children:
        if not isinstance(model, shard_safe_modules):
            yield name, f"unrecognized leaf module {type(model).__name__}"
        return
    if any(True for _ in model.parameters(recurse=False)):
        yield (
            name,
            (f"{type(model).__name__} owns parameters directly, which are not sharded"),
        )
    for child_name, child in children:
        yield from _unrecognized_modules(
            child,
            f"{prefix}.{child_name}" if prefix else child_name,
            convert_layer_norm,
        )


def convert_to_jigsaw(
    model: torch.nn.Module,
    process_mesh_dict: dict,
    device: torch.device,
    row_kw: str = "row",
    col_kw: str = "col",
    model_kw: str = "model",
    convert_layer_norm: bool = True,
    on_unrecognized: Literal["warn", "raise", "ignore"] = "warn",
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
        :class:`~jigsaw.modules.DistributedLayerNorm`. Leave
        this at the default unless the normalized dimension is *not* sharded
        at the point where the layer norm sits (e.g. it is never split across
        ``col_kw``) -- a plain ``nn.LayerNorm`` is only correct there, and
        skipping the distributed version avoids its extra all-reduces. With a
        sharded normalized dimension, turning this off silently computes
        statistics on the local shard only, which is wrong.
    on_unrecognized : {"warn", "raise", "ignore"}
        What to do about modules the converter leaves untouched: leaf modules
        that are neither converted nor in ``shard_safe_modules``, and container
        modules that own parameters directly. Such a module runs as plain torch
        on whatever local shard it receives, which is wrong if it reduces over
        or mixes along a sharded dimension. The check runs before anything is
        converted, so ``"raise"`` leaves ``model`` unchanged. Ops called
        functionally inside ``forward`` (e.g. ``torch.logsumexp``) are not
        visible to this check.
    **kwargs : Any
        Forwarded to the ``convert_to_jigsaw`` of each converted module.

    Returns
    -------
    torch.nn.Module
        The converted model.

    Raises
    ------
    ValueError
        If ``on_unrecognized`` is ``"raise"`` and an unrecognized module is found,
        or if ``on_unrecognized`` is not one of the accepted values.
    """
    if on_unrecognized not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"on_unrecognized must be 'warn', 'raise' or 'ignore', got {on_unrecognized!r}"
        )
    if on_unrecognized != "ignore":
        found = list(_unrecognized_modules(model, "", convert_layer_norm))
        if found:
            msg = (
                "convert_to_jigsaw leaves these modules as plain torch; they "
                "compute on the local shard only, which is wrong if they reduce "
                "over or mix along a sharded dimension:\n"
                + "\n".join(f"  {name}: {reason}" for name, reason in found)
            )
            if on_unrecognized == "raise":
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
    return _convert(
        model,
        process_mesh_dict,
        device,
        row_kw=row_kw,
        col_kw=col_kw,
        model_kw=model_kw,
        convert_layer_norm=convert_layer_norm,
        **kwargs,
    )


def _convert(
    model: torch.nn.Module,
    process_mesh_dict: dict,
    device: torch.device,
    row_kw: str,
    col_kw: str,
    model_kw: str,
    convert_layer_norm: bool,
    **kwargs: bool,
) -> torch.nn.Module | Any:
    """Recursive worker of :func:`convert_to_jigsaw`, without the module check."""
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
        new_child = _convert(
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
