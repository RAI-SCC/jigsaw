import torch
import torch.distributed as dist
from jigsaw import linear, convolutions

mapped_modules = {
    torch.nn.Linear: linear.DistributedXWT,
    torch.nn.Conv2d: convolutions.DistributedConv2D,
    torch.nn.ConvTranspose2d: convolutions.DistributedConvTranspose2D
}

_blacklist = {
    torch.nn.Identity,
    # in-place activations
    torch.nn.Threshold,
    torch.nn.ReLU,
    torch.nn.RReLU,
    torch.nn.Hardtanh,
    torch.nn.ReLU6,
    torch.nn.Sigmoid,
    torch.nn.Hardsigmoid,
    torch.nn.Tanh,
    torch.nn.SiLU,
    torch.nn.Mish,
    torch.nn.Hardswish,
    torch.nn.ELU,
    torch.nn.CELU,
    torch.nn.SELU,
    torch.nn.GLU,
    torch.nn.GELU,
    torch.nn.LeakyReLU,
}

def convert_to_jigsaw(
    model: torch.nn.Module,
    synchronize: bool,
    process_mesh,
    device
) -> torch.nn.Module:
    # Top-level replacement
    replacement_cls = mapped_modules.get(type(model))
    if replacement_cls is not None:
        return replacement_cls.convert_to_jigsaw(model, process_mesh=process_mesh, device=device)

    # Otherwise descend into children and replace in-place
    for name, child in model.named_children():
        new_child = convert_to_jigsaw(child, synchronize, process_mesh, device)
        if new_child is not child:
            setattr(model, name, new_child)
    return model