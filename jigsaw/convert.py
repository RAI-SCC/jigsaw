import torch
import torch.distributed as dist
from jigsaw import linear, convolutions, attention

mapped_modules = {
    torch.nn.Linear: linear.DistributedXWT,
    torch.nn.Conv2d: convolutions.DistributedConv2D,
    torch.nn.ConvTranspose2d: convolutions.DistributedConvTranspose2D
}

container_converters = {
    attention.Transformer: attention.Transformer.convert_transformer_to_jigsaw
}


def convert_to_jigsaw(model, synchronize, process_mesh, device):
    # 1. Container match: hand off entirely, do NOT recurse afterward
    container_fn = container_converters.get(type(model))
    if container_fn is not None:
        return container_fn(model, synchronize, process_mesh, device)

    # 2. Leaf match: swap with Distributed equivalent
    replacement_cls = mapped_modules.get(type(model))
    if replacement_cls is not None:
        return replacement_cls.convert_to_jigsaw(model, process_mesh=process_mesh, device=device)

    # 3. Generic container: recurse into children
    for name, child in model.named_children():
        new_child = convert_to_jigsaw(child, synchronize, process_mesh, device)
        if new_child is not child:
            setattr(model, name, new_child)
    return model