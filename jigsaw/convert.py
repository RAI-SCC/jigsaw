import torch

from jigsaw import attention, convolutions, linear, roll

mapped_modules = {
    torch.nn.Linear: linear.DistributedXWT,
    torch.nn.Conv2d: convolutions.DistributedConv2D,
    torch.nn.ConvTranspose2d: convolutions.DistributedConvTranspose2D,
    roll.Roll: roll.DistributedRoll
}

container_converters = {
    attention.Attention: attention.Attention.convert_attention_to_jigsaw
}


def convert_to_jigsaw(model, process_mesh, device, **kwargs):
    # 1. Container match: hand off entirely, do NOT recurse afterward
    container_fn = container_converters.get(type(model))
    if container_fn is not None:
        return container_fn(model, process_mesh, device, **kwargs)

    # 2. Leaf match: swap with Distributed equivalent
    replacement_cls = mapped_modules.get(type(model))
    if replacement_cls is not None:
        return replacement_cls.convert_to_jigsaw(
            model, process_mesh=process_mesh, device=device
        )

    # 3. Generic container: recurse into children
    for name, child in model.named_children():
        new_child = convert_to_jigsaw(child, process_mesh, device, **kwargs)
        if new_child is not child:
            setattr(model, name, new_child)
    return model
