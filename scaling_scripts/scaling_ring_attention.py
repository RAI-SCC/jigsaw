import argparse

import numpy as np
import torch
import torch.distributed as dist
import yaml

from jigsaw import utils
from jigsaw import attention_ring


def make_process_mesh(
    device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
):
    """Return process mesh with rows, columns, and model group."""
    assert len(mesh_shape) == len(mesh_dim_names)

    world_size = dist.get_world_size(group=None)
    if -1 in mesh_shape:
        index = mesh_shape.index(-1)
        mesh_shape[index] = world_size // -torch.prod(torch.tensor(mesh_shape)).item()

    row_col_mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )
    if len(mesh_shape) == 2:
        model_mesh = dist.device_mesh.init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple([mesh_shape[-1] * mesh_shape[-2]]),
            mesh_dim_names=tuple(["model"]),
        )
    else:  #
        print(f"mesh shape! {mesh_shape[:-2]}")
        model_mesh = dist.device_mesh.init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple(
                [math.prod(mesh_shape[:-2]), mesh_shape[-2] * mesh_shape[-1]]
            ),
            mesh_dim_names=tuple(["x", "model"]),
        )

    mesh = {
        "row": row_col_mesh.get_group("row"),
        "col": row_col_mesh.get_group("col"),
        "model": model_mesh.get_group("model"),
    }

    return mesh


def training_loop(
    device: torch.device, localid: int, rank: int
) -> None:
    """Training loop for scaling experiments."""
    mesh_dict = make_process_mesh("nccl")
    
    x_features = 256
    output_dim = embed_dim = 256
    B, heads, N, d = 1, 8, mesh_dict["row"].size() * 256*64, 256
    Nlocal = N // mesh_dict["row"].size()

    

    model = attention_ring.RingAttention(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=output_dim,
        heads=heads,
        dropout=0.0,
        local_rank=rank,
        device=device,
        parallelism=2,
        model_group=mesh_dict["model"],
        spatial_group=mesh_dict["row"],
        channel_group=mesh_dict["col"],
        bayesian=False,
        dtype=torch.float32,
    )
     
    
    epoch_times = np.zeros(11)
    torch.cuda.synchronize()

    for epoch in range(11):
        model.train()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(model, (B, Nlocal, d//2), d, device)
        end.record()
        torch.cuda.synchronize()

        if rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")

    if rank == 0:
        print(f"Average epoch time is {np.average(epoch_times[1:])}")
        print(f"Std epoch time is {np.std(epoch_times[1:])}")


def train_one_epoch(
    model: torch.nn.Module, 
    X_shape,
    d,
    device
):
    """Time and train one epoch."""
    
    for _ in range(100):
        # Loading data: Always with async ops and handles
        X = torch.randn(
            *X_shape, device=device, requires_grad=True, dtype=torch.float32
        ) / (d**0.5)
        
        # output shape [n_batch, 720, 720, 36]
        output = model(X, None)

        # Reshape output
        
        output.sum().backward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    device, localid, rank, world_size = utils.init_cuda()

    training_loop(device, localid, rank)
    dist.destroy_process_group()
