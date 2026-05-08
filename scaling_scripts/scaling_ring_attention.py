import argparse
import math

import numpy as np
import torch
import torch.distributed as dist

from jigsaw import attention_ring, utils


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


def training_loop(device: torch.device, local_rank: int) -> None:
    """Training loop for scaling experiments."""
    mesh_dict = make_process_mesh("nccl")

    x_features = 256
    output_dim = embed_dim = 256
    b, heads, n_global, d = 1, 32, 131072, 256

    torch.manual_seed(2)

    # pass in the QKV mapping alone
    w_qkv = torch.randn(
        3 * d * heads, d, device=device, requires_grad=True, dtype=torch.bfloat16
    )

    # randomize input data per rank
    torch.manual_seed(local_rank)
    model = attention_ring.Ring(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=output_dim,
        heads=heads,
        dropout=0.0,
        local_rank=local_rank,
        device=device,
        parallelism=mesh_dict["model"].size(),
        spatial_group=mesh_dict["model"],
        bayesian=False,
        dtype=torch.bfloat16,
    )
    model_baseline = attention_ring.RingBaseline(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=output_dim,
        heads=heads,
        dropout=0.0,
        local_rank=local_rank,
        device=device,
        parallelism=mesh_dict["model"].size(),
        spatial_group=mesh_dict["model"],
        bayesian=False,
        dtype=torch.bfloat16,
    )

    n_epochs = 4
    epoch_times = np.zeros(n_epochs)
    torch.cuda.synchronize()

    for epoch in range(n_epochs):
        model.train()

        x = torch.randn(
            (b, n_global // mesh_dict["model"].size(), d),
            device=device,
            requires_grad=True,
            dtype=torch.bfloat16,
        ) / (d**0.5)
        qkv = torch.matmul(x, w_qkv.mT)
        q, k, v = [
            t.detach().requires_grad_(True)
            for t in qkv.reshape(
                b, n_global // mesh_dict["model"].size(), 3, heads, d
            ).permute(2, 0, 3, 1, 4)
        ]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(
            model,
            x,
            q,
            k,
            v,
            w_qkv,
            b,
            n_global // mesh_dict["model"].size(),
            heads,
            d,
            device,
        )
        end.record()
        torch.cuda.synchronize()

        if local_rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")

    if local_rank == 0:
        print(f"Average speedup epoch time is {np.average(epoch_times[1:])}")
        print(f"Std epoch time is {np.std(epoch_times[1:])}")

    for epoch in range(n_epochs):
        model_baseline.train()
        x = torch.randn(
            (b, n_global, d), device=device, requires_grad=True, dtype=torch.bfloat16
        ) / (d**0.5)
        qkv = torch.matmul(x, w_qkv.mT)
        q, k, v = [
            t.detach().requires_grad_(True)
            for t in qkv.reshape(b, n_global, 3, heads, d).permute(2, 0, 3, 1, 4)
        ]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(
            model_baseline, x, q, k, v, w_qkv, b, n_global, heads, d, device
        )
        end.record()
        torch.cuda.synchronize()

        if local_rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")
    if local_rank == 0:
        print(f"Average epoch time is {np.average(epoch_times[1:])}")
        print(f"Std epoch time is {np.std(epoch_times[1:])}")


def train_one_epoch(model: torch.nn.Module, x, q, k, v, w_qkv):
    """Time and train one epoch."""
    # X has shape [B, Nlocal, d//2]
    for _ in range(10):
        output = model(x.detach(), q, k, v, w_qkv, None)
        output.sum().backward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    device, localid, local_rank, world_size = utils.init_cuda()

    training_loop(device, local_rank)
    dist.destroy_process_group()
