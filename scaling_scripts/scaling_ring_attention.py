import argparse
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import nvtx

from jigsaw import attention_ring, utils
from jigsaw import ring_attention_helpers as ring_attn_fns


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

    x_features = 4096
    batch_size = 1
    heads = 16
    n_global = 512
    embed_dim = x_features // heads

    torch.manual_seed(2)

    # pass in the QKV mapping alone
    w_qkv = torch.randn(
        3 * embed_dim * heads,
        x_features,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )

    # define configs for the ring attention
    torch.manual_seed(local_rank)
    ring_config = ring_attn_fns.RingAttentionConfig(
        send_in_forward=ring_attn_fns.SendForward.KV,
        save_for_backward=ring_attn_fns.SaveBackward.X,
        save_scope=ring_attn_fns.SaveScope.GLOBAL,
        send_backward=ring_attn_fns.SendBackward.GATHER_DKV,
        offload=ring_attn_fns.Offload.DISABLED,
        device=device,
    )

    # define stripped down
    ring_attention = attention_ring.Ring(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=embed_dim,
        heads=heads,
        dropout=0.0,
        local_rank=local_rank,
        device=device,
        parallelism=mesh_dict["model"].size(),
        spatial_group=mesh_dict["model"],
        bayesian=False,
        dtype=torch.bfloat16,
        ring_config=ring_config,
    )

    # define baseline model
    model_baseline = attention_ring.RingBaseline(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=embed_dim,
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

    # parallel experiment
    for epoch in range(n_epochs):
        nvtx.range_push("Parallel Experiment")
        ring_attention.train()

        nvtx.range_push("Data Generation")
        x = torch.randn(
            (batch_size, n_global // mesh_dict["model"].size(), x_features),
            device=device,
            requires_grad=True,
            dtype=torch.bfloat16,
        ) / (x_features**0.5)
        nvtx.range_pop()
        nvtx.range_push("QKV")
        qkv = torch.matmul(x, w_qkv.mT)
        nvtx.range_pop()
        nvtx.range_push("Reshape and Permute")
        q, k, v = [
            t.detach().requires_grad_(True)
            for t in qkv.reshape(
                batch_size, n_global // mesh_dict["model"].size(), 3, heads, embed_dim
            ).permute(2, 0, 3, 1, 4)
        ]
        nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(
            ring_attention,
            x,
            q,
            k,
            v,
            w_qkv,
        )
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()

        if local_rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")

    if local_rank == 0:
        parallel_time = np.average(epoch_times[1:])
        print(f"Average parallel epoch time is {parallel_time}")

    # non-parallel baseline
    for epoch in range(n_epochs):
        nvtx.range_push("Baseline Experiment")
        model_baseline.train()
        nvtx.range_push("Data Generation")
        x = torch.randn(
            (batch_size, n_global, x_features),
            device=device,
            requires_grad=True,
            dtype=torch.bfloat16,
        ) / (x_features**0.5)
        nvtx.range_pop()
        nvtx.range_push("QKV")
        qkv = torch.matmul(x, w_qkv.mT)
        nvtx.range_pop()
        nvtx.range_push("Reshape and Permute")
        q, k, v = [
            t.detach().requires_grad_(True)
            for t in qkv.reshape(batch_size, n_global, 3, heads, embed_dim).permute(
                2, 0, 3, 1, 4
            )
        ]
        nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(model_baseline, x, q, k, v, w_qkv)
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()

        if local_rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")
    if local_rank == 0:
        ref_time = np.average(epoch_times[1:])
        print(f"Average ref time is {ref_time}")
        print(f"Speedup is {ref_time / parallel_time}")

    if local_rank == 0:
        ref_time = np.average(epoch_times[1:])
        print(f"Average ref time is {ref_time}")
        print(f"Speedup is {ref_time / parallel_time}")


def train_one_epoch(model: torch.nn.Module, x, q, k, v, w_qkv):
    """Time and train one epoch."""
    # X has shape [B, Nlocal, d//2]
    for _ in range(10):
        nvtx.range_push("Forward")
        output = model(x.detach(), q, k, v, w_qkv, None)
        nvtx.range_pop()
        nvtx.range_push("Backward")
        output.sum().backward()
        nvtx.range_pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    device, localid, local_rank, world_size = utils.init_cuda()

    training_loop(device, local_rank)
    dist.destroy_process_group()
