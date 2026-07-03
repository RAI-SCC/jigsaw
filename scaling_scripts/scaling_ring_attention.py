import argparse
import itertools
import math
import sys

import numpy as np
import torch
import torch.distributed as dist

from jigsaw import  utils
from jigsaw.modules.attention import config

_default_layer_config: dict[str, int] = {
    "x_features": 256,
    "batch_size": 1,
    "heads": 8,
    "n_global": 512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale ring attention experiments with flexible configuration"
    )

    # Experiment control arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the baseline and compare the outputs.",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the baseline and compare the runtime.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training steps per epoch (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )

    # Same for the input and model configuration
    parser.add_argument(
        "--layer-config",
        nargs="*",
        default=[],
        help="Layer configuration overrides in the format key=value (e.g. --layer-config batch_size=32 n_global=1024)",
    )

    args = parser.parse_args()
    return args


def make_process_mesh(
    device_type="nccl", mesh_shape: list[int]=[2, 2], mesh_dim_names: list[str]=["row", "col"]
) -> dict:
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
    mesh_dict: dict[str, dist.ProcessGroup],
    device: torch.device,
    local_rank: int,
    ring_config: functional.RingAttentionConfig,
    layer_config: dict = _default_layer_config,
    steps: int = 10,
    epochs: int = 3,
) -> None:
    """Training loop for scaling experiments."""
    torch.manual_seed(2)

    # define configs for the ring attention
    torch.manual_seed(local_rank)

    # define stripped down attention calculation
    ring_attention = attention_ring.Ring(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=embed_dim,
        heads=heads,
        dropout_p=0.0,
        local_rank=local_rank,
        device=device,
        parallelism=mesh_dict["model"].size(),
        spatial_group=mesh_dict["model"],
        bayesian=False,
        dtype=torch.bfloat16,
        ring_config=ring_config,
    )

    n_epochs = epochs
    epoch_times = np.zeros(n_epochs)
    torch.cuda.synchronize()

    # parallel experiment
    for epoch in range(n_epochs):
        ring_attention.train()

        x = torch.randn(
            (batch_size, n_global // mesh_dict["model"].size(), x_features),
            device=device,
            requires_grad=True,
            dtype=torch.bfloat16,
        ) / (x_features**0.5)
        qkv = torch.matmul(x, w_qkv.mT)
        q, k, v = [
            t.detach().requires_grad_(True)
            for t in qkv.reshape(
                batch_size, n_global // mesh_dict["model"].size(), 3, heads, embed_dim
            ).permute(2, 0, 3, 1, 4)
        ]

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
            steps=steps,
        )
        end.record()
        torch.cuda.synchronize()

        epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60

    return epoch_times


def train_one_epoch(model: torch.nn.Module, x, q, k, v, w_qkv, steps: int = 10):
    # X has shape [B, Nlocal, d//2]
    for _ in range(steps):
        output = model(x.detach(), q, k, v, w_qkv, None)
        output.sum().backward()


def ref_training_loop(mesh_dict, layer_config, n_epochs, n_steps=10):
    epoch_times = np.zeros(n_epochs)
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

    # non-parallel baseline
    for epoch in range(n_epochs):
        model_baseline.train()
        x = torch.randn(
            (batch_size, n_global, x_features),
            device=device,
            requires_grad=True,
            dtype=torch.bfloat16,
        ) / (x_features**0.5)
        qkv = torch.matmul(x, w_qkv.mT)
        q, k, v = [
            t.detach().requires_grad_(True)
            for t in qkv.reshape(batch_size, n_global, 3, heads, embed_dim).permute(
                2, 0, 3, 1, 4
            )
        ]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(model_baseline, x, q, k, v, w_qkv, steps=n_steps)
        end.record()
        torch.cuda.synchronize()

        epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60

        if local_rank == 0:
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")
    return epoch_times


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    device, localid, local_rank, world_size = utils.init_cuda()
    mesh_dict = make_process_mesh("nccl")

    # Process layer configuration overrides in the same way
    layer_config = _default_layer_config.copy()
    for override in args.layer_config:
        try:
            key, value = override.split("=")
            if key in layer_config:
                if value.isdigit():
                    value = int(value)
                layer_config[key] = value
            else:
                print(
                    f"WARNING: Unknown layer config key '{key}' in override '{override}'",
                    file=sys.stderr,
                )
        except ValueError:
            print(
                f"ERROR: Invalid layer config override format: '{override}'. Expected format is key=value.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Extract experiment control parameters
    epochs = args.epochs
    steps = args.steps

    if local_rank == 0:
        print(f"\nExperiment settings: epochs={epochs}, steps={steps}")
        print(f"Layer Configuration: {layer_config}\n")

    # Initialize the shared weight matrix for QKV projections
    x_features = layer_config.get("x_features", _default_layer_config["x_features"])
    batch_size = layer_config.get("batch_size", _default_layer_config["batch_size"])
    heads = layer_config.get("heads", _default_layer_config["heads"])
    n_global = layer_config.get("n_global", _default_layer_config["n_global"])
    embed_dim = layer_config.get("embed_dim", x_features // heads)

    w_qkv = torch.randn(
        3 * embed_dim * heads,
        x_features,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    ## Run the reference training loop to get baseline epoch times
    ref_epoch_time = ref_training_loop(
        mesh_dict,
        layer_config,
        epochs,
        steps,
    )

    # Set all possible configuration options here
    enum_fields = {
        "send_in_forward": config.SendForward,
        "save_for_backward": config.SaveBackward,
        # "send_forward_op": ring_attn_fns.SendForwardOp,
        "save_scope": config.SaveScope,
        "send_backward": config.SendBackward,
        "offload": config.Offload,
    }

    ring_config = config.RingAttentionConfig(
        device=device,
        send_in_forward=config.SendForward.X,
        save_for_backward=config.SaveBackward.X,
        send_forward_op=config.SendForwardOp.SENDRECV,
        save_scope=config.SaveScope.GLOBAL,
        send_backward=config.SendBackward.GATHER_DKV,
        offload=configs.Offload.DISABLED,
    )

    for combo in itertools.product(*[e for e in enum_fields.values()]):
        ring_config = config.RingAttentionConfig(
            **dict(zip(enum_fields.keys(), combo)),
            device=device,
        )

        if local_rank == 0:
            print(f"Testing: {ring_config}")
        epoch_times = training_loop(
            mesh_dict,
            device,
            local_rank,
            ring_config=ring_config,
            layer_config=layer_config,
            epochs=epochs,
            steps=steps,
        )

        if local_rank == 0:
            parallel_time = np.average(epoch_times[1:])
            # print(f"Average parallel epoch time is {parallel_time}")
            ref_time = np.average(ref_epoch_time[1:])
            # print(f"Average ref time is {ref_time}")
            print(f"Config: {ring_config} - Speedup {ref_time / parallel_time}")

    dist.destroy_process_group()
