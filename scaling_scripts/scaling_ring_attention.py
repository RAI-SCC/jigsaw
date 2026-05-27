import argparse
import math
import os
import sys

import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import nvtx

from jigsaw import attention_ring, utils
from jigsaw import ring_attention_helpers as ring_attn_fns

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale ring attention experiments with flexible configuration"
    )
    
    # Config file
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML configuration file for ring attention settings"
    )
    # Experiment control arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the baseline and compare the outputs."
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the baseline and compare the runtime."
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps before timing (default: 0)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training steps per epoch (default: 5)"
    )
    
    # Additional arguments for the ring configuration. It should be a list of strings with the format --ring-config key=value. These will override both the defaults and the YAML config.
    parser.add_argument(
        "--ring-config",
        nargs='*',
        default=[],
        help="Ring attention configuration overrides in the format key=value (e.g. --ring-config send_in_forward=KV save_for_backward=X)"
    )
    
    # Same for the input and model configuration
    parser.add_argument(
        "--layer-config",
        nargs='*',
        default=[],
        help="Layer configuration overrides in the format key=value (e.g. --layer-config batch_size=32 n_global=1024)"
    ) 
    
    args = parser.parse_args()
    return args



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

_default_layer_config: dict[str, int] = {
    "x_features": 4096,
    "batch_size": 1,
    "heads": 16,
    "n_global": 512,
}

def training_loop(
    device: torch.device,
    local_rank: int,
    ring_config: ring_attn_fns.RingAttentionConfig,
    layer_config: dict = _default_layer_config,
    run_baseline: bool = True,
    warmup_steps: int = 0,
    steps: int = 10,
) -> None:
    """Training loop for scaling experiments.
    
    Args:
        device: PyTorch device.
        local_rank: Local rank in distributed training.
        ring_config: RingAttentionConfig object. If None, uses default config.
        run_baseline: Whether to run baseline experiments.
        warmup_steps: Number of warmup steps before timing.
        steps: Number of training steps per epoch.
    """
    mesh_dict = make_process_mesh("nccl")

    x_features = layer_config.get("x_features", _default_layer_config["x_features"])
    batch_size = layer_config.get("batch_size", _default_layer_config["batch_size"])
    heads = layer_config.get("heads", _default_layer_config["heads"])
    n_global = layer_config.get("n_global", _default_layer_config["n_global"])
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

    # Run warmup steps if requested
    if warmup_steps > 0 and local_rank == 0:
        print(f"Running {warmup_steps} warmup steps...")
    
    for _ in range(warmup_steps):
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
        output = ring_attention(x.detach(), q, k, v, w_qkv, None)
        output.sum().backward()
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
            steps=steps,
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
    if run_baseline:
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
            train_one_epoch(model_baseline, x, q, k, v, w_qkv, steps=steps)
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
    else:
        if local_rank == 0:
            print("Skipping baseline experiments (use --baseline to run them)")



def train_one_epoch(model: torch.nn.Module, x, q, k, v, w_qkv, steps: int = 10):
    """Time and train one epoch.
    
    Args:
        model: PyTorch model.
        x: Input tensor.
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        w_qkv: QKV weight matrix.
        steps: Number of training steps.
    """
    # X has shape [B, Nlocal, d//2]
    for _ in range(steps):
        nvtx.range_push("Forward")
        output = model(x.detach(), q, k, v, w_qkv, None)
        nvtx.range_pop()
        nvtx.range_push("Backward")
        output.sum().backward()
        nvtx.range_pop()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    config = ring_attn_fns.RingAttentionConfig.default_config() 
    device, localid, local_rank, world_size = utils.init_cuda()
    
    if args.config_file:
        try:
            yaml_config = yaml.load(args.config_file)
            if args.config_file and dist.get_rank() == 0:
                print(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            print(f"ERROR: Failed to load config file: {str(e)}", file=sys.stderr)
            sys.exit(1)
        
        config.update(yaml_config.get("ring", {}))  # Update defaults with YAML config
    
    # Override with any command line arguments
    for override in args.ring_config:
        try:
            key, value = override.split('=')
            if key in config:
                # Attempt to convert to the correct type (bool, int, etc.)
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                config[key] = value
            else:
                print(f"WARNING: Unknown config key '{key}' in override '{override}'", file=sys.stderr)
        except ValueError:
            print(f"ERROR: Invalid ring config override format: '{override}'. Expected format is key=value.", file=sys.stderr)
            sys.exit(1)
    
    ring_config = ring_attn_fns.RingAttentionConfig.from_dict(config)
    
    # Process layer configuration overrides in the same way
    layer_config = _default_layer_config.copy()
    for override in args.layer_config:
        try:
            key, value = override.split('=')
            if key in layer_config:
                if value.isdigit():
                    value = int(value)
                layer_config[key] = value
            else:
                print(f"WARNING: Unknown layer config key '{key}' in override '{override}'", file=sys.stderr)
        except ValueError:
            print(f"ERROR: Invalid layer config override format: '{override}'. Expected format is key=value.", file=sys.stderr)
            sys.exit(1)
    
    # Build ring attention configuration
    if local_rank == 0:
        print(f"Ring Attention Configuration: {ring_config}")
    
    # Handle validation-only mode
    if args.validate:
        if local_rank == 0:
            print("Configuration validation passed!")
            if ring_config:
                print(f"  {ring_config}")
        dist.destroy_process_group()
        sys.exit(0)
    
    # Extract experiment control parameters
    warmup_steps = args.warmup_steps
    steps = args.steps
    validate = args.validate
    run_baseline = args.baseline
    
    if validate and not run_baseline:
        if local_rank == 0:
            print("WARNING: Validation mode enabled, but not the baseline. Baseline will be run for validation purposes.")
        run_baseline = True
        
    
    if local_rank == 0:
        print(f"Experiment settings: warmup_steps={warmup_steps}, steps={steps}, run_baseline={run_baseline}")
    
    if local_rank == 0:
        print(f"Ring Configuration: {ring_config}")
        print(f"Layer Configuration: {layer_config}")

    # # Run training loop
    # training_loop(
    #     device,
    #     local_rank,
    #     ring_config=ring_config,
    #     layer_config=layer_config,
    #     run_baseline=run_baseline,
    #     warmup_steps=warmup_steps,
    #     steps=steps,
    # )
    
    dist.destroy_process_group()

