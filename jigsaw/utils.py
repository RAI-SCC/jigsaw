import os
from typing import Dict, Optional, Tuple

import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import parameters_to_vector

from . import eval, model, transformer_models

# from jigsaw import Dataloaders, Model, TransformerModels, eval
from .dataloaders import get_dataloader


def get_ddp_groups(rank: int, parallelism: int) -> dist.ProcessGroup:
    """Return DDP groups."""
    n_nodes = dist.get_world_size() // parallelism
    global_group_list = []
    global_groups = []

    for node in np.arange(parallelism):
        global_group_list.append((np.arange(n_nodes) * parallelism + node).tolist())

    for global_group in global_group_list:
        global_groups.append(dist.new_group(ranks=global_group))

    return global_groups[rank % parallelism]


def get_model_group(rank: int, parallelism: int) -> dist.ProcessGroup:
    """Return model groups."""
    local_groups = []
    for i in range(dist.get_world_size() // parallelism):
        local_start_rank = i * parallelism
        local_groups.append(
            dist.new_group(
                ranks=list(range(local_start_rank, local_start_rank + parallelism))
            )
        )
    return local_groups[rank // parallelism]


def get_model_group_rows(rank: int, parallelism: int, n_rows: int) -> dist.ProcessGroup:
    """
    Return mpi comm subgroups within model groups.

    rank: int
        global rank of process
    parallelism: int
        number of processes in model-parallel group
    n_rows: int
        number of rows that the model-parallel group is subdivided into.
        default for 4: 2; default for 8: 4
    """
    groups = []
    n_cols = parallelism // n_rows
    n_nodes = dist.get_world_size() // parallelism

    for node in range(n_nodes):
        base = node * parallelism
        for row in range(n_rows):
            ranks = [base + row * n_cols + c for c in range(n_cols)]
            groups.append(dist.new_group(ranks=ranks))

    node = rank // parallelism
    local = rank % parallelism
    row = local // n_cols
    idx = node * n_rows + row
    return groups[idx]


def get_model_group_columns(
    rank: int, parallelism: int, n_cols: int
) -> dist.ProcessGroup:
    """
    Return mpi comm subgroups within model groups.

    rank: int
        global rank of process
    parallelism: intval_loss
        number of processes in model-parallel group
    n_cols: int
        number of rows that the model-parallel group is subdivided into.
        default for 4: 2; default for 8: 2
    """
    groups = []
    n_rows = parallelism // n_cols
    n_nodes = dist.get_world_size() // parallelism

    for node in range(n_nodes):  # [0, 2], [1, 3], [4, 6], [5, 7]
        base = node * parallelism
        for col in range(n_cols):
            ranks = [base + r * n_cols + col for r in range(n_rows)]
            groups.append(dist.new_group(ranks=ranks))

    node = rank // parallelism
    local = rank % parallelism
    col = local % n_cols
    idx = node * n_cols + col

    return groups[idx]


def set_all_seeds(seed: int = 0) -> None:
    """Initialize all random seeds."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_all_seeds_model_group(
    seed: int = 0, rank: Optional[int] = None, parallelism: int = None
) -> None:
    """Set all seeds on all model groups to the same value."""
    set_all_seeds(seed + rank // parallelism)


def init_cuda() -> tuple[torch.device, int, int, int]:
    """
    Initialize DistributedDataParallel.

    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
    # Get overall number of processes.
    world_size = int(os.getenv("SLURM_NTASKS"))
    slurm_localid = int(os.getenv("SLURM_LOCALID"))

    print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)

    # Initialize DistributedDataParallel.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )

    if dist.is_initialized():
        print(
            f"Rank {rank}/{world_size}: Process group initialized with torch rank {dist.get_rank()} and torch world size {dist.get_world_size()}."
        )
    else:
        print(f"Rank {rank} not initialized")

    return device, slurm_localid, rank, world_size


def count_parameters(model: torch.nn.Module):
    """Return number of parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# save model


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim,
    scheduler: timm.scheduler,
    file_path: str,
    rank: int,
    epoch: int,
    model_name: str = "model",
):  # noqa: D103
    """Save model weights, optimizer state, epoch."""
    file_path = f"{file_path}/{model_name}_{rank}.pt"
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, file_path)


def load_model(
    model: torch.nn.Module,
    optimizer: torch.optim,
    file_path: str,
    rank: int = 0,
    model_parallel: bool = False,
    parallelism: int = 4,
):
    """Load model from state dict."""
    if model_parallel:
        load_rank = rank % parallelism
        file_path = f"{file_path}/model_{load_rank}.pt"
    try:
        state = torch.load(f"{file_path}", map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch = state["epoch"] + 1
        # epoch = 0
    except Exception:
        if rank == 0:
            print(f"Could not load model from {file_path}. Initializing from scratch.")
        epoch = 0
    return model, optimizer, epoch


def get_position_encoding(seq_len: int, d: int, n: int = 10000):
    """Return fixed positional encoding for a given sequence length."""
    p = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            p[k, 2 * i] = np.sin(k / denominator)
            p[k, 2 * i + 1] = np.cos(k / denominator)
    p = torch.tensor(p).to(torch.float32)
    return p


def setup_precision(tf32: bool) -> None:
    """Configure matmul and cudnn TF32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    return


def setup_comm_groups(rank: int, parallelism: int):
    """Define and initialize process groups."""
    match parallelism:
        case 1:
            groups = setup_groups(rank, parallelism, 1, 1)
        case 2:
            groups = setup_groups(rank, parallelism, 1, 2)
        case 4:
            groups = setup_groups(rank, parallelism, 2, 2)
    return groups


def setup_groups(rank: int, parallelism: int, nrows: int, ncols: int):
    """Return communication groups."""
    model_group = get_model_group(rank, parallelism)
    ddp_group = get_ddp_groups(rank, parallelism)
    model_subgroup_row = get_model_group_rows(rank, parallelism, nrows)
    model_subgroup_col = get_model_group_columns(rank, parallelism, ncols)

    return model_group, ddp_group, model_subgroup_row, model_subgroup_col


def build_megatron_model(
    config_dict: Dict,
    device: torch.device,
    localid: int,
    rank: int,
    parallelism: int,
    model_group: dist.ProcessGroup,
    ddp_group: dist.ProcessGroup,
    col_group: dist.ProcessGroup = None,
):
    """Build Megatron model from config file."""
    match parallelism:
        case 1:
            model = model.ModelSequential(config_dict, rank, device).to(device)
        case _:
            model = model.ModelMegatron(
                config_dict, rank, device, parallelism, model_group
            ).to(device)
    # need to match parameters for non-transformer, non-mlp layers
    for name, param in model.named_parameters():
        if ("blocks" not in name) or "layer_norm" in name:
            dist.all_reduce(param.data, op=dist.ReduceOp.AVG)

    model = DistributedDataParallel(
        model, device_ids=[localid], output_device=localid, process_group=ddp_group
    )

    if rank == 0:
        print(
            f"model has {count_parameters(model) * parallelism / 1e6} million parameters"
        )
        print(f"estimated forward FLOPS: {calc_flops_comm(config_dict)}")
    return model


def build_model(
    config_dict: Dict,
    device: torch.device,
    localid: int,
    rank: int,
    parallelism: int,
    model_group: dist.ProcessGroup,
    ddp_group: dist.ProcessGroup,
    col_group: dist.ProcessGroup = None,
) -> torch.nn.Module:
    """Instantiate model and wrap with DDP."""
    match parallelism:
        case 1:
            model = model.ModelSequential(config_dict, rank, device).to(device)
        case 2:
            model = model.ModelTwoway(
                config_dict, rank, device, parallelism, model_group
            ).to(device)
        case 4:
            model = model.Model(
                config_dict, rank, device, parallelism, model_group, col_group
            ).to(device)
            for name, param in model.named_parameters():
                if param is not None:
                    if "bias" in name:
                        dist.all_reduce(
                            param.data, op=dist.ReduceOp.AVG, group=col_group
                        )
        case 8:
            model = model.ModelEightway(
                config_dict, rank, device, parallelism, model_group, col_group
            ).to(device)
            for name, param in model.named_parameters():
                if param is not None:
                    if "bias" in name:
                        dist.all_reduce(
                            param.data, op=dist.ReduceOp.AVG, group=col_group
                        )

    model = DistributedDataParallel(
        model, device_ids=[localid], output_device=localid, process_group=ddp_group
    )

    if rank == 0:
        print(
            f"model has {count_parameters(model) * parallelism / 1e6} million parameters"
        )
        print(f"estimated forward FLOPS: {calc_flops_comm(config_dict)}")
    return model


def build_transformer_model(
    config_dict: dict,
    device: torch.device,
    localid: int,
    rank: int,
    parallelism: int,
    model_group: dist.ProcessGroup,
    ddp_group: dist.ProcessGroup,
    row_group: dist.ProcessGroup = None,
    col_group: dist.ProcessGroup = None,
):
    """Build transformer model from config file."""
    if config_dict["model_parallel"]["type"] == "megatron":
        match parallelism:
            case 1:
                model = transformer_models.TransformerSequential(
                    config_dict,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    row_group,
                    col_group,
                ).to(device)
            case _:
                model = transformer_models.MegatronTransformer(
                    config_dict,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    row_group,
                    col_group,
                ).to(device)
        for name, param in model.named_parameters():
            if ("blocks" not in name) or "layer_norm" in name:
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
    else:
        match parallelism:
            case 1:
                model = transformer_models.TransformerSequential(
                    config_dict,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    row_group,
                    col_group,
                ).to(device)
            case 2:
                model = transformer_models.TransformerTwoway(
                    config_dict,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    row_group,
                    col_group,
                ).to(device)
            case 4:
                model = transformer_models.TransformerFourway(
                    config_dict,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    row_group,
                    col_group,
                ).to(device)

    model = DistributedDataParallel(
        model, device_ids=[localid], output_device=localid, process_group=ddp_group
    )
    if rank == 0:
        print(
            f"model has {count_parameters(model) * parallelism / 1e6} million parameters"
        )
        print(f"estimated forward FLOPS: {calc_flops_transformer(config_dict)}")
    return model


def iter_layer_norm_params(model: torch.nn.Module):
    """Yield (name, parameter) pairs for LayerNorm weights and biases."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for suffix, p in (("weight", module.weight), ("bias", module.bias)):
                if p is not None:
                    yield f"{name}.{suffix}", p


@torch.no_grad()
def ddp_average_layer_norm_weights(
    model: torch.nn.Module, col_group: dist.ProcessGroup
) -> None:
    """
    All-reduce and in-place average LayerNorm weights/biases across processes.

    Shapes: each param is [hidden_dim].
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        # Single process: nothing to do.
        return
    for _, p in iter_layer_norm_params(model):
        torch.distributed.all_reduce(
            p.data, op=torch.distributed.ReduceOp.SUM, group=col_group
        )
        p.data.div_(2)


def build_optimizer(config_dict: Dict, model: torch.nn.Module) -> optim.Optimizer:
    """Build Adam optimizer with per-block lrs."""
    lr = config_dict["training"]["lr"]
    opt = optim.Adam(
        [
            {
                "params": model.module.params.embedding.parameters(),
                "lr": config_dict["training"]["lr_embedding"],
            },
            {
                "params": model.module.params.recovery.parameters(),
                "lr": config_dict["training"]["lr_recovery"],
            },
            {"params": model.module.params.common.parameters()},
        ],
        lr=lr,
        weight_decay=3e-6,
    )
    return opt


def build_scheduler(
    config_dict: Dict,
    optimizer: optim.Optimizer,
    start_epoch: int = 0,
    n_epochs: int = 50,
    lr_min: float = 1e-5,
) -> timm.scheduler:
    """Build LR scheduler and step it to start_epoch."""
    if config_dict["training"]["lr_constant"]:
        scheduler = StepLRScheduler(optimizer, decay_rate=1.0, decay_t=100)
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=n_epochs,
            lr_min=lr_min,
            cycle_decay=1,
            cycle_mul=1,
            cycle_limit=10,
        )
    scheduler.step(start_epoch)
    return scheduler


def build_dataloaders(
    config_dict: Dict, model_parallel: bool, rank: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and validation dataloaders."""
    train_subset = config_dict["data"]["train_subset"]
    valid_subset = config_dict["data"]["valid_subset"]
    train_subset = -1 if train_subset < 0 else train_subset
    valid_subset = -1 if valid_subset < 0 else valid_subset

    train_dataloader, _ = get_dataloader(
        config_dict,
        config_dict["data"]["train_data_path"],
        mode="train",
        distributed=(not model_parallel),
        batch_size=config_dict["training"]["train_batch"],
        subset_size=train_subset,
        rank=rank,
    )
    valid_dataloader, _ = get_dataloader(
        config_dict,
        config_dict["data"]["valid_data_path"],
        mode="validation",
        distributed=(not model_parallel),
        batch_size=config_dict["training"]["valid_batch"],
        subset_size=valid_subset,
        rank=rank,
    )

    return train_dataloader, valid_dataloader


def get_weights(
    config_dict: dict, device: torch.device, parallelism: int, rank: int
) -> tuple[torch.tensor, torch.tensor]:
    """Return variable and latitude-weighted weights."""
    xlat = config_dict["data"]["xlat"]
    surface_weights = torch.tensor(config_dict["training"]["surface_weights"])
    # [z, q, t, u, v]
    pressure_weights = (
        torch.tensor(config_dict["training"]["pressure_weights"])
        .unsqueeze(1)
        .expand(5, 13)
        .flatten()
    )
    variable_weights = (
        torch.concat([surface_weights, pressure_weights]).view(1, 1, 1, 69)
    ).to(device)

    area_weights = (
        eval.calc_weight(xlat).view(1, xlat, 1, 1).to(device).to(torch.float32)
    )
    if config_dict["model_parallel"]["type"] != "megatron":
        variable_weights = torch.nn.functional.pad(variable_weights, (0, 3))
        variable_weights = (
            variable_weights[0, 0, 0, :36]
            if rank % parallelism in [0, 2]
            else variable_weights[0, 0, 0, 36:]
        )

    return variable_weights, area_weights


def define_subset_sizes(config_dict: dict, rank: int) -> tuple[int, int]:
    """Return training and validation subset sizes."""
    train_subset, valid_subset = (
        config_dict["data"]["train_subset"],
        config_dict["data"]["valid_subset"],
    )
    train_subset = None if train_subset < 0 else train_subset
    valid_subset = None if valid_subset < 0 else valid_subset

    if rank == 0:
        print(f"Training subset size {train_subset}")
        print(f"Reading from {config_dict['data']['train_data_path']}")

    return train_subset, valid_subset


def build_loss_fn(config_dict):
    """Return loss function."""
    loss_type = config_dict["training"]["loss"]
    match loss_type:
        case "L1":
            loss_fn = torch.nn.L1Loss()
        case "L2":
            loss_fn = torch.nn.MSELoss()
        case _:
            raise ValueError("Only L1 and L2 losses are supported.")
    return loss_fn


def ab_flop_global(in_dim, hidden_dim, out_dim):
    """Global FLOPs in distributed matrix multiplication."""
    return in_dim * out_dim * (2 * hidden_dim - 1)


def ab_flop_local(in_dim, hidden_dim, out_dim):
    """Local FLOPs in distributed matrix multiplication."""
    in_local, hidden_local, out_local = in_dim / 2, hidden_dim / 2, out_dim / 2
    flop = 0
    for proc in range(4):
        if proc == 0:
            flop = (
                flop
                + 2 * ab_flop_global(in_local, hidden_local, out_local)
                + in_local * out_local
            )
        elif proc == 1:
            flop = (
                flop
                + 2 * ab_flop_global(in_local, hidden_local, out_local)
                + in_local * out_local
            )
        elif proc == 2:
            flop = (
                flop
                + 2 * ab_flop_global(in_local, hidden_local, out_local)
                + in_local * out_local
            )
        elif proc == 3:
            flop = (
                flop
                + 2 * ab_flop_global(in_local, hidden_local, out_local)
                + in_local * out_local
            )
    return flop / 1e12


def ab_communication(in_dim, hidden_dim, out_dim, dtype="float32"):
    """Communication for matrix multiplication."""
    # four processes
    # return parameters only
    in_local, hidden_local, out_local = in_dim / 2, hidden_dim / 2, out_dim / 2
    comm = 0
    for proc in range(4):
        if proc == 0:
            comm = comm + in_local * hidden_local + hidden_local * out_local
        elif proc == 1:
            comm = comm + in_local * hidden_local + hidden_local * out_local
        elif proc == 2:
            comm = comm + in_local * hidden_local + hidden_local * out_local
        elif proc == 3:
            comm = comm + in_local * hidden_local + hidden_local * out_local

    return comm / (2**30)


def transpose_communication(in_dim, out_dim):
    """Calculate volume of data transferred in a transpose."""
    in_local, out_local = in_dim / 2, out_dim / 2
    return in_local * out_local * 2 / (2**30)


def mixing_block(ntokens, patch_embedding_out, spatial_hidden_dim, channels_hidden_dim):
    """Estimate FLOPs and volume of data transfered for a mixing block."""
    # input shape = output of patch embedding shape
    input_shape = (ntokens, patch_embedding_out)
    flop = 0
    comm = 0
    # Transpose
    comm = comm + transpose_communication(*input_shape)
    # Token mixing
    flop = (
        flop
        + ab_flop_local(patch_embedding_out, ntokens, spatial_hidden_dim)
        + spatial_hidden_dim / 1e12
    )
    comm = comm + ab_communication(patch_embedding_out, ntokens, spatial_hidden_dim)
    # activation
    flop = flop + patch_embedding_out * spatial_hidden_dim / 1e12
    # token mixing 2
    flop = (
        flop
        + ab_flop_local(patch_embedding_out, spatial_hidden_dim, ntokens)
        + ntokens / 1e12
    )
    comm = comm + ab_communication(patch_embedding_out, spatial_hidden_dim, ntokens)

    # transpose
    comm = comm + transpose_communication(patch_embedding_out, ntokens)

    # Residual connection addition
    flop = flop + ntokens * patch_embedding_out / 1e12

    # channel mixing
    flop = (
        flop
        + ab_flop_local(ntokens, patch_embedding_out, channels_hidden_dim)
        + channels_hidden_dim / 1e12
    )
    comm = comm + ab_communication(ntokens, patch_embedding_out, channels_hidden_dim)
    # activation
    flop = flop + ntokens * channels_hidden_dim / 1e12
    # channel mixing 2
    flop = (
        flop
        + ab_flop_local(ntokens, channels_hidden_dim, patch_embedding_out)
        + patch_embedding_out / 1e12
    )
    comm = comm + ab_communication(ntokens, channels_hidden_dim, patch_embedding_out)

    # residual addition
    flop = flop + ntokens * patch_embedding_out / 1e12

    return flop, comm  # FLOP: GigaFLOP, Comm: GBs


def transformer_block(ntokens, hidden_dim, transformer_dim, out_dim, heads):
    """Calculate the FLOPs and data volume transferred for a Transformer."""
    # input shape = output of patch embedding shape
    flop = 0
    comm = 0

    # qkv calculation
    # X: (n_tokens, hidden_dim) @ (hidden_dim, 3 * transformer_dimension)
    flop = flop + ab_flop_local(ntokens, hidden_dim, 3 * transformer_dim)
    comm = comm + ab_communication(ntokens, hidden_dim, 3 * transformer_dim)

    # Q @ K.mT
    # Q: (n_tokens, transformer_dim) K.mT : (transformer_dim, n_tokens)
    flop = flop + heads * ab_flop_local(ntokens, transformer_dim // heads, ntokens)
    # TODO: currently ignore softmax
    # (Q@K.mT) @ V
    # (Q@K.mT): (n_tokens, n_tokens) @ (n_tokens, transformer_dim)
    flop = flop + heads * ab_flop_local(ntokens, ntokens, transformer_dim // heads)

    # concatenator
    flop = flop + ab_flop_local(ntokens, transformer_dim, out_dim)

    return flop, comm  # FLOP: GigaFLOP, Comm: GBs


def calc_flops_comm(config):
    """Estimates the number of FLOPs and volume of communicated data."""
    nlat, nlon = config["data"]["xlat"], config["data"]["xlon"]
    nvars_in, nvars_out = 72, 72
    patch_size = config["model"]["patch_size"]
    patch_embedding_in = patch_size * nvars_in
    patch_embedding_out = config["model"]["hidden_dim"]
    # where ntokens = nlat*nlon/patch_size
    input_data = (nlat * nlon / patch_size, patch_size * nvars_in)
    ntokens = nlat * nlon / patch_size
    spatial_hidden_dim = int(config["model"]["spatial_hidden_dim_fraction"] * ntokens)
    channels_hidden_dim = int(
        config["model"]["features_hidden_dim_fraction"] * patch_embedding_out
    )

    patch_embedding_comm = ab_communication(
        ntokens, patch_size * nvars_in, patch_embedding_out
    )  # GBs of data communicated
    patch_embedding_flop = (
        ab_flop_local(ntokens, patch_size * nvars_in, patch_embedding_out)
        + patch_embedding_out / 1e12
    )

    # including concat
    # (ntokens, patch_embedding_out)
    patch_recovery_comm = ab_communication(
        ntokens, patch_embedding_out * 2, patch_embedding_in
    )  # GBs of data communicated
    patch_recovery_flop = (
        ab_flop_local(ntokens, patch_embedding_out * 2, patch_embedding_in)
        + patch_embedding_in / 1e12
    )

    linlayer_comm = ab_communication(
        nlat * nlon, nvars_out * 2, nvars_out
    )  # GBs of data communicated
    linlayer_flop = (
        ab_flop_local(nlat * nlon, nvars_out * 2, nvars_out) + nvars_out / 1e12
    )

    tflop_per_block, gb_per_block = mixing_block(
        input_data[0], patch_embedding_out, spatial_hidden_dim, channels_hidden_dim
    )

    flop_forward = (
        tflop_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_flop
        + linlayer_flop
        + patch_recovery_flop
    )
    comm_forward = (
        gb_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_comm
        + linlayer_comm
        + patch_recovery_comm
    )

    return flop_forward, comm_forward


def calc_flops_transformer(config):
    """Estimates the number of FLOPS in the Transformer block."""
    nlat, nlon = config["data"]["xlat"], config["data"]["xlon"]
    nvars_in, nvars_out = 72, 72
    patch_size = config["model"]["patch_size"]
    patch_embedding_in = patch_size * nvars_in
    patch_embedding_out = config["model"]["hidden_dim"]
    transformer_dim = config["model"]["transformer_dim"]
    # where ntokens = nlat*nlon/patch_size
    input_data = (nlat * nlon / patch_size, patch_size * nvars_in)
    ntokens = nlat * nlon / patch_size
    patch_embedding_comm = ab_communication(
        ntokens, patch_size * nvars_in, patch_embedding_out
    )  # GBs of data communicated
    patch_embedding_flop = (
        ab_flop_local(ntokens, patch_size * nvars_in, patch_embedding_out)
        + patch_embedding_out / 1e12
    )

    # including concat
    # (ntokens, patch_embedding_out)
    patch_recovery_comm = ab_communication(
        ntokens, patch_embedding_out * 2, patch_embedding_in
    )  # GBs of data communicated
    patch_recovery_flop = (
        ab_flop_local(ntokens, patch_embedding_out * 2, patch_embedding_in)
        + patch_embedding_in / 1e12
    )

    linlayer_comm = ab_communication(
        nlat * nlon, nvars_out * 2, nvars_out
    )  # GBs of data communicated
    linlayer_flop = (
        ab_flop_local(nlat * nlon, nvars_out * 2, nvars_out) + nvars_out / 1e12
    )

    tflop_per_block, gb_per_block = transformer_block(
        input_data[0],
        patch_embedding_out,
        transformer_dim,
        patch_embedding_out,
        config["model"]["heads"],
    )

    flop_forward = (
        tflop_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_flop
        + linlayer_flop
        + patch_recovery_flop
    )
    comm_forward = (
        gb_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_comm
        + linlayer_comm
        + patch_recovery_comm
    )

    return flop_forward, comm_forward


def reduce_ddp_layernorms(model, col_groups):
    """
    Reduce gradients across local groups for 4-way parallel.

    For processes that predict the same variables.
    """
    # Now manually all-reduce each parameter's grad
    handles = []
    gradients = [
        param.grad
        for name, param in model.named_parameters()
        if (param.grad is not None and ("layer_norm" in name or "bias" in name))
    ]
    flat_grad = parameters_to_vector(gradients)
    handles = torch.distributed.all_reduce(
        flat_grad,
        op=torch.distributed.ReduceOp.AVG,
        group=col_groups,
        async_op=True,
    )

    return handles, flat_grad, gradients
