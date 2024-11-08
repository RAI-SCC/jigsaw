from mpi4py import MPI
import numpy as np
from Dataset import ERA5Dataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler


def init_mpi():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # rank is from 0 to 3
    world_size = comm.Get_size()
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
    return comm, i, j

def get_dataloader(config_dict, valid_subset=0, distributed=False, batch_size=1, normalize=True):
    if valid_subset == 0:
        mode = 'train'
    else:
        mode = 'validation'
    
    dataset = ERA5Dataset(config_dict, '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr', mode=mode, patch_size=(2,4,4), forecast_length=1, normalize=normalize)
    if valid_subset != 0:
        subset_indices = torch.randperm(len(dataset))[:valid_subset]
        dataset = Subset(dataset, subset_indices)
    sampler    = DistributedSampler(dataset, shuffle=True) if distributed else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=True, pin_memory=torch.cuda.is_available(), prefetch_factor=config_dict['data']['prefetch_factor'], num_workers=config_dict['data']['num_workers'])

    return dataloader

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_CUDA():
    """
    Initialize DistributedDataParallel 
    
    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID"))

    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)
    
    # Initialize DistributedDataParallel.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    
    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    
    return device, slurm_localid, rank, world_size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_constant_mask_2d(
    patch_size,
    folder_path="constant_masks/",
):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int)
        Number of pixels in (lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    # Load data from numpy files
    data_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    data = {}
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        data[file] = np.load(file_path)

    soil_type = data["soil_type.npy"]
    topography = data["topography.npy"]

    soil_type = (soil_type - np.mean(soil_type)) / np.std(soil_type)
    topography = (topography - np.mean(topography)) / np.std(topography)
    # Torch tensors
    land_mask = torch.tensor(data["land_mask.npy"]).to(torch.float32)
    soil_type = torch.tensor(soil_type).to(torch.float32)
    topography = torch.tensor(topography).to(torch.float32)

    # Check that the shapes of all the data are the same
    assert (
        land_mask.shape == soil_type.shape == topography.shape
    ), "Shapes of the three constant masks are not equal."

    # Now that the shapes are equal, use land_mask as the actual shapes
    x1_pad = (patch_size[0] - (land_mask.shape[0] % patch_size[0])) % patch_size[0] // 2
    x2_pad = (patch_size[0] - (land_mask.shape[0] % patch_size[0])) % patch_size[
        0
    ] - x1_pad
    y1_pad = (patch_size[1] - (land_mask.shape[1] % patch_size[1])) % patch_size[1] // 2
    y2_pad = (patch_size[1] - (land_mask.shape[1] % patch_size[1])) % patch_size[
        1
    ] - y1_pad

    # Apply padding according to patch embedding size
    # Pad the same way as input shape (ensure code is cohesive)
    #land_mask = torch.nn.functional.pad(
    #    land_mask, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    #)
    #soil_type = torch.nn.functional.pad(
    #    soil_type, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    #)
    #topography = torch.nn.functional.pad(
    #    topography, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    #)

    land_mask = land_mask[:720]
    soil_type = soil_type[:720]
    topography = topography[:720]
    
    return land_mask, soil_type, topography