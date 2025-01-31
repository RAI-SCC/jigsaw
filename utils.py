from mpi4py import MPI
import numpy as np
from Dataset import ERA5Dataset, ERA5DatasetDistributed, worker_init_fn
import torch
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
import matplotlib.pyplot as plt
import sys
from multiprocessing import Value
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/')
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/networks')
from networks.PanguLite2DAttentionPosEmbed import PanguModel as TwoDimPosEmbLite
import random

def set_all_seeds(seed, rank=None):
    if rank is None:
        rank = 0
    os.environ["PL_GLOBAL_SEED"] = str(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def init_mpi():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # rank is from 0 to 3
    world_size = comm.Get_size()
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
    return comm, i, j

def get_sender(iteration, rank):
    # rank 0, 2 receive 0; Rank 1, 3 receive from 1 
    # rank 0, 2 receive from 2, rank 1, 3 receive from 3
    sender = rank % 2 if (iteration % 2 == 0) else (rank % 2 + 2)
    
    return sender 

def send_data_twoway(data, iteration, buffer, rank, device, groups, alt_sender=True):
    sender = get_sender(iteration, rank) if alt_sender else rank % 2   
    data_list = [d.to(torch.float32).to(device) for d in data] if rank == sender else None
    torch.distributed.scatter(buffer, data_list, src=sender, group=groups[rank % 2])
    data = buffer.permute(0, 2, 3, 1)

    return data

def send_data_fourway(data, buffer, rank, device):
    sender = 0
    input_data_list = [d.to(torch.float32).to(device) for d in data] if rank == sender else None
    
    torch.distributed.scatter(buffer, input_data_list, src=sender)
    
    input_data = buffer.permute(0, 2, 3, 1)
    return input_data

class AlternatingRandomSampler(Sampler):
    def __init__(self, dataset_size, rank, batch_size=3):
        self.dataset_size = dataset_size
        self.rank = rank
        self.batch_size = batch_size
        self.indices = list(range(dataset_size))
        random.seed(0)
        random.shuffle(self.indices)
        # Group shuffled indices into batches of size batch_size
        self.batches = [self.indices[i:i+batch_size]
                        for i in range(0, dataset_size, batch_size)]
        
    def __iter__(self):
        # Alternate active ranks [0,1] vs. [2,3] each batch
        for i, batch in enumerate(self.batches):
            active_ranks = [0,1] if i % 2 == 0 else [2,3]
            if self.rank in active_ranks:
                yield batch
            else:
                # Yield empty so that DataLoader sees "no real samples"
                yield []

    def __len__(self):
        # Approximately half the dataset goes to  each rank pair
        return (self.dataset_size + 1) // 2
    
def get_dataloader(config_dict, data_path, mode='train', subset_size=0, distributed=False, batch_size=1, normalize=True, rank=None):
    if config_dict['model']['model_parallel']:

        if config_dict['model_parallel']['alt_sender']:
            batch_counter = Value('i', 0)  # Shared integer for batch counter (process-safe)
            dataset = ERA5DatasetDistributed(config_dict, data_path, mode=mode, patch_size=(2,4,4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, batch_counter=batch_counter)
        elif config_dict['model_parallel']['split'] == 1:
            dataset = ERA5Dataset(config_dict, data_path, mode=mode, patch_size=(2,4,4), forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, rank=rank)    
        else:
            dataset = ERA5DatasetDistributed(config_dict, data_path, mode=mode, patch_size=(2,4,4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, batch_counter=None)
    else:
        dataset = ERA5Dataset(config_dict, data_path, mode=mode, patch_size=(2,4,4), forecast_length=config_dict['model']['latent_rollout'], normalize=normalize)
    if subset_size != 0:
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, subset_indices)
    sampler    = DistributedSampler(dataset, shuffle=False) if distributed else None
    #if config_dict['model']['model_parallel']:
    #    sampler = AlternatingRandomSampler(dataset_size=len(dataset), rank=rank, batch_size=batch_size)
    if mode == 'validation' and config_dict['model']['model_parallel']:
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                sampler=sampler, 
                                drop_last=True, 
                                pin_memory=torch.cuda.is_available(), 
                                prefetch_factor=config_dict['data']['prefetch_factor'], 
                                num_workers=config_dict['data']['num_workers']
                                )
    if config_dict['model']['model_parallel']:
        if config_dict['model_parallel']['alt_sender']:
            shared_batch_counter = Value('i', 0)  # Shared counter for all workers in the rank
            dataloader = DataLoader(dataset, 
                                    shuffle=(sampler is None), 
                                    sampler=sampler, 
                                    drop_last=True, 
                                    pin_memory=torch.cuda.is_available(), 
                                    prefetch_factor=config_dict['data']['prefetch_factor'], 
                                    num_workers=config_dict['data']['num_workers'],
                                    worker_init_fn = lambda worker_id: worker_init_fn(worker_id, shared_batch_counter)
                                    )
        else:
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=(sampler is None), 
                                    sampler=sampler, 
                                    drop_last=True, 
                                    pin_memory=torch.cuda.is_available(), 
                                    prefetch_factor=config_dict['data']['prefetch_factor'], 
                                    num_workers=config_dict['data']['num_workers']
                                    )
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=True, pin_memory=torch.cuda.is_available(), prefetch_factor=config_dict['data']['prefetch_factor'], num_workers=config_dict['data']['num_workers'])

    return dataloader

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

    print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)
    
    # Initialize DistributedDataParallel.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    
    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    else:
        print(f"Rank {rank} not initialized")

    
    return device, slurm_localid, rank, world_size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_constant_mask_2d(
    patch_size,
    folder_path="/hkfs/work/workspace/scratch/ke4365-summa/constant_masks/",
    xlat=721
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
    #x1_pad = (patch_size[0] - (land_mask.shape[0] % patch_size[0])) % patch_size[0] // 2
    #x2_pad = (patch_size[0] - (land_mask.shape[0] % patch_size[0])) % patch_size[
    #    0
    #] - x1_pad
    #y1_pad = (patch_size[1] - (land_mask.shape[1] % patch_size[1])) % patch_size[1] // 2
    #y2_pad = (patch_size[1] - (land_mask.shape[1] % patch_size[1])) % patch_size[
    #    1
    #] - y1_pad

    land_mask = land_mask[:xlat]
    soil_type = soil_type[:xlat]
    topography = topography[:xlat]

    
    return land_mask, soil_type, topography

def plot_fields(input, output, target, save_path, index=1):
    input_print = input[index].detach().cpu()
    output_print = output[index].detach().cpu()
    target_print = target[index].detach().cpu()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    c0 = ax0.pcolormesh(input_print)
    fig.colorbar(c0, ax=ax0)
    c1 = ax1.pcolormesh(output_print)
    fig.colorbar(c1, ax=ax1)
    c2 = ax2.pcolormesh(target_print)
    fig.colorbar(c2, ax=ax2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# save model
def save_model(model, optimizer, file_path, rank, epoch, distributed=True):
    file_path = f"{file_path}/model_{rank}.pt"
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, file_path)

def load_model(model, optimizer, file_path, rank, model_parallel=False):
    if not model_parallel:
        rank = 0
    if model_parallel:
        file_path = f'{file_path}/model_{rank}.pt'
    state = torch.load(f"{file_path}", map_location='cpu')
    model.load_state_dict(state['model_state'])
#    optimizer.load_state_dict(state['optimizer_state'])

    epoch = state['epoch']
    return model, optimizer, epoch

def load_pretrained_convolution(mlp_model, config_dict):
    state = torch.load(config_dict['training']['preload_path'], map_location='cpu')
    model_pretrained_state = state['model_state']
    
    try:
        mlp_model.state_dict()['module.embedding.conv_surface.weight'][:config_dict['model']['hidden_dim']] = model_pretrained_state['module._input_layer.conv_surface.weight']
        mlp_model.state_dict()['module.embedding.conv_surface.bias'][:config_dict['model']['hidden_dim']] = model_pretrained_state['module._input_layer.conv_surface.bias']
    except Exception as err:
        print(err)
    return mlp_model

def load_pretrained_embedding(mlp_model, config_dict):
    state = torch.load(config_dict['training']['preload_path'], map_location='cpu')
    model_pretrained_state = state['model_state']
    
    try:
        mlp_model.load_state_dict({
            **mlp_model.state_dict(),
            'module.embedding.conv_surface.weight': model_pretrained_state['module.embed.weight'],
            'module.recovery.conv.weight': model_pretrained_state['module.recover.weight'],
        })
        
        embed =  mlp_model.state_dict()['module.embedding.conv_surface.weight'].reshape(1152, -1)
        recover = mlp_model.state_dict()['module.recovery.conv.weight'].reshape(1152, -1)
        print("embed, recover", recover.mT @ embed)

        #mlp_model.state_dict()['module.params.embedding.conv_surface.weight'] = model_pretrained_state['module.embedding.conv_surface.weight']
        #mlp_model.state_dict()['module.params.embedding.conv_surface.bias'] = model_pretrained_state['module.embedding.conv_surface.bias']
        #mlp_model.state_dict()['module.params.recovery.conv_surface.weight'] = model_pretrained_state['module.recovery.conv.weight']
        #mlp_model.state_dict()['module.params.recovery.conv_surface.bias'] = model_pretrained_state['module.recovery.conv.bias']
        #mlp_model.state_dict()['module.params.layer_norm3.weight'] = model_pretrained_state['module.layer_norm3.weight']
        #mlp_model.state_dict()['module.params.layer_norm3.bias'] = model_pretrained_state['module.layer_norm3.bias']        
        #mlp_model.state_dict()['module.params.linearlayer.bias'] = model_pretrained_state['module.linearlayer.bias']
        
    except Exception as err:
        print(f"unable to load: {err}")

    return mlp_model


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    P = torch.tensor(P).to(torch.float32)
    return P