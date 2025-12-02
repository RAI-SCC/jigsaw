import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from Dataset import (
    DistributedHighResDataset,
    HighResDataset,
    ERA5Dataset,
    ERA5DatasetDistributed,
    ERA5Preprocessed,
    ERA5PreprocessedDistributed
)


def get_highres_dataloader(config_dict, data_path, mode='train', subset_size=0, distributed=False, batch_size=1, normalize=True, rank=None, dt=None, type='dummy'):
    n_parallel = config_dict['model_parallel']['parallelism']
    dt = config_dict['data']['dt'] if dt is None else dt
    if n_parallel == 1:
        DistributedHighResDataset(config_dict, data_path, mode=mode, patch_size=(
            2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
    elif type == 'upsample':
        dataset = DistributedHighResDataset(config_dict, data_path, mode=mode, patch_size=(
            2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)

    if subset_size != 0:
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, subset_indices)

    if mode == 'validation' and n_parallel > 1:
        sampler = DistributedSampler(dataset,
                                     shuffle=False,
                                     num_replicas=(
                                         torch.distributed.get_world_size()//n_parallel),
                                     rank=rank//n_parallel
                                     )
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                prefetch_factor=config_dict['data']['prefetch_factor'],
                                num_workers=config_dict['data']['num_workers']
                                )
        return dataloader
    else:
        sampler = DistributedSampler(dataset,
                                     shuffle=True,
                                     num_replicas=(
                                         torch.distributed.get_world_size()//n_parallel),
                                     rank=rank//n_parallel
                                     )
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=(sampler is None),
                                sampler=sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                prefetch_factor=config_dict['data']['prefetch_factor'],
                                num_workers=config_dict['data']['num_workers']
                                )

    return dataloader


def get_dataloader(config_dict, data_path, mode='train', subset_size=0, distributed=False, batch_size=1, normalize=True, rank=None, dt=None):
    parallelism = config_dict['model_parallel']['parallelism']
    dt = config_dict['data']['dt'] if dt is None else dt
    if parallelism == 1:
        if config_dict['data']['preprocessed']:
            if config_dict['data']['resolution'] == 0.25:
                dataset = ERA5Preprocessed(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
            else:
                dataset = HighResDataset(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
        else:
            dataset = ERA5Dataset(config_dict, data_path, mode=mode, patch_size=(
                2, 4, 4), forecast_length=config_dict['model']['latent_rollout'], normalize=normalize)
    else:
        if config_dict['data']['preprocessed']:
            if config_dict['model_parallel']['type'] == 'megatron':
                dataset = ERA5Preprocessed(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
            elif config_dict['data']['resolution'] == 0.25:
                dataset = ERA5PreprocessedDistributed(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
            elif config_dict['data']['resolution'] == 0.125:
                dataset = DistributedHighResDataset(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
        else:
            if config_dict['model_parallel']['type'] == 'megatron':
                dataset = ERA5Dataset(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), forecast_length=config_dict['model']['latent_rollout'], normalize=normalize)
            else:
                dataset = ERA5DatasetDistributed(config_dict, data_path, mode=mode, patch_size=(
                    2, 4, 4), rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
            
    if subset_size > 0:
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, subset_indices)
    if subset_size > len(dataset):
        subset_indices = torch.randperm(800).repeat(config_dict['data']['train_subset'] // 800)
        dataset = Subset(dataset, subset_indices)
    if mode == 'validation' and parallelism > 1:
        sampler = DistributedSampler(dataset,
                                     shuffle=False,
                                     num_replicas=(
                                         torch.distributed.get_world_size()//parallelism),
                                     rank=rank//parallelism
                                     )
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                prefetch_factor=config_dict['data']['prefetch_factor'],
                                num_workers=config_dict['data']['num_workers']
                                )
        return dataloader
    else:
        sampler = DistributedSampler(dataset,
                                     shuffle=True,
                                     num_replicas=(
                                         torch.distributed.get_world_size()//parallelism),
                                     rank=rank//parallelism
                                     )
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=(sampler is None),
                                sampler=sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                prefetch_factor=config_dict['data']['prefetch_factor'],
                                num_workers=config_dict['data']['num_workers']
                                )
    return dataloader
