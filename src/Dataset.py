import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
import os
class ERA5Dataset(Dataset):
    """Define 3D dataset."""
    def __init__(self, config, file_path, mode, patch_size, forecast_length=1, normalize=True, rank=None):
        """
        Initialize dataset.

        Parameters
        ----------
        params: Dict
            configuration yaml file.
        file_path: String
            path to data directory.
        distributed: bool
            flag for DDP.
        mode: String
            of value 'training', 'testing', 'validation'.
        device: String
            Device to perform computation on.
        patch_size: Tuple(int, int, Optional[int])
            Number of pixels in ([vert], lat, lon) dimensions per patch
        forecast_length: int
            Defines the number of future steps to return.
        normalize: bool
            Flag for normalization.
        rank: int
            Global rank, optional.
        """
        self.file_path = file_path
        self.mode = mode
        self.dt = config['data']['dt']
        self.lite = config['data']['lite']
        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':        
            self.deltaTDivisor = 6
        elif config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5_subset.zarr':        
            self.deltaTDivisor = 6
        else:
            if self.lite:
                self.deltaTDivisor = 6
            else:
                self.deltaTDivisor = 6
        self.forecast_length = forecast_length
        self.normalize = normalize
        
        self._get_files_stats(mode=self.mode)
        
        # If data is downloaded from weatherbench, the 6-hourly subsampled data is stored in the reverse pressure level order.
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if (config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr' or 
            config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5_subset.zarr'):
            self.level_ordering = range(13-1, -1, -1)
        elif (config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr' or 
              config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr'):
            self.level_ordering = range(0, 13)
        else: # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13-1, -1, -1)

        self.p_mean = np.load('/leonardo_work/EUHPC_B27_052/data/constant_masks/pressure_zarr.npy', allow_pickle=True)[0].reshape(5, 13, 1, 1)
        self.p_std  = np.load('/leonardo_work/EUHPC_B27_052/data/constant_masks/pressure_zarr.npy', allow_pickle=True)[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load('/leonardo_work/EUHPC_B27_052/data/constant_masks/surface_zarr.npy', allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load('/leonardo_work/EUHPC_B27_052/data/constant_masks/surface_zarr.npy', allow_pickle=True)[1].reshape(4, 1, 1)
        
        self.patch_size = patch_size
        self.xlat = config['data']['xlat']
        self.rank = rank
        
        
    def _get_files_stats(self, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if mode == 'train': # training case, lite
            if self.lite:
                train_years = times[(times.year<2017) & (times.year > 2006) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))] 
            else:
                train_years = times[(times.year<2017) & (times.year > 1979) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == 'validation':
            validation_years = times[(times.year == 2017) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'testing':            
            validation_years = times[(times.year == 2020) | (times.year == 2021)] 
            self.zarr_data = self.zarr_data.sel(time=validation_years)
            
        self.n_samples_total = len(self.zarr_data['time'])
        
        
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  - 1 # -1 to avoid last data point
    
    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_pressure = []
        target_surface  = []
        step = self.dt 

        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(torch.arange(1, self.forecast_length+1)*step, self.deltaTDivisor , rounding_mode='trunc')
            
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__(): 
                target_idx = self.__len__() - 1
            
        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        input_surface  = self.zarr_data.isel(time=global_idx)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        target_pressure_ds = self.zarr_data.isel(time=target_file_idxs, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        target_surface_ds  = self.zarr_data.isel(time=target_file_idxs)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        # Stack and convert to numpy array
        input_pressure = np.stack([input_pressure['geopotential'].values, input_pressure['specific_humidity'].values, input_pressure['temperature'].values, input_pressure['u_component_of_wind'].values, input_pressure['v_component_of_wind'].values], axis=0)
        input_surface  = np.stack([input_surface['mean_sea_level_pressure'].values, input_surface['10m_u_component_of_wind'].values, input_surface['10m_v_component_of_wind'].values, input_surface['2m_temperature'].values], axis=0)
        
        input_pressure = (input_pressure - self.p_mean) / self.p_std
        input_surface  = (input_surface  - self.s_mean) / self.s_std
        input_pressure = np.reshape(input_pressure, (-1, 721, 1440))
        input_surface  = input_surface[:, :self.xlat]
        input_pressure = input_pressure[:, :self.xlat]
        
        
        
        input = np.concatenate([input_surface, input_pressure], axis=0)
        
        target = []
        for i in range(self.forecast_length):
            pressure_ds = target_pressure_ds.isel(time=i)
            surface_ds  = target_surface_ds.isel(time=i)
            target_pressure.append(np.stack([pressure_ds['geopotential'].values, pressure_ds['specific_humidity'].values, pressure_ds['temperature'].values, pressure_ds['u_component_of_wind'].values, pressure_ds['v_component_of_wind'].values], axis=0))
            target_surface.append(np.stack([surface_ds['mean_sea_level_pressure'].values, surface_ds['10m_u_component_of_wind'].values, surface_ds['10m_v_component_of_wind'].values, surface_ds['2m_temperature'].values], axis=0))
            target_pressure[i] = (target_pressure[i] - self.p_mean) / self.p_std
            target_surface[i]  = (target_surface[i] - self.s_mean) / self.s_std
            target_pressure[i], target_surface[i] = torch.tensor(target_pressure[i]), torch.tensor(target_surface[i])
            
            target_pressure[i] = target_pressure[i].reshape(-1, 721, 1440)
            target_pressure[i] = target_pressure[i][:, :self.xlat]
            target_surface[i] = target_surface[i][:, :self.xlat]
            target.append(torch.concat([target_surface[i], target_pressure[i]], axis=0))

        input = torch.tensor(input)
        
        return input, target    
        
class ERA5DatasetDistributed(Dataset):
    """Define 3D dataset."""
    def __init__(self, config, file_path, mode, patch_size, rank, forecast_length=1, normalize=True, dt=None):
        """
        initialize.

        params: Dict
            configuration file
        file_path: String
            path to data directory
        distributed: bool
            flag for DDP
        mode: String
            of value 'training', 'testing', 'validation'
        device: String
            device that the code is running/offloaded on
        patch_size: Tuple(int, int, Optional[int])
            Number of pixels in ([vert], lat, lon) dimensions per patch
        forecast_length: int
            For training, always 1. For validation, defines the number of autoregressive steps to roll out to.
        """
        self.file_path = file_path
        self.mode = mode
        self.dt = dt if dt is not None else config['data']['dt']
        self.lite = config['data']['lite']
        self.rank = rank
        self.datapath = config['data']['train_data_path']     

        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':        
            self.deltaTDivisor = 6
        elif config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr' and not self.lite:
            self.deltaTDivisor = 1
        elif config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr' and not self.lite:
            self.deltaTDivisor = 1
        else:
            if self.lite:
                self.deltaTDivisor = 1 
            else:
                self.deltaTDivisor = 6

        self.forecast_length = forecast_length
        self.normalize = normalize
        
        self._get_files_stats(mode=self.mode)
        
        # Current copies of subsampled WeatherBench2 data saves in reverse pressure-level order
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
            self.level_ordering = range(13-1, -1, -1)
        elif config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
            self.level_ordering = range(0, 13)
        elif config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr':
            self.level_ordering = range(0, 13)
        else: # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13-1, -1, -1)

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/pressure_zarr.npy', allow_pickle=True)[0].reshape(65, 1, 1)
        self.p_std  = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/pressure_zarr.npy', allow_pickle=True)[1].reshape(65, 1, 1)
        self.s_mean = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/surface_zarr.npy', allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/surface_zarr.npy', allow_pickle=True)[1].reshape(4, 1, 1)
        
        self.patch_size = patch_size
        self.xlat, self.xlon = config['data']['xlat'], config['data']['xlon']
        self.batch_size = config['training']['train_batch']
        self.parallelism = config['model_parallel']['parallelism']

        if self.parallelism == 2:
            self.start, self.end = 0, self.xlon
        elif self.parallelism == 4:
            if self.rank % 4 in [0, 1]: 
                self.start, self.end = 0, self.xlon//2
            else:
                self.start, self.end = self.xlon//2, self.xlon
        elif self.parallelism == 8:
            if self.rank % 4 in [0, 1]:
                self.start, self.end = 0, self.xlon//4
            elif self.rank % 4 in [2, 3]:
                self.start, self.end = self.xlon//4, self.xlon//2
            elif self.rank % 4 in [4, 5]:
                self.start, self.end = self.xlon//2, self.xlon*3//4
            elif self.rank % 4 in [6, 7]:
                self.start, self.end = self.xlon*3//4, self.xlon
               
        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size, xlat=self.xlat)
        self.land_mask = torch.unsqueeze(self.land_mask[:,self.start:self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:,self.start:self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:,self.start:self.end], 0)

    def _get_files_stats(self, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if mode == 'train': # training case, lite
            if self.lite:
                train_years = times[(times.year<2018) & (times.year > 2006) ]#& ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))] 
            elif self.datapath == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr' and not self.lite:
                train_years = times[(times.year<2018) & (times.year > 1979)]
            else:
                train_years = times[(times.year<2018) & (times.year > 1979) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == 'validation':           # validation
            if self.lite:
                validation_years = times[(times.year == 2018)]# & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            elif (not self.lite) and self.datapath == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr':
                validation_years = times[(times.year == 2018)]
            else:
                validation_years = times[(times.year < 2018)& ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'testing':           # validation
            validation_years = times[(times.year == 2020) | (times.year == 2021)] 
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'verify':
            validation_years = times[(times.year == 2020) & (times.month == 1) & (times.day == 1) & (times.hour == 0)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
            
        self.n_samples_total = len(self.zarr_data['time'])
        
        # N channels
        self.n_in_channels   = len(self.zarr_data.data_vars)
        
        
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  
    
        
    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface  = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(torch.arange(1, self.forecast_length+1)*step, self.deltaTDivisor , rounding_mode='trunc')
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__(): 
                target_idx = self.__len__() - 1
            
        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        input_surface  = self.zarr_data.isel(time=global_idx)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        target_pressure_ds = self.zarr_data.isel(time=target_file_idxs, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        target_surface_ds  = self.zarr_data.isel(time=target_file_idxs)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            
        target = []
        if self.rank % 2 == 0: # first half of data
            # Input data
            # Stack and convert to numpy array
            geopotential = input_pressure['geopotential'][:,:self.xlat,self.start:self.end].values
            humidity = input_pressure['specific_humidity'][:,:self.xlat,self.start:self.end].values
            temperature = input_pressure['temperature'][:6,:self.xlat,self.start:self.end].values
            
            mslp = input_surface['mean_sea_level_pressure'][:self.xlat,self.start:self.end].values
            u10  = input_surface['10m_u_component_of_wind'][:self.xlat,self.start:self.end].values
            v10  = input_surface['10m_v_component_of_wind'][:self.xlat,self.start:self.end].values
            t2m  = input_surface['2m_temperature'][:self.xlat,self.start:self.end].values
            
        
            input_pressure = np.concatenate([geopotential, humidity, temperature], axis=0)
            input_surface  = np.stack([mslp, u10, v10, t2m], axis=0)
            input_pressure = (input_pressure - self.p_mean[:32]) / self.p_std[:32]
            input_surface  = (input_surface  - self.s_mean) / self.s_std
            input_pressure = np.reshape(input_pressure, (-1, self.xlat, self.end-self.start))
            
            input = np.concatenate([input_surface, input_pressure], axis=0)
            
            # Output data
            for i in range(self.forecast_length):

                pressure_ds = target_pressure_ds.isel(time=i)
                surface_ds  = target_surface_ds.isel(time=i)
                
                geopotential = pressure_ds['geopotential'][:,:self.xlat,self.start:self.end].values
                humidity = pressure_ds['specific_humidity'][:,:self.xlat,self.start:self.end].values
                temperature = pressure_ds['temperature'][:6,:self.xlat,self.start:self.end].values
                
                mslp = surface_ds['mean_sea_level_pressure'][:self.xlat,self.start:self.end].values
                u10  = surface_ds['10m_u_component_of_wind'][:self.xlat,self.start:self.end].values
                v10  = surface_ds['10m_v_component_of_wind'][:self.xlat,self.start:self.end].values
                t2m  = surface_ds['2m_temperature'][:self.xlat,self.start:self.end].values
                temp = np.concatenate([geopotential, humidity, temperature], axis=0)
                target_surface.append(np.stack([mslp, u10, v10, t2m], axis=0))

                temp = (temp - self.p_mean[:32]) / self.p_std[:32]
                target_surface[i]  = (target_surface[i] - self.s_mean) / self.s_std
                temp = temp.reshape(-1, self.xlat, self.end-self.start)
                
                temp = np.concatenate([target_surface[i], temp], axis=0)
                target.append(torch.tensor(temp).permute(1, 2, 0))
        else:
            temperature = input_pressure['temperature'][6:,:self.xlat,self.start:self.end].values
            u_velocity  = input_pressure['u_component_of_wind'][:,:self.xlat,self.start:self.end].values
            v_velocity = input_pressure['v_component_of_wind'][:,:self.xlat,self.start:self.end].values
            
            input = np.concatenate([temperature, u_velocity, v_velocity], axis=0)
            
            input = (input - self.p_mean[32:]) / self.p_std[32:]
            input = np.reshape(input, ( -1, self.xlat, self.end-self.start))
            input = np.concatenate([input, self.land_mask, self.soil_type, self.topography], axis=0)

            for i in range(self.forecast_length):
                pressure_ds = target_pressure_ds.isel(time=i)
                temperature = pressure_ds['temperature'][6:,:self.xlat,self.start:self.end].values
                u_velocity  = pressure_ds['u_component_of_wind'][:,:self.xlat,self.start:self.end].values
                v_velocity = pressure_ds['v_component_of_wind'][:,:self.xlat,self.start:self.end].values
                
                temp = np.concatenate([temperature, u_velocity, v_velocity], axis=0)
                
                temp = (temp - self.p_mean[32:]) / self.p_std[32:]
                
                temp = temp.reshape(-1, self.xlat, self.end-self.start) 
                
                target.append(torch.nn.functional.pad(torch.tensor(temp), (0, 0, 0, 0, 0, 3)).permute(1, 2, 0)) 
                            
        input = torch.tensor(input).permute(1,2,0)
        
        return input, target


class ERA5Preprocessed(Dataset):
    """Define 3D dataset."""
    def __init__(self, config, file_path, mode, patch_size, rank, forecast_length=1, normalize=True, dt=None):
        """
        initialize.

        params: Dict
            configuration file
        file_path: String
            path to data directory
        distributed: bool
            flag for DDP
        mode: String
            of value 'training', 'testing', 'validation'
        device: String
            device that the code is running/offloaded on
        patch_size: Tuple(int, int, Optional[int])
            Number of pixels in ([vert], lat, lon) dimensions per patch
        forecast_length: int
            For training, always 1. For validation, defines the number of autoregressive steps to roll out to.
        """
        self.file_path = file_path
        self.mode = mode
        self.dt = dt if dt is not None else config['data']['dt']
        self.lite = config['data']['lite']
        self.rank = rank
        self.datapath = config['data']['train_data_path']     

        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':        
            self.deltaTDivisor = 6
        elif config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr' and not self.lite:
            self.deltaTDivisor = 1
        elif config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr' and not self.lite:
            self.deltaTDivisor = 1
        else:
            if self.lite:
                self.deltaTDivisor = 1 
            else:
                self.deltaTDivisor = 6

        self.forecast_length = forecast_length
        self.normalize = normalize
        
        self._get_files_stats(mode=self.mode)
        
        # Current copies of subsampled WeatherBench2 data saves in reverse pressure-level order
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
            self.level_ordering = range(13-1, -1, -1)
        elif config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
            self.level_ordering = range(0, 13)
        elif config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr':
            self.level_ordering = range(0, 13)
        else: # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13-1, -1, -1)

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/pressure_zarr.npy', allow_pickle=True)[0].reshape(65, 1, 1)
        self.p_std  = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/pressure_zarr.npy', allow_pickle=True)[1].reshape(65, 1, 1)
        self.s_mean = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/surface_zarr.npy', allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load('/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/surface_zarr.npy', allow_pickle=True)[1].reshape(4, 1, 1)
        
        self.patch_size = patch_size
        self.xlat, self.xlon = config['data']['xlat'], config['data']['xlon']
        self.batch_size = config['training']['train_batch']
        self.parallelism = config['model_parallel']['parallelism']

        if self.parallelism == 2:
            self.start, self.end = 0, self.xlon
        elif self.parallelism == 4:
            if self.rank % 4 in [0, 1]: 
                self.start, self.end = 0, self.xlon//2
            else:
                self.start, self.end = self.xlon//2, self.xlon
        elif self.parallelism == 8:
            if self.rank % 4 in [0, 1]:
                self.start, self.end = 0, self.xlon//4
            elif self.rank % 4 in [2, 3]:
                self.start, self.end = self.xlon//4, self.xlon//2
            elif self.rank % 4 in [4, 5]:
                self.start, self.end = self.xlon//2, self.xlon*3//4
            elif self.rank % 4 in [6, 7]:
                self.start, self.end = self.xlon*3//4, self.xlon
               
        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size, xlat=self.xlat)
        self.land_mask = torch.unsqueeze(self.land_mask[:,self.start:self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:,self.start:self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:,self.start:self.end], 0)

    def _get_files_stats(self, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if mode == 'train': # training case, lite
            if self.lite:
                train_years = times[(times.year<2018) & (times.year > 2006) ]#& ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))] 
            elif self.datapath == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr' and not self.lite:
                train_years = times[(times.year<2018) & (times.year > 1979)]
            else:
                train_years = times[(times.year<2018) & (times.year > 1979) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == 'validation':           # validation
            if self.lite:
                validation_years = times[(times.year == 2018)]# & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            elif (not self.lite) and self.datapath == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr':
                validation_years = times[(times.year == 2018)]
            else:
                validation_years = times[(times.year < 2018)& ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'testing':           # validation
            validation_years = times[(times.year == 2020) | (times.year == 2021)] 
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'verify':
            validation_years = times[(times.year == 2020) & (times.month == 1) & (times.day == 1) & (times.hour == 0)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
            
        self.n_samples_total = len(self.zarr_data['time'])
        
        # N channels
        self.n_in_channels   = len(self.zarr_data.data_vars)
        
        
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  
    
        
    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface  = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(torch.arange(1, self.forecast_length+1)*step, self.deltaTDivisor , rounding_mode='trunc')
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__(): 
                target_idx = self.__len__() - 1
            
        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input = self.zarr_data.isel(time=global_idx)
        targets = self.zarr_data.isel(time=target_file_idxs)
            
        target_list = []
        
        if self.rank % 2 == 0: # first half of data
            # Input data
            # Stack and convert to numpy array
            variables = input['fields']
            variables = variables[:36, :self.xlat]
            # normalize
            variables[:4] = (variables[:4] - self.s_mean) / self.s_std
            variables[4:] = (variables[4:] - self.p_mean[:32]) / self.p_std[:32]
            print(f' I am rank {self.rank}! with var shape {variables.shape} !!!!!!!!!!')
            # Output data
            for i in range(self.forecast_length):
                target = targets.isel(time=i)['fields'].values
                target = target[:36, :self.xlat]
                # normalize
                target[:4] = (target[:4] - self.s_mean) / self.s_std
                target[4:] = (target[4:] - self.p_mean[:32]) / self.p_std[:32]
                
                target_list.append(torch.tensor(target).permute(1, 2, 0))
            variables = torch.tensor(variables).permute(1,2,0)
        else:
            # Input data
            # Stack and convert to numpy array
            variables = input['fields'].values
            variables = variables[36:, :self.xlat]
            # normalize
            variables = (variables - self.p_mean[32:]) / self.p_std[32:]
            variables = np.concatenate([variables, self.land_mask, self.soil_type, self.topography], axis=0)
            
            for i in range(self.forecast_length):
                target = targets.isel(time=i)['fields'].values
                target = target[36:, :self.xlat]
                # normalize
                target = (target - self.p_mean[32:]) / self.p_std[32:]
                target = torch.tensor(target)
                target = torch.nn.functional.pad(target, (0,0,0,0,0,3)).permute(1, 2, 0)
                target_list.append(target)
                            
            variables = torch.tensor(variables).permute(1,2,0)
            
        return variables, target
    
def load_constant_mask_2d(
    patch_size,
    folder_path="/hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/constant_masks/",
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


    land_mask = land_mask[:xlat]
    soil_type = soil_type[:xlat]
    topography = topography[:xlat]

    
    return land_mask, soil_type, topography

class DummyHighResDataset(Dataset):
    def __init__(self, config, nvars=72, nsamples=100):
        self.x = config['data']['xlat']
        self.y = config['data']['xlon']
        self.nvars = nvars
        self.seed = 1 # random seed for now
        self.config = config
        self.nsamples = nsamples
    
    def __len__(self):
        if self.config['data']['train_subset'] == -1:
            return self.nsamples
        else:
            return self.config['data']['train_subset']
    
    def __getitem__(self, global_idx):
        return torch.rand((self.nvars, self.x, self.y), dtype=torch.float32), torch.rand((self.nvars, self.x, self.y), dtype=torch.float32)
    
    
class DistributedDummyHighResDataset(DummyHighResDataset):
    def __init__(self, config, nvars=72):
        super().__init__(config)
        self.parallelism = config['model_parallel']['parallelism']
        self.y = -(-self.y//(self.parallelism//2))
        self.nvars = -(-nvars//2) # ceiling function
        self.seed = 1 # random seed for now
        print(self.x, self.y, self.nvars)

    def __getitem__(self, global_idx):
        return torch.rand((self.x, self.y, self.nvars), dtype=torch.float32), torch.rand((self.x, self.y, self.nvars), dtype=torch.float32)

class DistributedHighResDataset(ERA5DatasetDistributed):
    def __init__(self, config, file_path, mode, patch_size, rank, forecast_length=1, normalize=True, dt=None):
        super().__init__(config, file_path, mode, patch_size, rank, forecast_length=1, normalize=True, dt=None)
        
    
    def __getitem__(self, global_idx):
        input, target = super().__getitem__(global_idx)
        input = input.permute(2, 0, 1).unsqueeze(0)

        input_up  = torch.nn.functional.interpolate(input, scale_factor=(2,2), mode='bilinear', align_corners=True)        

        for i, t in enumerate(target):
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, scale_factor=(2,2), mode='bilinear', align_corners=True)
            target[i] = t.squeeze(0).permute(1,2,0)
        input_up =  input_up.squeeze(0).permute(1,2,0)
        return input_up, target
    
            
