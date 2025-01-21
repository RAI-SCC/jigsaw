import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

class ERA5Dataset(Dataset):
    """Define 3D dataset."""
    def __init__(self, config, file_path, mode, patch_size, forecast_length=1, normalize=True):
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
        self.dt = config['data']['dt']
        self.lite = config['data']['lite']
        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':        
            self.deltaTDivisor = 6
        else:
            if self.lite:
                self.deltaTDivisor = 6
            else:
                self.deltaTDivisor = 1
        self.forecast_length = forecast_length
        self.normalize = normalize
        
        self._get_files_stats(mode=self.mode)
        
        # If data is downloaded from weatherbench, the 6-hourly subsampled data is stored in the reverse pressure level order.
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
            self.level_ordering = range(13-1, -1, -1)
        elif config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
            self.level_ordering = range(0, 13)
        else: # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13-1, -1, -1)

        self.p_mean = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/pressure_zarr.npy', allow_pickle=True)[0].reshape(5, 13, 1, 1)
        self.p_std  = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/pressure_zarr.npy', allow_pickle=True)[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/surface_zarr.npy', allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/surface_zarr.npy', allow_pickle=True)[1].reshape(4, 1, 1)
        
        self.patch_size = patch_size
        self.xlat = config['data']['xlat']
        
    def _get_files_stats(self, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if mode == 'train': # training case, lite
            if self.lite:
                train_years = times[(times.year<2017) & (times.year > 2006) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))] 
            else:
                train_years = times[(times.year<2017) & (times.year > 1979)]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == 'validation':           # validation
            if self.lite:
                validation_years = times[(times.year == 2018) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            else:
                validation_years = times[(times.year == 2018)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'testing':           # validation
            validation_years = times[(times.year == 2020) | (times.year == 2021)] 
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == 'verify':
            validation_years = times[(times.year == 2020) & (times.month == 1) & (times.day == 1) & (times.hour == 0)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
            
        self.n_samples_total = len(self.zarr_data['time'])
        self.img_shape_x     = len(self.zarr_data['latitude'])
        self.img_shape_y     = len(self.zarr_data['longitude'])
        
        # N channels
        self.n_in_channels   = len(self.zarr_data.data_vars)
        
        
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  # -1 to avoid last data point
    
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
