import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import os
from multiprocessing import Value
import threading

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


class ERA5DatasetDistributed(Dataset):
    """Define 3D dataset."""
    def __init__(self, config, file_path, mode, patch_size, rank, forecast_length=1, normalize=True, batch_counter=None):
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
        self.rank = rank

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

        self.p_mean = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/pressure_zarr.npy', allow_pickle=True)[0].reshape(1, 65, 1, 1)
        self.p_std  = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/pressure_zarr.npy', allow_pickle=True)[1].reshape(1, 65, 1, 1)
        self.s_mean = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/surface_zarr.npy', allow_pickle=True)[0].reshape(1, 4, 1, 1)
        self.s_std  = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/surface_zarr.npy', allow_pickle=True)[1].reshape(1, 4, 1, 1)
        
        self.patch_size = patch_size
        self.xlat, self.xlon = config['data']['xlat'], config['data']['xlon']
        self.batch_size = config['training']['train_batch']

        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size, xlat=self.xlat)
        
        self.split = config['model_parallel']['split']
        if self.split == 4:
            if self.rank in [0, 1]:
                self.start, self.end = 0, 720
            else:
                self.start, self.end = 720, 1440

        else: 
            self.start, self.end = 0, 1440
        self.alt_sender = config['model_parallel']['alt_sender']
        self.land_mask = torch.unsqueeze(self.land_mask[:,self.start:self.end], 0).unsqueeze(0)
        self.soil_type = torch.unsqueeze(self.soil_type[:,self.start:self.end], 0).unsqueeze(0)
        self.topography = torch.unsqueeze(self.topography[:,self.start:self.end], 0).unsqueeze(0)

        
    def update_index_order(self, index_order):
        self.index_order = index_order
        
    def find_batch_index(self,idx):
        
        for i, batch in enumerate(self.index_order):
            if idx in batch:
                return i
        return -1  

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
        #if self.split == 2:
        #    if self.alt_sender:
        #        worker_info = torch.utils.data.get_worker_info()
        #        
        #        if worker_info is not None:
        #            # Get the worker's unique counter
        #            batch_counter = worker_state["batch_counter"]
        #            with batch_counter.get_lock():  # Thread-safe counter access
        #                current_batch = batch_counter.value
        #                # Alternate source ranks based on the worker's counter
#       #                 print(f'{self.rank} current_batch {current_batch}, ')
        #                skipped_ranks = [2,3] if current_batch % (2*self.batch_size) < self.batch_size else [0,1]
        #                #skipped_ranks = [2,3] if current_batch % 2 < self.batch_size else [0,1]  
        #                batch_counter.value += 1  # Increment worker's counter
        #                
        #        else:
        #            # No multiprocessing (e.g., num_workers=0), use a default
        #            current_batch = 0
        #    else: 
        #        # If two-way split, half of the ranks should not read any of the data
        #        
        #        skipped_ranks = [2, 3]    
        #    if self.rank in skipped_ranks: # 0, 1 always senders
        #        return [torch.zeros(1), torch.zeros(1)], [torch.zeros(1), torch.zeros(1)]
        if self.split == 2:
            if not self.alt_sender:
                skipped_ranks = [2,3]
            else: 
                skipped_ranks = []
            if self.rank in skipped_ranks:
                return [torch.zeros(1), torch.zeros(1)], [torch.zeros(1), torch.zeros(1)]
        worker_info = torch.utils.data.get_worker_info()
        
        if global_idx.numel() == 0:
            return [torch.zeros(1), torch.zeros(1)], [torch.zeros(1), torch.zeros(1)]
        
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

        if self.rank in [0, 2]: # first half of data
            # Input data
            # Stack and convert to numpy array


            geopotential = input_pressure['geopotential'][:,:,:self.xlat,self.start:self.end].values
            humidity = input_pressure['specific_humidity'][:,:,:self.xlat,self.start:self.end].values
            temperature = input_pressure['temperature'][:,:6,:self.xlat,self.start:self.end].values
            
            mslp = input_surface['mean_sea_level_pressure'][:,:self.xlat,self.start:self.end].values
            u10  = input_surface['10m_u_component_of_wind'][:,:self.xlat,self.start:self.end].values
            v10  = input_surface['10m_v_component_of_wind'][:,:self.xlat,self.start:self.end].values
            t2m  = input_surface['2m_temperature'][:,:self.xlat,self.start:self.end].values
            
        
            input_pressure = np.concatenate([geopotential, humidity, temperature], axis=1)
            input_surface  = np.stack([mslp, u10, v10, t2m], axis=1)
            input_pressure = (input_pressure - self.p_mean[0,:32]) / self.p_std[0,:32]
            input_surface  = (input_surface  - self.s_mean) / self.s_std
            input_pressure = np.reshape(input_pressure, (input_surface.shape[0], -1, self.xlat, self.end-self.start))
            
            input = np.concatenate([input_surface, input_pressure], axis=1)
            
            # Output data
            for i in range(self.forecast_length):

                pressure_ds = target_pressure_ds
                surface_ds  = target_surface_ds
                
                geopotential = pressure_ds['geopotential'][:,:,:self.xlat,self.start:self.end].values
                humidity = pressure_ds['specific_humidity'][:,:,:self.xlat,self.start:self.end].values
                temperature = pressure_ds['temperature'][:,:6,:self.xlat,self.start:self.end].values
                
                mslp = surface_ds['mean_sea_level_pressure'][:,:self.xlat,self.start:self.end].values
                u10  = surface_ds['10m_u_component_of_wind'][:,:self.xlat,self.start:self.end].values
                v10  = surface_ds['10m_v_component_of_wind'][:,:self.xlat,self.start:self.end].values
                t2m  = surface_ds['2m_temperature'][:,:self.xlat,self.start:self.end].values
                temp = np.concatenate([geopotential, humidity, temperature], axis=1)
                target_surface.append(np.stack([mslp, u10, v10, t2m], axis=1))

                temp = (temp - self.p_mean[:,:32]) / self.p_std[:,:32]
                target_surface[i]  = (target_surface[i] - self.s_mean) / self.s_std
                #temp = temp.reshape(-1, self.xlat, self.end-self.start)
                
                temp = np.concatenate([target_surface[i], temp], axis=1)
                if self.split == 2:
                    target.append([torch.tensor(temp[:,:,:,:720]), torch.tensor(temp[:,:,:,720:])])
                else:
                    target.append(torch.tensor(temp)) 
        else:
            temperature = input_pressure['temperature'][:,6:,:self.xlat,self.start:self.end].values
            u_velocity  = input_pressure['u_component_of_wind'][:,:,:self.xlat,self.start:self.end].values
            v_velocity = input_pressure['v_component_of_wind'][:,:,:self.xlat,self.start:self.end].values
        
            
            input = np.concatenate([temperature, u_velocity, v_velocity], axis=1)
            
            input = (input - self.p_mean[:,32:]) / self.p_std[:,32:]
            input = np.reshape(input, (temperature.shape[0], -1, self.xlat, self.end-self.start))
            
            self.land_mask = self.land_mask.broadcast_to(input.shape[0], *self.land_mask.shape[1:])
            self.soil_type = self.land_mask.broadcast_to(input.shape[0], *self.land_mask.shape[1:])
            self.topography = self.land_mask.broadcast_to(input.shape[0], *self.land_mask.shape[1:])
            input = np.concatenate([input, self.land_mask, self.soil_type, self.topography], axis=1)

            for i in range(self.forecast_length):

                pressure_ds = target_pressure_ds
                temperature = pressure_ds['temperature'][:,6:,:self.xlat,self.start:self.end].values
                u_velocity  = pressure_ds['u_component_of_wind'][:,:,:self.xlat,self.start:self.end].values
                v_velocity = pressure_ds['v_component_of_wind'][:,:,:self.xlat,self.start:self.end].values
                
                temp = np.concatenate([temperature, u_velocity, v_velocity], axis=1)
                
                temp = (temp - self.p_mean[:,32:]) / self.p_std[:,32:]
                
                temp = temp.reshape(temperature.shape[0], -1, self.xlat, self.end-self.start) 
                
                if self.split == 2:
                    target.append([torch.nn.functional.pad(torch.tensor(temp[:,:,:,:720]), (0, 0, 0, 0, 0, 3)), 
                                   torch.nn.functional.pad(torch.tensor(temp[:,:,:,720:]), (0, 0, 0, 0, 0, 3))])
                else:
                    target.append(torch.nn.functional.pad(torch.tensor(temp), (0, 0, 0, 0, 0, 3))) 
                    
                
        if self.split == 2:
            input = [torch.tensor(input[:,:,:,:720]), torch.tensor(input[:,:,:,720:])]
        else:
            input = torch.tensor(input)

                        
        return input, target    



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


    land_mask = land_mask[:xlat]
    soil_type = soil_type[:xlat]
    topography = topography[:xlat]

    
    return land_mask, soil_type, topography

# Worker initialization function
def worker_init_fn(worker_id, shared_batch_counter):
    global worker_state
    worker_state = {
        "batch_counter": shared_batch_counter  # Assign the shared counter to all workers
    }
    #worker_info = torch.utils.data.get_worker_info()
    
# Worker state is stored globally
worker_state = {"batch_counters": {}}