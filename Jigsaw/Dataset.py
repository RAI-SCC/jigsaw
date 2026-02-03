
import os
from datetime import datetime
import yaml
import csv

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        forecast_length=1,
        normalize=True,
        rank=None,
    ):
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
        self.dt = config["data"]["dt"]
        if (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
        ):
            self.deltaTDivisor = 6
        elif (
            config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5_subset.zarr"
        ):
            self.deltaTDivisor = 6
        else:
            self.deltaTDivisor = 6
        self.forecast_length = forecast_length
        self.normalize = normalize

        self._get_files_stats(mode=self.mode)

        # If data is downloaded from weatherbench, the 6-hourly subsampled data is stored in the reverse pressure level order.
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
            or config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5_subset.zarr"
        ):
            self.level_ordering = range(13 - 1, -1, -1)
        elif (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr"
            or config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr"
        ):
            self.level_ordering = range(0, 13)
        else:  # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13 - 1, -1, -1)

        self.p_mean = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[0].reshape(5, 13, 1, 1)
        self.p_std = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[0].reshape(4, 1, 1)
        self.s_std = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[1].reshape(4, 1, 1)

        self.patch_size = patch_size
        self.xlat = config["data"]["xlat"]
        self.rank = rank
        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(
            config["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(self.land_mask, 0)
        self.soil_type = torch.unsqueeze(self.soil_type, 0)
        self.topography = torch.unsqueeze(self.topography, 0)

    def _get_files_stats(self, mode="train"):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(self.zarr_data["time"].values)
        if mode == "train":  # training case, lite
            train_years = times[
                (times.year < 2017)
                & (times.year > 1979)
                & (
                    (times.hour == 0)
                    | (times.hour == 6)
                    | (times.hour == 12)
                    | (times.hour == 18)
                )
            ]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == "validation":
            validation_years = times[(times.year == 2018)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == "testing":
            validation_years = times[(times.year == 2020) | (times.year == 2021)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)

        self.n_samples_total = len(self.zarr_data["time"])

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total
            - self.forecast_length * self.dt // self.deltaTDivisor
            - 1
        )  # -1 to avoid last data point

    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_pressure = []
        target_surface = []
        step = self.dt

        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(
            torch.arange(1, self.forecast_length + 1) * step,
            self.deltaTDivisor,
            rounding_mode="trunc",
        )

        for target_idx in target_file_idxs:
            if target_idx >= self.__len__():
                target_idx = self.__len__() - 1

        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(
            time=global_idx, level=self.level_ordering
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        input_surface = self.zarr_data.isel(time=global_idx)[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]
        target_pressure_ds = self.zarr_data.isel(
            time=target_file_idxs, level=self.level_ordering
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        target_surface_ds = self.zarr_data.isel(time=target_file_idxs)[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]
        # Stack and convert to numpy array
        input_pressure = np.stack(
            [
                input_pressure["geopotential"].values,
                input_pressure["specific_humidity"].values,
                input_pressure["temperature"].values,
                input_pressure["u_component_of_wind"].values,
                input_pressure["v_component_of_wind"].values,
            ],
            axis=0,
        )
        input_surface = np.stack(
            [
                input_surface["mean_sea_level_pressure"].values,
                input_surface["10m_u_component_of_wind"].values,
                input_surface["10m_v_component_of_wind"].values,
                input_surface["2m_temperature"].values,
            ],
            axis=0,
        )

        input_pressure = (input_pressure - self.p_mean) / self.p_std
        input_surface = (input_surface - self.s_mean) / self.s_std
        input_pressure = np.reshape(input_pressure, (-1, 721, 1440))
        input_surface = input_surface[:, : self.xlat]
        input_pressure = input_pressure[:, : self.xlat]

        input = np.concatenate(
            [
                input_surface,
                input_pressure,
                self.land_mask,
                self.soil_type,
                self.topography,
            ],
            axis=0,
        )

        target = []
        for i in range(self.forecast_length):
            pressure_ds = target_pressure_ds.isel(time=i)
            surface_ds = target_surface_ds.isel(time=i)
            target_pressure.append(
                np.stack(
                    [
                        pressure_ds["geopotential"].values,
                        pressure_ds["specific_humidity"].values,
                        pressure_ds["temperature"].values,
                        pressure_ds["u_component_of_wind"].values,
                        pressure_ds["v_component_of_wind"].values,
                    ],
                    axis=0,
                )
            )
            target_surface.append(
                np.stack(
                    [
                        surface_ds["mean_sea_level_pressure"].values,
                        surface_ds["10m_u_component_of_wind"].values,
                        surface_ds["10m_v_component_of_wind"].values,
                        surface_ds["2m_temperature"].values,
                    ],
                    axis=0,
                )
            )
            target_pressure[i] = (target_pressure[i] - self.p_mean) / self.p_std
            target_surface[i] = (target_surface[i] - self.s_mean) / self.s_std
            target_pressure[i], target_surface[i] = (
                torch.tensor(target_pressure[i]),
                torch.tensor(target_surface[i]),
            )

            target_pressure[i] = target_pressure[i].reshape(-1, 721, 1440)
            target_pressure[i] = target_pressure[i][:, : self.xlat]
            target_surface[i] = target_surface[i][:, : self.xlat]
            target.append(
                torch.concat([target_surface[i], target_pressure[i]], axis=0).permute(
                    1, 2, 0
                )
            )

        input = torch.tensor(input)

        return input, target


class ERA5DatasetDistributed(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
        normalize=True,
        dt=None,
    ):
        """
        Initialize.

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
        self.dt = dt if dt is not None else config["data"]["dt"]
        self.rank = rank
        self.datapath = config["data"]["train_data_path"]

        if (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
        ):
            self.deltaTDivisor = 6
        elif (
            config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_dataset/era5.zarr"
        ):
            self.deltaTDivisor = 1
        elif (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr"
        ):
            self.deltaTDivisor = 1
        else:
            self.deltaTDivisor = 1

        self.forecast_length = forecast_length
        self.normalize = normalize

        self._get_files_stats(mode=self.mode)

        # Current copies of subsampled WeatherBench2 data saves in reverse pressure-level order
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
        ):
            self.level_ordering = range(13 - 1, -1, -1)
        elif (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr"
        ):
            self.level_ordering = range(0, 13)
        elif (
            config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr"
        ):
            self.level_ordering = range(0, 13)
        else:  # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13 - 1, -1, -1)

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[0].reshape(65, 1, 1)
        self.p_std = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[1].reshape(65, 1, 1)
        self.s_mean = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[0].reshape(4, 1, 1)
        self.s_std = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[1].reshape(4, 1, 1)

        self.patch_size = patch_size
        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        self.batch_size = config["training"]["train_batch"]
        self.parallelism = config["model_parallel"]["parallelism"]

        if self.parallelism == 2:
            self.start, self.end = 0, self.xlon
        elif self.parallelism == 4:
            if self.rank % 4 in [0, 1]:
                self.start, self.end = 0, self.xlon // 2
            else:
                self.start, self.end = self.xlon // 2, self.xlon
        elif self.parallelism == 8:
            if self.rank % 4 in [0, 1]:
                self.start, self.end = 0, self.xlon // 4
            elif self.rank % 4 in [2, 3]:
                self.start, self.end = self.xlon // 4, self.xlon // 2
            elif self.rank % 4 in [4, 5]:
                self.start, self.end = self.xlon // 2, self.xlon * 3 // 4
            elif self.rank % 4 in [6, 7]:
                self.start, self.end = self.xlon * 3 // 4, self.xlon

        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(
            config["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(self.land_mask[:, self.start : self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:, self.start : self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:, self.start : self.end], 0)

    def _get_files_stats(self, mode="train"):
        """Filter desired time points based on parameters and return file statistics."""
        self.zarr_data = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(self.zarr_data["time"].values)
        if mode == "train":  # training case, lite
            if (
                self.datapath
                == "/hkfs/work/workspace/scratch/ke4365-era5_dataset/era5.zarr"
            ):
                train_years = times[(times.year < 2018) & (times.year > 1979)]
            else:
                train_years = times[
                    (times.year < 2018)
                    & (times.year > 1979)
                    & (
                        (times.hour == 0)
                        | (times.hour == 6)
                        | (times.hour == 12)
                        | (times.hour == 18)
                    )
                ]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == "validation":  # validation
            validation_years = times[(times.year == 2018)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == "testing":  # validation
            validation_years = times[(times.year == 2020) | (times.year == 2021)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)

        self.n_samples_total = len(self.zarr_data["time"])

        # N channels
        self.n_in_channels = len(self.zarr_data.data_vars)

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor
        )

    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(
            torch.arange(1, self.forecast_length + 1) * step,
            self.deltaTDivisor,
            rounding_mode="trunc",
        )
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__():
                target_idx = self.__len__() - 1

        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(
            time=global_idx, level=self.level_ordering
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        input_surface = self.zarr_data.isel(time=global_idx)[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]
        target_pressure_ds = self.zarr_data.isel(
            time=target_file_idxs, level=self.level_ordering
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        target_surface_ds = self.zarr_data.isel(time=target_file_idxs)[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        target = []

        if self.rank % 2 == 0:  # first half of data
            # Input data
            # Stack and convert to numpy array
            input_pressure = (
                input_pressure[["geopotential", "specific_humidity", "temperature"]]
                .isel(
                    latitude=range(0, self.xlat), longitude=range(self.start, self.end)
                )
                .to_array("pressure_variables")
                .values
            ).reshape(-1, len(range(0, self.xlat)), len(range(self.end - self.start)))[
                :32
            ]

            input_surface = (
                input_surface[
                    [
                        "mean_sea_level_pressure",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_temperature",
                    ]
                ]
                .isel(
                    latitude=range(0, self.xlat), longitude=range(self.start, self.end)
                )
                .to_array("surface_variables")
                .values
            )

            input_pressure = (input_pressure - self.p_mean[:32]) / self.p_std[:32]
            input_surface = (input_surface - self.s_mean) / self.s_std

            input = np.concatenate([input_surface, input_pressure], axis=0)

            # Output data
            for i in range(self.forecast_length):
                pressure_ds = target_pressure_ds.isel(time=i)
                surface_ds = target_surface_ds.isel(time=i)

                temp = (
                    pressure_ds[["geopotential", "specific_humidity", "temperature"]]
                    .isel(
                        latitude=range(0, self.xlat),
                        longitude=range(self.start, self.end),
                    )
                    .to_array("pressure_variables")
                    .values
                ).reshape(
                    -1, len(range(0, self.xlat)), len(range(self.end - self.start))
                )[:32]

                target_surface.append(
                    surface_ds[
                        [
                            "mean_sea_level_pressure",
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            "2m_temperature",
                        ]
                    ]
                    .isel(
                        latitude=range(0, self.xlat),
                        longitude=range(self.start, self.end),
                    )
                    .to_array("surface_variables")
                    .values
                )

                temp = (temp - self.p_mean[:32]) / self.p_std[:32]
                target_surface[i] = (target_surface[i] - self.s_mean) / self.s_std

                temp = np.concatenate([target_surface[i], temp], axis=0)
                target.append(torch.tensor(temp))
        else:
            input = (
                input_pressure[
                    ["temperature", "u_component_of_wind", "v_component_of_wind"]
                ]
                .isel(
                    latitude=range(0, self.xlat), longitude=range(self.start, self.end)
                )
                .to_array("pressure_variables")
                .values
            ).reshape(-1, len(range(0, self.xlat)), len(range(self.end - self.start)))[
                6:
            ]

            input = (input - self.p_mean[32:]) / self.p_std[32:]

            input = np.concatenate(
                [input, self.land_mask, self.soil_type, self.topography], axis=0
            )

            for i in range(self.forecast_length):
                pressure_ds = target_pressure_ds.isel(time=i)

                temp = (
                    pressure_ds[
                        ["temperature", "u_component_of_wind", "v_component_of_wind"]
                    ]
                    .isel(
                        latitude=range(0, self.xlat),
                        longitude=range(self.start, self.end),
                    )
                    .to_array("pressure_variables")
                    .values
                ).reshape(
                    -1, len(range(0, self.xlat)), len(range(self.end - self.start))
                )[6:]

                temp = (temp - self.p_mean[32:]) / self.p_std[32:]

                target.append(
                    torch.nn.functional.pad(
                        torch.tensor(temp), (0, 0, 0, 0, 0, 3)
                    )
                )

        input = torch.tensor(input)

        return input, target


class ERA5Preprocessed(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
        normalize=True,
        dt=None,
    ):
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
        self.dt = dt if dt is not None else config["data"]["dt"]
        self.rank = rank
        self.datapath = config["data"]["train_data_path"]

        if (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
        ):
            self.deltaTDivisor = 6
        elif (
            config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr"
        ):
            self.deltaTDivisor = 1
        elif (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr"
        ):
            self.deltaTDivisor = 1
        else:
            self.deltaTDivisor = 6

        self.forecast_length = forecast_length
        self.normalize = normalize

        self._get_files_stats(mode=self.mode)

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[0].reshape(65, 1, 1)
        self.p_std = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[1].reshape(65, 1, 1)
        self.s_mean = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[0].reshape(4, 1, 1)
        self.s_std = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[1].reshape(4, 1, 1)

        self.patch_size = patch_size
        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        self.batch_size = config["training"]["train_batch"]
        self.parallelism = config["model_parallel"]["parallelism"]

        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(
            config["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(self.land_mask, 0)
        self.soil_type = torch.unsqueeze(self.soil_type, 0)
        self.topography = torch.unsqueeze(self.topography, 0)

    def _get_files_stats(self, mode="train"):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(self.zarr_data["time"].values)
        if mode == "train":  # training case, lite
            if (
                self.datapath
                == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr"
            ):
                train_years = times[(times.year < 2018) & (times.year > 1979)]
            else:
                train_years = times[
                    (times.year < 2018)
                    & (times.year > 1979)
                    & (
                        (times.hour == 0)
                        | (times.hour == 6)
                        | (times.hour == 12)
                        | (times.hour == 18)
                    )
                ]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == "validation":  # validation
            if (
                self.datapath
                == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr"
            ):
                validation_years = times[(times.year == 2018)]
            else:
                validation_years = times[
                    (times.year < 2018)
                    & (
                        (times.hour == 0)
                        | (times.hour == 6)
                        | (times.hour == 12)
                        | (times.hour == 18)
                    )
                ]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == "testing":  # validation
            validation_years = times[(times.year == 2020) | (times.year == 2021)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == "verify":
            validation_years = times[
                (times.year == 2020)
                & (times.month == 1)
                & (times.day == 1)
                & (times.hour == 0)
            ]
            self.zarr_data = self.zarr_data.sel(time=validation_years)

        self.n_samples_total = len(self.zarr_data["time"])

        # N channels
        self.n_in_channels = len(self.zarr_data.data_vars)

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor
        )

    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(
            torch.arange(1, self.forecast_length + 1) * step,
            self.deltaTDivisor,
            rounding_mode="trunc",
        )
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__():
                target_idx = self.__len__() - 1

        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input = self.zarr_data.isel(time=global_idx)
        targets = self.zarr_data.isel(time=target_file_idxs)

        target_list = []

        # Input data
        # Stack and convert to numpy array
        variables = input["fields"][:, : self.xlat].values
        # normalize
        variables[:4] = (variables[:4] - self.s_mean) / self.s_std
        variables[4:] = (variables[4:] - self.p_mean) / self.p_std
        variables = np.concatenate(
            [variables, self.land_mask, self.soil_type, self.topography], axis=0
        )
        # Output data
        for i in range(self.forecast_length):
            target = targets.isel(time=i)["fields"][:, : self.xlat].values
            # normalize
            target[:4] = (target[:4] - self.s_mean) / self.s_std
            target[4:] = (target[4:] - self.p_mean) / self.p_std

            target_list.append(torch.tensor(target).permute(1, 2, 0))

        variables = torch.tensor(variables)

        return variables, target_list


class ERA5PreprocessedDistributed(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
        normalize=True,
        dt=None,
    ):
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
        self.dt = dt if dt is not None else config["data"]["dt"]
        self.rank = rank
        self.datapath = config["data"]["train_data_path"]

        if (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
        ):
            self.deltaTDivisor = 6
        elif (
            config["data"]["train_data_path"]
            == "/hkfs/work/workspace/scratch/ke4365-era5_dataset/era5.zarr"
        ):
            self.deltaTDivisor = 1
        elif (
            config["data"]["train_data_path"]
            == "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr"
        ):
            self.deltaTDivisor = 1
        else:
            self.deltaTDivisor = 6

        self.forecast_length = forecast_length
        self.normalize = normalize

        self._get_files_stats(mode=self.mode)

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[0].reshape(65, 1, 1)
        self.p_std = np.load(
            f"{config['data']['constant_masks_path']}/pressure_zarr.npy",
            allow_pickle=True,
        )[1].reshape(65, 1, 1)
        self.s_mean = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[0].reshape(4, 1, 1)
        self.s_std = np.load(
            f"{config['data']['constant_masks_path']}/surface_zarr.npy",
            allow_pickle=True,
        )[1].reshape(4, 1, 1)

        self.patch_size = patch_size
        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        self.batch_size = config["training"]["train_batch"]
        self.parallelism = config["model_parallel"]["parallelism"]

        if self.parallelism == 2:
            self.start, self.end = 0, self.xlon
        elif self.parallelism == 4:
            if self.rank % 4 in [0, 1]:
                self.start, self.end = 0, self.xlon // 2
            else:
                self.start, self.end = self.xlon // 2, self.xlon
        elif self.parallelism == 8:
            if self.rank % 4 in [0, 1]:
                self.start, self.end = 0, self.xlon // 4
            elif self.rank % 4 in [2, 3]:
                self.start, self.end = self.xlon // 4, self.xlon // 2
            elif self.rank % 4 in [4, 5]:
                self.start, self.end = self.xlon // 2, self.xlon * 3 // 4
            elif self.rank % 4 in [6, 7]:
                self.start, self.end = self.xlon * 3 // 4, self.xlon

        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(
            config["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(self.land_mask[:, self.start : self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:, self.start : self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:, self.start : self.end], 0)

    def _get_files_stats(self, mode="train"):
        """Filter desired time points based on parameters and return file statistics."""

        self.zarr_data = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(self.zarr_data["time"].values)
        if mode == "train":  # training case, lite
            if (
                self.datapath
                == "/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr"
            ):
                train_years = times[(times.year < 2018) & (times.year > 1979)]
            else:
                train_years = times[
                    (times.year < 2018)
                    & (times.year > 1979)
                    & (
                        (times.hour == 0)
                        | (times.hour == 6)
                        | (times.hour == 12)
                        | (times.hour == 18)
                    )
                ]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == "validation":  # validation
            if (
                self.datapath
                == "/hkfs/work/workspace/scratch/ke4365-era5_dataset/era5.zarr"
            ):
                validation_years = times[(times.year == 2018)]
            else:
                validation_years = times[
                    (times.year < 2018)
                    & (
                        (times.hour == 0)
                        | (times.hour == 6)
                        | (times.hour == 12)
                        | (times.hour == 18)
                    )
                ]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == "testing":  # validation
            validation_years = times[(times.year == 2020) | (times.year == 2021)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif mode == "verify":
            validation_years = times[
                (times.year == 2020)
                & (times.month == 1)
                & (times.day == 1)
                & (times.hour == 0)
            ]
            self.zarr_data = self.zarr_data.sel(time=validation_years)

        self.n_samples_total = len(self.zarr_data["time"])

        # N channels
        self.n_in_channels = len(self.zarr_data.data_vars)

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor
        )

    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(
            torch.arange(1, self.forecast_length + 1) * step,
            self.deltaTDivisor,
            rounding_mode="trunc",
        )
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__():
                target_idx = self.__len__() - 1

        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input = self.zarr_data.isel(time=global_idx)
        targets = self.zarr_data.isel(time=target_file_idxs)

        target_list = []

        if self.rank % 2 == 0:  # first half of data
            # Input data
            # Stack and convert to numpy array
            variables = input["fields"][:36, : self.xlat, self.start : self.end].values
            # normalize
            variables[:4] = (variables[:4] - self.s_mean) / self.s_std
            variables[4:] = (variables[4:] - self.p_mean[:32]) / self.p_std[:32]
            # Output data
            for i in range(self.forecast_length):
                target = targets.isel(time=i)["fields"][
                    :36, : self.xlat, self.start : self.end
                ].values
                # normalize
                target[:4] = (target[:4] - self.s_mean) / self.s_std
                target[4:] = (target[4:] - self.p_mean[:32]) / self.p_std[:32]

                target_list.append(torch.tensor(target).permute(1, 2, 0))

        else:
            # Input data
            # Stack and convert to numpy array
            variables = input["fields"][36:, : self.xlat, self.start : self.end].values
            # normalize
            variables = (variables - self.p_mean[32:]) / self.p_std[32:]
            variables = np.concatenate(
                [variables, self.land_mask, self.soil_type, self.topography], axis=0
            )

            for i in range(self.forecast_length):
                target = targets.isel(time=i)["fields"][
                    36:, : self.xlat, self.start : self.end
                ].values
                # normalize
                target = (target - self.p_mean[32:]) / self.p_std[32:]
                target = torch.tensor(target)
                target = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, 3))
                target_list.append(target.permute(1, 2, 0))

        variables = torch.tensor(variables).permute(1, 2, 0)

        return variables, target_list


def load_constant_mask_2d(folder_path, xlat=721):
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
    assert land_mask.shape == soil_type.shape == topography.shape, (
        "Shapes of the three constant masks are not equal."
    )

    land_mask = land_mask[:xlat]
    soil_type = soil_type[:xlat]
    topography = topography[:xlat]

    return land_mask, soil_type, topography


class ERAJigsaw(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
    ):
        """
        Initialize.

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
        self.config = config
        self.mode = mode
        self.rank = rank
        if mode == "train":
            self.file_path = self.config["data"]["train_data_path"]
            self.time_range = self.config["data"]["train_time_range"]
        elif mode == "validation":
            self.file_path = self.config["data"]["valid_data_path"]
            self.time_range = self.config["data"]["valid_time_range"]
        elif mode == "test":
            self.file_path = self.config["data"]["valid_data_path"]
            self.time_range = self.config["data"]["test_time_range"]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.dt = self.config["data"]["dt"]
        self.deltaTDivisor = self.config["data"]["dataset_subsampling"]

        self.patch_size = (config["model"]["patch_size"], config["model"]["patch_size"])
        self.resolution = (self.config["data"]["xlat"], self.config["data"]["xlon"])
        self.padded_resolution = (
            self.resolution[0]
            + (self.patch_size[0] - self.resolution[0] % self.patch_size[0])
            % self.patch_size[0],
            self.resolution[1]
            + (self.patch_size[1] - self.resolution[1] % self.patch_size[1])
            % self.patch_size[1],
        )
        self.padding = (
            (
                (self.padded_resolution[0] - self.resolution[0]) // 2,
                (self.padded_resolution[0] - self.resolution[0])
                - (self.padded_resolution[0] - self.resolution[0]) // 2,
            ),
            (
                (self.padded_resolution[1] - self.resolution[1]) // 2,
                (self.padded_resolution[1] - self.resolution[1])
                - (self.padded_resolution[1] - self.resolution[1]) // 2,
            ),
        )

        self.translation_dict = None
        self.in_surface_variables = self._translate_variable_names(
            self.config["data"]["in_surface_variables"]
        )
        self.in_pressure_variables = self._translate_variable_names(
            self.config["data"]["in_pressure_variables"]
        )
        self.out_surface_variables = self._translate_variable_names(
            self.config["data"]["out_surface_variables"]
        )
        self.out_pressure_variables = self._translate_variable_names(
            self.config["data"]["out_pressure_variables"]
        )
        self.in_pressure_levels = self.config["data"]["in_pressure_levels"]
        self.out_pressure_levels = self.config["data"]["out_pressure_levels"]

        self.forecast_length = self.config["data"]["rollout"]
        self.patch_size = patch_size
        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        self.batch_size = config["training"]["train_batch"]
        self.parallelism = config["model_parallel"]["parallelism"]
        self.uses_constant_masks = len(self.config["data"]["constant_masks"]) > 0

        self._get_spatial_ranges()
        
        if self.uses_constant_masks:
            self._load_constant_masks()
            
        self._get_data_files_stats()
        self._get_normalization_constants()

    def _get_spatial_ranges(self):
        match self.parallelism:
            case 1 | 2:
                self.start, self.end = 0, self.xlon
            case 4:
                if self.rank % 4 in [0, 1]:
                    self.start, self.end = 0, self.xlon // 2
                else:
                    self.start, self.end = self.xlon // 2, self.xlon
            case 8:
                if self.rank % 4 in [0, 1]:
                    self.start, self.end = 0, self.xlon // 4
                elif self.rank % 4 in [2, 3]:
                    self.start, self.end = self.xlon // 4, self.xlon // 2
                elif self.rank % 4 in [4, 5]:
                    self.start, self.end = self.xlon // 2, self.xlon * 3 // 4
                elif self.rank % 4 in [6, 7]:
                    self.start, self.end = self.xlon * 3 // 4, self.xlon
        return

        
    def _translate_variable_names(self, varname_list):
        if self.translation_dict is None:
            self._load_translation_dict()

        return_list = [
            (
                self.translation_dict[varname]
                if varname in self.translation_dict.keys()
                else varname
            )
            for varname in varname_list
        ]

        return return_list

    def _load_translation_dict(self):
        """Load yaml file that translates shorthands to ERA5 dataset variable names."""
        self.translation_dict = {}
        full_path = os.path.join(os.path.dirname(__file__), "variable_aliases.yaml")
        with open(full_path) as f:
            self.translation_dict = yaml.load(f, Loader=yaml.SafeLoader)

    def _get_data_files_stats(self):
        """Filter desired time points based on parameters and return file statistics."""

        dataset = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(dataset["time"].values)

        self.zarr_in = dataset[self.in_pressure_variables]
        self.zarr_out = dataset[self.out_pressure_variables]
        self.zarr_in_surface = dataset[self.in_surface_variables]
        self.zarr_out_surface = dataset[self.out_surface_variables]

        start_date = datetime.strptime(self.time_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(self.time_range[1], "%Y-%m-%d")

        dataset_in_range = times[
            (times.year >= start_date.year)
            & (times.year <= end_date.year)
            & ((times.month >= start_date.month) | (times.year > start_date.year))
            & ((times.month <= end_date.month) | (times.year < end_date.year))
            & (
                (times.day >= start_date.day)
                | (times.year > start_date.year)
                | (times.month > start_date.month)
            )
            & (
                (times.day <= end_date.day)
                | (times.year < end_date.year)
                | (times.month < end_date.month)
            )
            & (times.hour % self.config["data"]["target_subsampling"] == 0)
        ]
        dataset_out_range = times[
            (times.year >= start_date.year)
            & (times.year <= end_date.year)
            & ((times.month >= start_date.month) | (times.year > start_date.year))
            & ((times.month <= end_date.month) | (times.year < end_date.year))
            & (
                (times.day >= start_date.day)
                | (times.year > start_date.year)
                | (times.month > start_date.month)
            )
            & (
                (times.day <= end_date.day)
                | (times.year < end_date.year)
                | (times.month < end_date.month)
            )
            & ((times.hour - self.dt) % self.config["data"]["target_subsampling"] == 0)
        ]

        self.zarr_in = self.zarr_in.sel(time=dataset_in_range)
        self.zarr_in = self.zarr_in.sel(level=self.config["data"]["in_pressure_levels"])
        self.zarr_in = self.zarr_in.transpose("time", "level", "latitude", "longitude")

        self.zarr_out = self.zarr_out.sel(time=dataset_out_range)
        self.zarr_out = self.zarr_out.sel(
            level=self.config["data"]["out_pressure_levels"]
        )
        self.zarr_out = self.zarr_out.transpose(
            "time", "level", "latitude", "longitude"
        )

        self.zarr_in_surface = self.zarr_in_surface.sel(time=dataset_in_range)
        self.zarr_in_surface = self.zarr_in_surface.transpose(
            "time", "latitude", "longitude"
        )

        self.zarr_out_surface = self.zarr_out_surface.sel(time=dataset_out_range)
        self.zarr_out_surface = self.zarr_out_surface.transpose(
            "time", "latitude", "longitude"
        )

        self.n_samples_total = len(self.zarr_in["time"])
        self.img_shape_x = len(self.zarr_in["latitude"])
        self.img_shape_y = len(self.zarr_in["longitude"])

        self.num_masks = len(self.config["data"]["constant_masks"])
        self.n_in_channels = len(self.zarr_in.data_vars) + self.num_masks

    def _get_normalization_constants(self):
        surface_normalizations = []
        with open(
            f"{self.config['data']['normalization_dir']}/surface_norms.csv", "r"
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                surface_normalizations.append([row[0], float(row[1]), float(row[2])])

        in_s_mean_data = []
        in_s_std_data = []
        for variable_name in self.in_surface_variables:
            for row in surface_normalizations:
                if variable_name == row[0]:
                    in_s_mean_data.append(row[1])
                    in_s_std_data.append(row[2])
                    break
            else:
                raise ValueError(
                    f"Surface variable {variable_name} has no stored normalization constants"
                )

        out_s_mean_data = []
        out_s_std_data = []
        for variable_name in self.out_surface_variables:
            for row in surface_normalizations:
                if variable_name == row[0]:
                    out_s_mean_data.append(row[1])
                    out_s_std_data.append(row[2])
                    break
            else:
                raise ValueError(
                    f"Surface variable {variable_name} has no stored normalization constants"
                )

        pressure_normalizations = []
        with open(
            f"{self.config['data']['normalization_dir']}/pressure_norms.csv", "r"
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pressure_normalizations.append([row[0], float(row[1]), float(row[2])])

        in_p_mean_data = []
        in_p_std_data = []
        for variable_name in self.in_pressure_variables:
            for level in self.config["data"]["in_pressure_levels"]:
                combined_variable_name = f"{variable_name}_{level}"
                for row in pressure_normalizations:
                    if combined_variable_name == row[0]:
                        in_p_mean_data.append(row[1])
                        in_p_std_data.append(row[2])
                        break
                else:
                    raise ValueError(
                        f"Variable {variable_name} has no stored normalization constants at pressure level {level}"
                    )

        out_p_mean_data = []
        out_p_std_data = []
        for variable_name in self.out_pressure_variables:
            for level in self.config["data"]["out_pressure_levels"]:
                combined_variable_name = f"{variable_name}_{level}"
                for row in pressure_normalizations:
                    if combined_variable_name == row[0]:
                        out_p_mean_data.append(row[1])
                        out_p_std_data.append(row[2])
                        break
                else:
                    raise ValueError(
                        f"Variable {variable_name} has no stored normalization constants at pressure level {level}"
                    )

        self.in_means = np.array(in_s_mean_data + in_p_mean_data)
        self.in_stds = np.array(in_s_std_data + in_p_std_data)
        self.out_means = np.array(out_s_mean_data + out_p_mean_data)
        self.out_stds = np.array(out_s_std_data + out_p_std_data)
        
        self.total_variables_in  = self._get_num_variables_in()
        self.total_variables_out = self._get_num_variables_out()
        
        match self.rank % 2:
            case 0: # first half of variables
                self.in_means = self.in_means[:self.total_variables_in//2]
                self.in_stds = self.in_stds[:self.total_variables_in//2]
                self.out_means = self.out_means[:self.total_variables_out//2]
                self.out_stds = self.out_stds[:self.total_variables_out//2]
            case 1:
                self.in_means = self.in_means[self.total_variables_in//2:]
                self.in_stds = self.in_stds[self.total_variables_in//2:]
                self.out_means = self.out_means[self.total_variables_out//2:]
                self.out_stds = self.out_stds[self.total_variables_out//2:]

        # unsqueeze to correct shapes
        self.in_means = self.in_means.reshape(1,1,-1)   # self.in_means.reshape((self.in_means.shape[0], 1, 1))
        self.in_stds = self.in_stds.reshape(1,1,-1)     # self.in_stds.reshape((self.in_stds.shape[0], 1, 1))
        self.out_means = self.out_means.reshape(1,1,-1) # self.out_means.reshape((self.out_means.shape[0], 1, 1))
        self.out_stds = self.out_stds.reshape(1,1,-1)   # self.out_stds.reshape((self.out_stds.shape[0], 1, 1))
        
    def _load_constant_masks(self):
        mask_file_paths = {}
        with open(f"{self.config['data']['constant_mask_dir']}/file_paths.yaml") as f:
            mask_file_paths = yaml.load(f, Loader=yaml.SafeLoader)

        ordered_path_list = []

        for mask_name in self.config["data"]["constant_masks"]:
            full_key = f"{mask_name}_{self.resolution[0]}_{self.resolution[1]}"
            if full_key in mask_file_paths.keys():
                ordered_path_list.append(
                    (
                        mask_file_paths[full_key]["path"],
                        mask_file_paths[full_key]["normalize"],
                    )
                )
            else:
                raise ValueError(
                    f"Mask {mask_name} not supported at resolution {self.resolution[0]}x{self.resolution[1]}"
                )

        array_list = []
        for path, normalize in ordered_path_list:
            data = np.load(f"{self.config['data']['constant_mask_dir']}/{path}")

            if (
                data.shape[1] == self.resolution[0]
                and data.shape[0] == self.resolution[1]
            ):
                data = data.transpose((1, 0))

            if not (
                data.shape[0] == self.resolution[0]
                and data.shape[1] == self.resolution[1]
            ):
                raise ValueError(
                    f"Data stored at {path} does not have the required shape ({self.resolution[0]}, {self.resolution[1]})"
                )

            if normalize:
                mean = data.mean()
                std = data.std()
                data = (data - mean) / std

            array_list.append(data)

        stacked_data = np.stack(array_list, axis=0)

        self.constant_mask_data = torch.tensor(stacked_data, requires_grad=False)[
            :, : self.xlat, self.start : self.end
        ].permute(1, 2, 0)  # shape: (x_lat, x_lon, num_masks)

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor
        )

    def _get_num_variables_in(self):
        n_surface = len(self.in_surface_variables)
        n_pressure = len(self.in_pressure_variables) * len(self.in_pressure_levels)
        n_masks = self.constant_mask_data.shape[-1]
        return n_surface + n_pressure + n_masks

    def _get_num_variables_out(self):
        n_surface = len(self.out_surface_variables)
        n_pressure = len(self.out_pressure_variables) * len(self.out_pressure_levels)
        n_masks = self.constant_mask_data.shape[-1]
        return n_surface + n_pressure + n_masks

    def _getitem_distributed(self, input_surface_ds, input_pressure_ds, target_surface_ds, target_pressure_ds):
        
        if self.rank % 2 == 0:  # first half of data
            # Input data
            input_surface = input_surface_ds.values
            input_pressure = input_pressure_ds.isel(vl=slice(0, self.total_variables_in//2 - len(self.in_surface_variables))).values
            
            # TODO: manage the normalization properly
            input = np.concatenate([input_surface, input_pressure], axis=-1)
            input = (input - self.in_means) / self.in_stds
            
            # Target data
            target_surface = target_surface_ds.values
            target_pressure = target_pressure_ds.isel(vl=slice(0, self.total_variables_out//2 - len(self.out_surface_variables))).values

            target = np.concatenate([target_surface, target_pressure], axis=-1)
            target = (target - self.out_means) / self.out_stds
            target = torch.tensor(target)
        else:
            input_pressure = input_pressure_ds.isel(vl=slice(self.total_variables_in//2 - len(self.in_surface_variables), None)).values
            # Normalization
            input = (input_pressure - self.in_means) / self.in_stds
            input = np.concatenate((input_pressure, self.constant_mask_data), axis=-1)
            
            target = target_pressure_ds.isel(vl=slice(self.total_variables_out//2 - len(self.in_surface_variables), None)).values
            target = (target - self.out_means) / self.out_stds
            target = torch.nn.functional.pad(torch.tensor(target), (0, self.constant_mask_data.shape[-1]))
            
        input = torch.tensor(input)
        return input, target
    
    def _getitem_nondistributed(self, input_surface_ds, input_pressure_ds, target_surface_ds, target_pressure_ds):
        target = []

        # Input data
        input_surface = input_surface_ds.values
        input_pressure = input_pressure_ds.values
            
        input = np.concatenate([input_surface, input_pressure], axis=-1)
        input = (input - self.in_means) / self.in_stds
        
        # Target data
        target_surface = target_surface_ds.values
        target_pressure = target_pressure_ds.values

        target = np.concatenate([target_surface, target_pressure], axis=-1)
        target = (target - self.out_means) / self.out_stds
        target = torch.tensor(target)
        
        input = torch.tensor(input)
        return input, target
    
    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = (
            global_idx
            + torch.arange(1, self.forecast_length + 1)
            * step
            // self.config["data"]["target_subsampling"]
        )

        input_surface_ds = (
            self.zarr_in_surface.isel(time=global_idx)
            .isel(latitude=range(0, self.xlat), longitude=range(self.start, self.end))
            .to_array("surface_variables")
            .transpose("latitude", "longitude", "surface_variables")
        )
        
        input_pressure_ds = (
            self.zarr_in.isel(time=global_idx)
            .sel(level=self.config["data"]["in_pressure_levels"])
            .isel(latitude=range(0, self.xlat), longitude=range(self.start, self.end))
            .to_array("pressure_variables")
            .stack(vl=("pressure_variables", "level"))
            .transpose("latitude", "longitude", "vl")
        )

        target_surface_ds = (
            self.zarr_out_surface.isel(time=target_file_idxs)
            .isel(latitude=range(0, self.xlat), longitude=range(self.start, self.end))
            .to_array("surface_variables")
            .transpose("time", "latitude", "longitude", "surface_variables")
        )
        
        target_pressure_ds = (
            self.zarr_out.isel(time=target_file_idxs)
            .sel(level=self.config["data"]["out_pressure_levels"])
            .isel(latitude=range(0, self.xlat), longitude=range(self.start, self.end))
            .to_array("pressure_variables")
            .stack(vl=("pressure_variables", "level"))
            .transpose("time", "latitude", "longitude", "vl")
        )

        if self.parallelism > 1:
            input, target = self._getitem_distributed(input_surface_ds, input_pressure_ds, target_surface_ds, target_pressure_ds)
        else: # non distributed
            input, target = self._getitem_nondistributed(input_surface_ds, input_pressure_ds, target_surface_ds, target_pressure_ds)
        return input.to(torch.float32), target.to(torch.float32)

class DummyHighResDataset(Dataset):
    def __init__(self, config, nvars=72, nsamples=100):
        self.x = config["data"]["xlat"]
        self.y = config["data"]["xlon"]
        self.nvars = nvars
        self.seed = 1  # random seed for now
        self.config = config
        self.nsamples = nsamples

    def __len__(self):
        if self.config["data"]["train_subset"] == -1:
            return self.nsamples
        else:
            return self.config["data"]["train_subset"]

    def __getitem__(self, global_idx):
        return torch.rand(
            (self.nvars, self.x, self.y), dtype=torch.float32
        ), torch.rand((self.nvars, self.x, self.y), dtype=torch.float32)


class DistributedDummyHighResDataset(DummyHighResDataset):
    def __init__(self, config, nvars=72):
        super().__init__(config)
        self.parallelism = config["model_parallel"]["parallelism"]
        self.y = -(-self.y // (self.parallelism // 2))
        self.nvars = -(-nvars // 2)  # ceiling function
        self.seed = 1  # random seed for now

    def __getitem__(self, global_idx):
        return torch.rand(
            (self.x, self.y, self.nvars), dtype=torch.float32
        ), torch.rand((self.x, self.y, self.nvars), dtype=torch.float32)


class DistributedHighResDataset(ERA5PreprocessedDistributed):
    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
        normalize=True,
        dt=None,
    ):
        super().__init__(
            config,
            file_path,
            mode,
            patch_size,
            rank,
            forecast_length=1,
            normalize=True,
            dt=None,
        )

    def __getitem__(self, global_idx):
        input, target = super().__getitem__(global_idx)
        input = input.permute(2, 0, 1).unsqueeze(0)

        input_up = torch.nn.functional.interpolate(
            input, scale_factor=(2, 2), mode="bilinear", align_corners=True
        )

        for i, t in enumerate(target):
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, scale_factor=(2, 2), mode="bilinear", align_corners=True
            )
            target[i] = t.squeeze(0).permute(1, 2, 0)
        input_up = input_up.squeeze(0).permute(1, 2, 0)
        return input_up, target


class HighResDataset(ERA5Preprocessed):
    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
        normalize=True,
        dt=None,
    ):
        super().__init__(
            config,
            file_path,
            mode,
            patch_size,
            rank,
            forecast_length=1,
            normalize=True,
            dt=None,
        )

    def __getitem__(self, global_idx):
        input, target = super().__getitem__(global_idx)
        input = input.unsqueeze(0)
        input_up = torch.nn.functional.interpolate(
            input, scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        for i, t in enumerate(target):
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, scale_factor=(2, 2), mode="bilinear", align_corners=True
            )
            target[i] = t.squeeze(0).permute(1, 2, 0)
        input_up = input_up.squeeze(0)
        return input_up, target
