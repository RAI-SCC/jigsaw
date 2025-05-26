# Jigsaw
Code for performing domain- and tensor-parallel training optimized for large image-based data.

## Structure of repository
- Jigsaw/ contains all source code for WeatherMixer and Jigsaw parallelization method
- Jigsaw/data contains
- cases/ contains all config files and batch submit scripts
  
##  Data
To train the model, approximately 60 TB of ERA5 data will need to be downloaded, specifically, the variables (Z, Q, T, U, V) at pressure levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50), and surface variables (MSLP, U10, V10, T2M) The current dataloader is adapted to the following two datasets downloaded from WeatherBench2 (WB2) https://weatherbench2.readthedocs.io/en/latest/data-guide.html.

1. xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
2. xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr') Dataset(1) is subsampled to a 6-hourly time resolution and to the pressure levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50) hPa. Note that the Dataset(2) has an hourly time resolution and contains all 37 pressure levels. Data can also be obtained from the Copernicus API directly, https://cds.climate.copernicus.eu/api-how-to.

Climatology values can be obtained via https://github.com/google-research/weatherbenchX

## Environment
The correct Python environment can be obtained with
```
pip install -r requirements.txt
```
Set the path to the source code as an environment variable.
```
SRC='path/to/src/Jigsaw'
```
