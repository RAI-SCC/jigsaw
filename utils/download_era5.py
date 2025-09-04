import os
import argparse as ap
import xarray as xr
import dask
import dask.array as da
from tqdm import tqdm


def main(args):
    var_list = ['geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'mean_sea_level_pressure',
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'surface_pressure', 'total_column_water_vapour', 'total_precipitation']

    level_indices = [1000,925,850,700,600,500,400,300,250,200,150,100,50]

    # Open the Zarr dataset with dask
    wb_data = xr.open_dataset('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr', engine='zarr', chunks={'time': 'auto'})
    
    for var in tqdm(var_list):
        # Subset the dataset to keep only the desired levels
        start_date = None
        if args.start_year is not None:
            start_date = f"{args.start_year}-01-01"
        end_date = None
        if args.end_year is not None:
            end_date = f"{args.end_year}-12-31"
        wb_data_subset = wb_data.sel(level=level_indices,time=slice(start_date,end_date))[var]
        wb_data_subset = wb_data_subset.chunk({"time": 1})
        wb_data_subset.to_zarr(args.output_file, mode='a', compute=True)

    return


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--output_file", type=str, help="Name of the output file (in zarr format).", required=True)
    parser.add_argument("--start_year", type=int, default=1970, help="Start year for download.", required=False)
    parser.add_argument("--end_year", type=int, default=None, help="Final year for download (inclusive).", required=False)
    args = parser.parse_args()

    main(args)

